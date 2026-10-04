"""Stage-based experiment, shared by notebook and command line."""
import csv
import json
import random
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .common import environment_info, read_json, utc_now, validate_config, write_json
from .data import prepare
from .inference import Detector, degrade, load_rgb
from .metrics import ap50, coco_ap50, evaluate


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


class Experiment:
    def __init__(self, project_root, config, detector=None):
        self.root = Path(project_root).resolve()
        self.config = json.loads(json.dumps(config))
        validate_config(self.config)
        self.started = time.perf_counter()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.output = self.root / "results" / run_id
        self.output.mkdir(parents=True)
        self.detector = detector
        self.validation = self.test_records = None
        self.validation_rows, self.test_rows = [], []
        self.test_artifacts = []
        self.selection = None
        self.audit = None
        self.state = {"started_utc": utc_now(), "status": "created", "measured": False}
        write_json(self.output / "config.json", self.config)
        write_json(self.output / "environment.json", environment_info())
        self.checkpoint("created")

    def checkpoint(self, status):
        self.state.update({"status": status, "elapsed_seconds": time.perf_counter()-self.started, "updated_utc": utc_now()})
        write_json(self.output / "run_status.json", self.state)

    def check_budget(self):
        if time.perf_counter()-self.started > 60*self.config["experiment_budget_minutes"]:
            self.checkpoint("time_budget_exceeded_partial_results")
            raise TimeoutError("실험 시간 한도를 넘었습니다. 현재 results 폴더의 부분 결과를 보존했습니다. 설정·샘플 수를 줄인 새 실행을 시작하세요.")

    def prepare_data(self):
        self.validation, self.test_records, self.audit = prepare(self.config, self.output)
        version = self.audit["acquisition"].get("resolved_version_directory", "")
        if version.isdigit() and "/versions/" not in self.config["dataset_handle"]:
            self.config["dataset_handle"] += f"/versions/{version}"
        write_json(self.output / "config.json", self.config)
        self.checkpoint("data_ready")
        return self.audit

    def load_model(self):
        self.detector = self.detector or Detector(self.root, self.config)
        write_json(self.output / "model_provenance.json", self.detector.provenance)
        self.config["upstream_ref"] = self.detector.provenance["upstream_commit"]
        write_json(self.output / "config.json", self.config)
        self.checkpoint("model_ready")
        return self.detector.provenance

    def predictions_for(self, records, size, condition, split_name):
        if self.detector is None:
            raise RuntimeError("Call load_model first.")
        predictions = {}
        # Accuracy predictions use the common low score floor. They are not latency samples.
        for index, record in enumerate(records):
            self.check_budget()
            rgb = degrade(load_rgb(record), condition, self.config)
            predictions[record["id"]], _ = self.detector.predict(rgb, size, self.config["ap_score_floor"])
            if index == 0 or (index+1) % 20 == 0 or index+1 == len(records):
                print(f"{split_name} / {condition} / {size}: {index+1}/{len(records)}", flush=True)
        write_json(self.output / "predictions" / f"{split_name}_{size}_{condition}.json", predictions)
        return predictions

    def ap_values(self, records, predictions):
        measured_ap = ap50(records, predictions, self.config["match_iou"], self.config["max_detections"])
        official = None
        if self.config.get("verify_coco_ap", True):
            official = coco_ap50(records, predictions, self.config["max_detections"])
            if self.config["match_iou"] == .5 and abs(measured_ap-official) > 1e-5:
                raise RuntimeError(f"AP50 evaluator disagreement: own={measured_ap}, pycocotools={official}; inspect tied scores/annotations before reporting.")
        return {"ap50": measured_ap, "coco_ap50_check": official}

    def benchmark(self, records, size, threshold, condition):
        self.check_budget()
        images = [degrade(load_rgb(r), condition, self.config) for r in records[:self.config["timing_images"]]]
        result = self.detector.benchmark(images, size, threshold)
        samples = result.pop("timing_samples")
        name = f"timing_{size}_{threshold}_{condition}_{len(records)}_{time.time_ns()}.json"
        write_json(self.output / "timings" / name, samples)
        return result

    def run_validation(self):
        if not self.validation:
            raise RuntimeError("Call prepare_data first.")
        self.checkpoint("validation_running")
        sizes = list(self.config["sizes"])
        random.Random(self.config["seed"]).shuffle(sizes)
        rows = []
        for size in sizes:
            predictions = self.predictions_for(self.validation, size, "original", "validation")
            ap = self.ap_values(self.validation, predictions)
            thresholds = list(self.config["thresholds"])
            random.Random(self.config["seed"]+size).shuffle(thresholds)
            for threshold in thresholds:
                metrics, _ = evaluate(self.validation, predictions, threshold, self.config["match_iou"], self.config["max_detections"])
                latency = self.benchmark(self.validation, size, threshold, "original")
                rows.append({"split": "validation", "condition": "original", "size": size, "threshold": threshold, **metrics, **ap, **latency})
                write_csv(self.output / "validation.csv", rows)
        self.validation_rows = sorted(rows, key=lambda r: (r["size"],r["threshold"]))
        # Selection uses validation only. Test images never influence this decision.
        best = min(rows, key=lambda r: (-r["f1"],r["pipeline_median_ms"],r["size"],r["threshold"]))
        self.selection = {"size": best["size"], "threshold": best["threshold"], "validation_f1": best["f1"],
                          "rule": "maximum validation F1; exact tie: lowest median memory-pipeline latency", "frozen_utc": utc_now()}
        write_json(self.output / "selection.json", self.selection)
        self.checkpoint("selection_frozen_before_test")
        return self.validation_rows, self.selection

    def run_test(self):
        if self.selection is None or not self.test_records:
            raise RuntimeError("Finish validation and freeze selection first.")
        self.checkpoint("test_running")
        policies = [("baseline", self.config["baseline"]), ("selected", self.selection)]
        prediction_cache, metric_cache = {}, {}
        rows, artifacts = [], []
        for condition in self.config["conditions"]:
            for label, policy in policies:
                size, threshold = policy["size"], policy["threshold"]
                key = (size,condition)
                if key not in prediction_cache:
                    prediction_cache[key] = self.predictions_for(self.test_records, size, condition, "test")
                predictions = prediction_cache[key]
                metric_key = (size,threshold,condition)
                if metric_key not in metric_cache:
                    metrics, details = evaluate(self.test_records, predictions, threshold, self.config["match_iou"], self.config["max_detections"])
                    ap = self.ap_values(self.test_records, predictions)
                    latency = self.benchmark(self.test_records,size,threshold,condition)
                    metric_cache[metric_key] = ({**metrics,**ap,**latency}, details)
                measured, details = metric_cache[metric_key]
                rows.append({"split": "test", "policy": label, "condition": condition, "size": size, "threshold": threshold, **measured})
                artifacts.append({"policy": label, "condition": condition, "size": size, "threshold": threshold, "predictions": predictions, "details": details})
                write_csv(self.output / "test.csv", rows)
                write_json(self.output / f"details_{label}_{condition}.json", details)
        self.test_rows, self.test_artifacts = rows, artifacts
        self.state["measured"] = True
        self.checkpoint("measurements_complete")
        return rows

    def write_report(self, figures=True):
        if not self.test_rows:
            raise RuntimeError("No completed test experiment to report.")
        from .report import create_report
        result = create_report(self, figures=figures)
        self.checkpoint("report_ready_personal_reflection_and_publication_pending")
        return result

    def export(self):
        """An explicit results archive; excludes dataset images and model weights."""
        path = self.root / "results" / f"vehicle_results_{self.output.name}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(self.output.rglob("*")):
                if p.is_file():
                    archive.write(p, f"results/{self.output.name}/{p.relative_to(self.output).as_posix()}")
        return path


def run_all(root, config):
    experiment = Experiment(root, config)
    try:
        experiment.prepare_data()
        experiment.load_model()
        experiment.run_validation()
        experiment.run_test()
        experiment.write_report()
        archive = experiment.export()
        print(f"완료: {archive}", flush=True)
        return experiment
    except Exception as error:
        experiment.state["error"] = f"{type(error).__name__}: {error}"
        experiment.checkpoint("failed_partial_results_preserved")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/experiment.json")
    args = parser.parse_args()
    run_all(args.root, read_json(args.config))
