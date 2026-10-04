"""Offline fixtures verify orchestration only. They are NOT EfficientDet results."""
import json
import unittest
from pathlib import Path
from PIL import Image
from vehicle_project.common import read_json
from vehicle_project.experiment import Experiment
from test_helpers import test_directory


class FixtureDetector:
    provenance={"upstream_commit":"offline-test-fixture","device":"fixture"}
    def predict(self,rgb,size,score_floor=None):
        return [{"box":[0,0,10,10],"score":.9}],{"pipeline_ms":2.,"forward_ms":1.}
    def benchmark(self,images,size,threshold):
        return {"timing_runs":1,"pipeline_mean_ms":float(size),"pipeline_median_ms":float(size),"pipeline_p95_ms":float(size),"forward_median_ms":1.,"memory_pipeline_fps":1000/size,"timing_samples":[{"pipeline_ms":float(size),"forward_ms":1.}]}


class ExperimentTests(unittest.TestCase):
    def test_stage_order_selection_and_export(self):
        source=Path(__file__).resolve().parents[1]
        config=read_json(source/"configs/experiment.json")
        config.update({"verify_coco_ap":False,"conditions":["original","dark"],"require_cuda":False})
        with test_directory() as temporary:
            root=Path(temporary)
            image=root/"fixture.png"
            Image.new("RGB",(20,20),(100,100,100)).save(image)
            experiment=Experiment(root,config,FixtureDetector())
            with self.assertRaises(RuntimeError):
                experiment.run_test()
            record={"id":"v","path":str(image),"relative_path":"fixture.png","width":20,"height":20,"boxes":[[0,0,10,10]]}
            experiment.validation=[record]
            experiment.test_records=[dict(record,id="t")]
            experiment.audit={"split":{"limitation":"OFFLINE SYNTHETIC FIXTURE ONLY"}}
            experiment.load_model()
            rows,selection=experiment.run_validation()
            self.assertEqual(len(rows),6)
            self.assertEqual(selection["size"],512)
            self.assertEqual(selection["threshold"],.1)
            frozen=json.dumps(selection,sort_keys=True)
            results=experiment.run_test()
            self.assertEqual(len(results),4)
            self.assertEqual(json.dumps(experiment.selection,sort_keys=True),frozen)
            report=experiment.write_report(figures=False)
            self.assertIn("OFFLINE SYNTHETIC FIXTURE ONLY",report.read_text(encoding="utf-8"))
            self.assertTrue(experiment.export().is_file())
            self.assertTrue(read_json(experiment.output/"run_status.json")["measured"])


if __name__ == "__main__":
    unittest.main()
