"""Measured result summaries, plots and annotated examples."""
from pathlib import Path

from .common import write_json
from .inference import degrade, load_rgb
from .metrics import match_image


def markdown_table(rows, columns):
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key)
            values.append("N/A" if value is None else f"{value:.4f}" if isinstance(value,float) else str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def make_figures(experiment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    output = experiment.output / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8,5))
    for row in experiment.validation_rows:
        ax.scatter(row["pipeline_median_ms"],row["f1"],s=75,color="#167a8b" if row["size"] == 512 else "#e08a32")
        ax.annotate(f"{row['size']} / {row['threshold']}",(row["pipeline_median_ms"],row["f1"]),xytext=(5,6),textcoords="offset points",fontsize=9)
    ax.set(xlabel="Memory-pipeline latency, median (ms/image)",ylabel="F1 @ IoU 0.5",title="Validation: accuracy vs latency (D0, FP32, batch 1)",ylim=(-.03,1.1))
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "validation_tradeoff.png",bbox_inches="tight")
    plt.close(fig)
    conditions = experiment.config["conditions"]
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    import numpy as np
    for offset, policy, color in [(-.18,"baseline","#7598ac"),(.18,"selected","#167a8b")]:
        rows = [next(r for r in experiment.test_rows if r["policy"] == policy and r["condition"] == condition) for condition in conditions]
        for ax,metric in zip(axes,["f1","recall"]):
            ax.bar(np.arange(len(conditions))+offset,[r[metric] for r in rows],width=.34,label=policy,color=color)
            ax.set(xticks=np.arange(len(conditions)),xticklabels=conditions,ylim=(0,1.1),title=f"Test {metric}: fixed settings")
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "test_degradation.png",bbox_inches="tight")
    plt.close(fig)
    selected = next(a for a in experiment.test_artifacts if a["policy"] == "selected" and a["condition"] == "original")
    ordered = sorted(selected["details"],key=lambda d: (-(d["fn"]+d["fp"]),d["image_id"]))
    # Pick both errors and a successful/easier example; record why each was shown.
    picked = list(dict.fromkeys([d["image_id"] for d in ordered[:2]] + [ordered[-1]["image_id"]]))
    selection_log = []
    for number, image_id in enumerate(picked,1):
        record = next(r for r in experiment.test_records if r["id"] == image_id)
        fig, axes = plt.subplots(1,len(conditions),figsize=(5*len(conditions),4),squeeze=False)
        for ax, condition in zip(axes[0],conditions):
            artifact = next(a for a in experiment.test_artifacts if a["policy"] == "selected" and a["condition"] == condition)
            canvas = Image.fromarray(degrade(load_rgb(record),condition,experiment.config))
            draw = ImageDraw.Draw(canvas)
            for box in record["boxes"]:
                draw.rectangle(box,outline=(0,220,100),width=2)
            matches, missed = match_image(record["boxes"],artifact["predictions"][image_id],artifact["threshold"],experiment.config["match_iou"],experiment.config["max_detections"])
            for match in matches:
                p=match["prediction"]
                draw.rectangle(p["box"],outline=(255,110,20),width=2)
                draw.text((max(0,p["box"][0]),max(0,p["box"][1]-12)),f"{p['score']:.2f}",fill=(255,110,20))
            ax.imshow(canvas)
            ax.axis("off")
            ax.set_title(f"{condition} | GT {len(record['boxes'])} / pred {len(matches)} / missed {len(missed)}")
        fig.suptitle(f"Example {number} | green: ground truth, orange: prediction",fontsize=12)
        fig.tight_layout()
        fig.savefig(output/f"example_{number}.png",bbox_inches="tight")
        plt.close(fig)
        selection_log.append({"figure": f"example_{number}.png", "image_id": image_id,"source":record["relative_path"],"selection":"two highest original FP+FN errors and one lowest; descriptive only"})
    write_json(output / "example_sources.json",selection_log)


def create_report(experiment, figures=True):
    if figures:
        make_figures(experiment)
    baseline = next(r for r in experiment.test_rows if r["policy"] == "baseline" and r["condition"] == "original")
    selected = next(r for r in experiment.test_rows if r["policy"] == "selected" and r["condition"] == "original")
    delta = selected["f1"] - baseline["f1"]
    latency_ratio = baseline["pipeline_median_ms"] / selected["pipeline_median_ms"]
    lines = ["# EfficientDet 차량 탐지 실험 결과", "", "실제 실행 로그에서 생성한 결과 표와 관찰 기록입니다.", "",
             f"- 검증 {len(experiment.validation)}장 / 최종 평가 {len(experiment.test_records)}장",
             f"- 검증으로 선택한 설정: 입력 {selected['size']}, confidence {selected['threshold']}",
             f"- 원본 최종 평가 F1: 기본 {baseline['f1']:.4f} → 선택 {selected['f1']:.4f} (차이 {delta:+.4f})",
             f"- 기본/선택 처리 시간 비율: {latency_ratio:.3f}배. 1보다 크면 선택 설정이 이 측정에서 더 빠릅니다.",
             "- 개선되지 않은 결과도 그대로 보고합니다. 설정 선택에는 최종 평가를 사용하지 않았습니다.", "",
             "## 검증 결과", "", markdown_table(experiment.validation_rows,["size","threshold","precision","recall","f1","ap50","pipeline_median_ms","pipeline_p95_ms","count_mae"]), "",
             "AP50은 낮은 공통 score cutoff에서 계산하므로 같은 해상도의 임계값 행에서 동일합니다. 임계값 최적화 효과는 F1·Precision·Recall로 비교합니다.", "",
             "## 최종 평가", "", markdown_table(experiment.test_rows,["policy","condition","size","threshold","precision","recall","f1","ap50","coco_ap50_check","pipeline_median_ms","count_mae","small_gt","small_recall"]), "",
             "## 영상 열화에 대한 관찰", ""]
    for condition in experiment.config["conditions"]:
        if condition == "original":
            continue
        row = next(r for r in experiment.test_rows if r["policy"] == "selected" and r["condition"] == condition)
        lines.append(f"- {condition}: 원본 대비 F1 차이 {row['f1']-selected['f1']:+.4f}, Recall 차이 {row['recall']-selected['recall']:+.4f}. 같은 이미지의 변형본을 짝 비교했습니다.")
    lines += ["", "## 측정 방법과 해석 범위", "",
              "- 모델: COCO 사전학습 EfficientDet-D0, FP32, batch 1. 추가 학습 없음. 정확도 평가는 car 단일 클래스.",
              f"- AP score floor={experiment.config['ap_score_floor']}, NMS IoU={experiment.config['nms_iou']}, 정답 대응 IoU={experiment.config['match_iou']}, max detections={experiment.config['max_detections']}.",
              "- 단일 클래스 AP50을 계산하고 pycocotools와 대조합니다. AP@[.5:.95]로 표기하지 않습니다.",
              "- 속도: GPU 준비 실행 후 전처리·CPU→GPU 전송·추론·NMS·원본 좌표 복원·CPU 결과 변환을 포함합니다. CUDA 동기화를 사용합니다.",
              "- 디스크 읽기·합성 열화 생성·그림 그리기·영상 디코딩은 속도 측정에서 제외합니다. memory_pipeline_fps는 실서비스 영상 FPS가 아닙니다.",
              "- 상대 면적 1% 이하를 작은 차량으로 정의한 보조 Recall을 표시합니다. COCO의 small 정의와 다릅니다. 분모가 없으면 N/A입니다.",
              "- 합성 어두움·Gaussian blur는 실제 야간·우천·주행 모션블러와 다릅니다. 공장 출입구 적용을 가정한 공개 차량 데이터 예비 실험입니다.",
              f"- 분할 한계: {experiment.audit['split']['limitation']}",
              "- 표본 수가 작고 동일 카메라 장면일 수 있습니다. 일반화 또는 통계적 유의성을 주장하지 않습니다.", ""]
    if figures:
        lines += ["## 그림", "", "![검증 정확도와 속도](figures/validation_tradeoff.png)", "", "![최종 평가 열화 비교](figures/test_degradation.png)", ""]
    lines += ["## 해석과 KPT", "", "같은 실행의 해석과 KPT는 `PERSONAL_REFLECTION.md`에 저장합니다.", "",
              "## 재현 자료", "", "`config.json`, `data_audit.json`, `split_manifest.json`, `model_provenance.json`, `environment.json`, `selection.json`, `timings/`를 함께 보존합니다.", "",
              "데이터 출처: https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset", "모델 출처: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch", ""]
    path = experiment.output / "REPORT.md"
    path.write_text("\n".join(lines),encoding="utf-8")
    return path
