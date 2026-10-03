"""Classify one local image with a Hub model using Transformers pipeline().

The default is the same pinned DeiT model used in notebooks 00 and 01.
Model files download to the Hugging Face cache when they are not cached yet.
"""
import argparse
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import tempfile


MODEL_ID = "facebook/deit-tiny-patch16-224"
MODEL_REVISION = "b3428f18dcc7b543470d07f14b4a4157815d1880"


def select_device(torch_module, requested):
    """GPU demonstrations must fail instead of silently falling back to CPU."""
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("GPU가 없습니다. Colab GPU를 연결하거나 --device cpu를 명시하세요.")
        return "cuda:0", torch_module.cuda.get_device_name(0)
    if requested == "cpu":
        return "cpu", "CPU"
    raise ValueError("device는 cuda 또는 cpu여야 합니다.")


def create_classifier(model_id, revision, device, *, pipeline_factory=None, processor_factory=None):
    """Explicit pipeline configuration; no loading happens when importing this file."""
    if pipeline_factory is None or processor_factory is None:
        from transformers import AutoImageProcessor, pipeline

        pipeline_factory = pipeline if pipeline_factory is None else pipeline_factory
        processor_factory = AutoImageProcessor.from_pretrained if processor_factory is None else processor_factory
    image_processor = processor_factory(
        model_id, revision=revision, use_fast=False, trust_remote_code=False,
    )
    model_kwargs = {"weights_only": True, "attn_implementation": "eager"}
    # This exact course checkpoint has pytorch_model.bin, not model.safetensors.
    if model_id == MODEL_ID and revision == MODEL_REVISION:
        model_kwargs["use_safetensors"] = False
    return pipeline_factory(
        "image-classification", model=model_id, revision=revision,
        image_processor=image_processor, framework="pt", device=device,
        trust_remote_code=False, model_kwargs=model_kwargs,
    )


def validate_predictions(predictions, top_k):
    """Reject malformed results before labelling a run completed."""
    if not isinstance(predictions, list) or len(predictions) != top_k:
        raise ValueError("pipeline 결과가 요청한 top-k 개수와 다릅니다.")
    result = []
    for item in predictions:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not item["label"]:
            raise ValueError("예측 결과에 label 문자열이 필요합니다.")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("예측 score는 숫자여야 합니다.")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("예측 score는 유한한 0~1 값이어야 합니다.")
        result.append({"label": item["label"], "score": float(score)})
    if any(first["score"] < second["score"] for first, second in zip(result, result[1:])):
        raise ValueError("예측 결과가 score 내림차순이 아닙니다.")
    return result


def save_new_json(path, report):
    """Publish a complete JSON atomically without replacing an earlier result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as file:
            temporary = Path(file.name)
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        # A hard link creates the final name only if it does not exist, even in a race.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_inference(args):
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"기존 결과를 보존합니다. 새 --output 경로를 사용하세요: {output}")
    if args.top_k < 1:
        raise ValueError("top-k는 1 이상이어야 합니다.")
    if not args.run_id.strip():
        raise ValueError("run-id는 비어 있을 수 없습니다.")

    import torch
    from PIL import Image

    device, device_name = select_device(torch, args.device)
    image_bytes = Path(args.image).read_bytes()
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")
    classifier = create_classifier(args.model_id, args.revision, device)
    if args.top_k > classifier.model.config.num_labels:
        raise ValueError("top-k가 모델의 분류 항목 수보다 큽니다.")
    predictions = validate_predictions(classifier(image, top_k=args.top_k), args.top_k)
    report = {
        "status": "completed", "task": "image-classification", "run_id": args.run_id,
        "model_id": args.model_id, "model_revision": args.revision,
        "device": args.device, "device_name": device_name,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "predictions": predictions,
        "versions": {name: version(name) for name in ("torch", "transformers", "huggingface-hub")},
    }
    save_new_json(output, report)
    print(f"장치: {device_name} · 모델: {args.model_id}")
    for item in predictions:
        print(f"{item['score']:7.2%}  {item['label']}")
    print("HF_PIPELINE_INFERENCE_COMPLETE", output)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="로컬 이미지 파일")
    parser.add_argument("--output", required=True, type=Path, help="새 JSON 결과 경로 (덮어쓰기 없음)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION, help="모델을 바꾸면 그 모델의 커밋도 지정하세요")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_inference(parse_args())
