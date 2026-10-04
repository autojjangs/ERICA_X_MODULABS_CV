"""Small, explicit filesystem helpers shared by the notebook and CLI."""
import hashlib
import json
import os
import platform
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def download(url, target):
    """Download only when needed; interrupted files never become final weights."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return target
    temporary = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "vehicle-course-project/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    if temporary.stat().st_size < 1024:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {temporary}")
    os.replace(temporary, target)
    return target


def environment_info():
    import importlib.metadata
    versions = {}
    for name in ["torch", "torchvision", "numpy", "opencv-python-headless", "Pillow", "matplotlib", "pandas", "kagglehub", "pycocotools"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"created_utc": utc_now(), "python": sys.version, "platform": platform.platform(), "packages": versions}


def validate_config(config):
    if any(s <= 0 or s % 128 for s in config["sizes"]):
        raise ValueError("EfficientDet-D0 input sizes must be positive multiples of 128.")
    if config["baseline"]["size"] not in config["sizes"]:
        raise ValueError("Baseline resolution must be in sizes.")
    if config["baseline"]["threshold"] not in config["thresholds"]:
        raise ValueError("Baseline confidence must be in thresholds.")
    if not 0 < config["ap_score_floor"] < min(config["thresholds"]) <= max(config["thresholds"]) < 1:
        raise ValueError("Require 0 < AP score floor < operating thresholds < 1.")
    for name in ["nms_iou", "match_iou"]:
        if not 0 < config[name] <= 1:
            raise ValueError(name)
    if config["match_iou"] != 0.5:
        raise ValueError("This project reports AP50/F1@0.5; match_iou must remain 0.5.")
    for name in ["validation_images", "test_images", "warmup", "timing_images", "timing_repeats", "max_detections"]:
        if config[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if config["blur_kernel"] % 2 != 1 or config["blur_kernel"] < 1:
        raise ValueError("blur_kernel must be a positive odd number.")
    if not 0 < config["dark_factor"] <= 1 or config["blur_sigma"] <= 0:
        raise ValueError("Invalid corruption parameters.")
    if "original" not in config["conditions"] or not set(config["conditions"]) <= {"original", "dark", "blur"}:
        raise ValueError("Unknown/missing image condition.")
    if config["frame_gap"] < 0 or config["experiment_budget_minutes"] <= 0:
        raise ValueError("Invalid gap or time budget.")
