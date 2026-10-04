"""D0 inference with consistent RGB preprocessing and explicit CUDA timing."""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .common import download, sha256


UPSTREAM = "https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git"
WEIGHTS_URL = "https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d0.pth"


def install_model(project_root, config):
    root = Path(project_root)
    repo = root / "vendor" / "efficientdet"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", UPSTREAM, str(repo)], check=True)
    if not (repo / "backbone.py").is_file():
        raise RuntimeError(f"Incomplete upstream clone: {repo}")
    if config["upstream_ref"] != "master":
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", config["upstream_ref"]], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    weight_path = download(WEIGHTS_URL, root / "weights" / "efficientdet-d0.pth")
    return repo, weight_path, {"upstream_url": UPSTREAM, "upstream_commit": commit, "weights_url": WEIGHTS_URL, "weights_sha256": sha256(weight_path)}


def load_rgb(record):
    from PIL import Image
    with Image.open(record["path"]) as image:
        return np.asarray(image.convert("RGB")).copy()


def degrade(rgb, condition, config):
    if condition == "original":
        return rgb.copy()
    if condition == "dark":
        return np.clip(rgb.astype(np.float32) * config["dark_factor"], 0, 255).astype(np.uint8)
    if condition == "blur":
        import cv2
        return cv2.GaussianBlur(rgb, (config["blur_kernel"], config["blur_kernel"]), config["blur_sigma"])
    raise ValueError(f"Unknown condition: {condition}")


class Detector:
    # The upstream checkpoint has 90 COCO slots, not a compact 80-class list.
    CAR_INDEX = 2

    def __init__(self, project_root, config):
        import torch
        import torchvision
        from torchvision.ops import nms
        self.torch, self.nms, self.config = torch, nms, config
        if config["require_cuda"] and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU가 없습니다. VS Code에서 Colab GPU 커널을 선택한 뒤 다시 실행하세요. CPU 실험은 require_cuda=false로 명시해야 합니다.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Fail early on the common torch/torchvision binary mismatch.
        nms(torch.tensor([[0.,0.,10.,10.]], device=self.device), torch.tensor([.9], device=self.device), .5)
        repo, weights, self.provenance = install_model(project_root, config)
        sys.path.insert(0, str(repo))
        from backbone import EfficientDetBackbone
        from efficientdet.utils import BBoxTransform, ClipBoxes
        self.model = EfficientDetBackbone(compound_coef=0, num_classes=90)
        # Never fall back to arbitrary checkpoint pickle execution.
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.decode, self.clip = BBoxTransform(), ClipBoxes()
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.benchmark = False
        self.provenance.update({"torch": torch.__version__, "torchvision": torchvision.__version__, "device": str(self.device),
                               "gpu_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
                               "precision": "float32", "target_score": "COCO slot 2 (car), independent class score", "cudnn_benchmark": False})

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

    def preprocess(self, rgb, size):
        import cv2
        height, width = rgb.shape[:2]
        ratio = size / max(height, width)
        new_w, new_h = max(1, int(width*ratio)), max(1, int(height*ratio))
        mean = np.array([.485,.456,.406], dtype=np.float32)
        std = np.array([.229,.224,.225], dtype=np.float32)
        normalized = (rgb.astype(np.float32)/255 - mean)/std
        resized = cv2.resize(normalized, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size,size,3), dtype=np.float32)
        canvas[:new_h,:new_w] = resized
        tensor = self.torch.from_numpy(canvas).permute(2,0,1).unsqueeze(0).contiguous().to(self.device)
        return tensor, (new_w/width, new_h/height, width, height)

    def predict(self, rgb, size, score_floor=None):
        score_floor = self.config["ap_score_floor"] if score_floor is None else score_floor
        torch = self.torch
        self.sync()
        start = time.perf_counter()
        tensor, (scale_x,scale_y,width,height) = self.preprocess(rgb, size)
        self.sync()
        forward_start = time.perf_counter()
        with torch.inference_mode():
            _, regression, classification, anchors = self.model(tensor)
            self.sync()
            forward_end = time.perf_counter()
            boxes = self.clip(self.decode(anchors, regression), tensor)[0]
            scores = classification[0,:,self.CAR_INDEX]
            valid = (scores >= score_floor) & torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
            boxes, scores = boxes[valid], scores[valid]
            keep = self.nms(boxes, scores, self.config["nms_iou"])[:self.config["max_detections"]]
            boxes, scores = boxes[keep], scores[keep]
            boxes[:,[0,2]] /= scale_x
            boxes[:,[1,3]] /= scale_y
            boxes[:,[0,2]] = boxes[:,[0,2]].clamp(0,width)
            boxes[:,[1,3]] = boxes[:,[1,3]].clamp(0,height)
            valid = (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])
            boxes, scores = boxes[valid].cpu().numpy(), scores[valid].cpu().numpy()
        predictions = [{"box": box.astype(float).tolist(), "score": float(score)} for box,score in zip(boxes,scores)]
        self.sync()
        end = time.perf_counter()
        return predictions, {"pipeline_ms": (end-start)*1000, "forward_ms": (forward_end-forward_start)*1000}

    def benchmark(self, images, size, threshold):
        # Images and artificial degradations are prepared before this timer.
        for i in range(self.config["warmup"]):
            self.predict(images[i % len(images)], size, threshold)
        samples = []
        chosen = images[:self.config["timing_images"]]
        for _ in range(self.config["timing_repeats"]):
            for rgb in chosen:
                _, timing = self.predict(rgb, size, threshold)
                samples.append(timing)
        pipeline = np.array([s["pipeline_ms"] for s in samples])
        forward = np.array([s["forward_ms"] for s in samples])
        return {"timing_runs": len(samples), "pipeline_mean_ms": float(pipeline.mean()), "pipeline_median_ms": float(np.median(pipeline)),
                "pipeline_p95_ms": float(np.percentile(pipeline,95)), "forward_median_ms": float(np.median(forward)),
                "memory_pipeline_fps": float(1000/pipeline.mean()), "timing_samples": samples}
