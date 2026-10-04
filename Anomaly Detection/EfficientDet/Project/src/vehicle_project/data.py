"""Discover the course dataset, validate COCO boxes, and freeze evaluation splits."""
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .common import read_json, sha256, write_json


def source_name(filename):
    # Roboflow's hash is not a new scene: keep derived copies together.
    name = Path(filename).name.split(".rf.")[0]
    return re.sub(r"_(jpg|jpeg|png)$", "", name, flags=re.I)


def frame_number(filename):
    match = re.search(r"frame[_-](\d+)", source_name(filename), re.I)
    return int(match.group(1)) if match else None


def acquire(config):
    if config.get("dataset_root"):
        root = Path(config["dataset_root"]).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset_root does not exist on this runtime: {root}")
        return root, {"method": "existing_directory", "handle": config["dataset_handle"], "cache_directory_name": root.name}
    import kagglehub
    print("공개 차량 데이터 다운로드 중입니다. 인증이 필요하면 오류 안내에 따라 직접 로그인하세요.", flush=True)
    root = Path(kagglehub.dataset_download(config["dataset_handle"])).resolve()
    return root, {"method": "kagglehub", "handle": config["dataset_handle"], "resolved_version_directory": root.name}


def discover_annotations(root, config):
    search_root = Path(config["annotation_root"]).expanduser().resolve() if config.get("annotation_root") else root
    if not search_root.is_dir():
        raise FileNotFoundError(search_root)
    if not search_root.resolve().is_relative_to(root.resolve()):
        raise ValueError("annotation_root must be inside dataset_root.")
    files = sorted(search_root.rglob("*.json"))
    candidates = [p for p in files if p.name == "_annotations.coco.json" or p.name.startswith("instances_")]
    if not config.get("annotation_root") and config.get("dataset_variant"):
        variant = config["dataset_variant"].lower()
        candidates = [p for p in candidates if variant in [part.lower() for part in p.parts]]
    if not candidates:
        sample = [str(p.relative_to(search_root)) for p in files[:20]]
        raise ValueError("COCO annotation files not found. Set annotation_root to the selected COCO version directory. " + str(sample))
    # Reject mixing two Roboflow versions. A single version's train/valid/test are OK.
    version_dirs = {str(p.parent.parent) for p in candidates if p.name == "_annotations.coco.json"}
    if len(version_dirs) > 1:
        raise ValueError("Multiple COCO versions found. Set annotation_root to ONE directory: " + str(sorted(version_dirs)))
    return candidates


def load_records(root, annotation_files, target_names):
    targets = {name.lower() for name in target_names}
    records, categories, rejected = [], Counter(), []
    seen_origins, seen_contents = set(), set()
    dropped_duplicates = 0
    image_index = None
    resolved_names = set()
    for annotation_path in annotation_files:
        coco = read_json(annotation_path)
        class_names = {c["id"]: c["name"] for c in coco["categories"]}
        image_ids = {image["id"] for image in coco["images"]}
        if len(image_ids) != len(coco["images"]) or len(class_names) != len(coco["categories"]):
            raise ValueError(f"Duplicate COCO image/category IDs in {annotation_path.name}")
        selected = {i for i, name in class_names.items() if name.lower() in targets}
        resolved_names.update(class_names[i] for i in selected)
        by_image = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["image_id"] not in image_ids:
                raise ValueError(f"Annotation references an unknown image in {annotation_path.name}")
            if ann["category_id"] not in class_names:
                raise ValueError(f"Unknown category ID in {annotation_path.name}")
            categories[class_names[ann["category_id"]]] += 1
            if ann["category_id"] in selected:
                if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                    raise ValueError("Crowd/ignore target annotations need a different evaluator; do not silently drop them.")
                by_image[ann["image_id"]].append(ann)
        for item in coco["images"]:
            path = annotation_path.parent / item["file_name"]
            if not path.is_file():
                if image_index is None:
                    image_index = defaultdict(list)
                    for p in root.rglob("*"):
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            image_index[p.name].append(p)
                options = image_index[Path(item["file_name"]).name]
                if len(options) != 1:
                    raise FileNotFoundError(f"Ambiguous/missing image: {item['file_name']}; candidates={len(options)}")
                path = options[0]
            origin = source_name(item["file_name"])
            with Image.open(path) as image:
                width, height = image.size
                if (width, height) != (item["width"], item["height"]):
                    raise ValueError(f"Image/annotation size mismatch: {item['file_name']}")
                image.load()
                # Pixel hash also catches identical images saved under different names.
                pixel_hash = hashlib.sha256(image.convert("RGB").tobytes() + f"{width}x{height}".encode()).hexdigest()
            if origin in seen_origins or pixel_hash in seen_contents:
                dropped_duplicates += 1
                continue
            seen_origins.add(origin)
            seen_contents.add(pixel_hash)
            boxes = []
            for ann in by_image[item["id"]]:
                x, y, w, h = map(float, ann["bbox"])
                import math
                if not all(math.isfinite(v) for v in (x,y,w,h)) or w <= 0 or h <= 0:
                    raise ValueError(f"Invalid box in {item['file_name']}: {ann['bbox']}")
                x1, y1 = max(0., x), max(0., y)
                x2, y2 = min(float(width), x+w), min(float(height), y+h)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Box entirely outside image: {item['file_name']}")
                if [x1,y1,x2,y2] != [x,y,x+w,y+h]:
                    rejected.append({"image": item["file_name"], "action": "clipped_to_image"})
                boxes.append([x1,y1,x2,y2])
            relative = path.relative_to(root).as_posix()
            records.append({"id": hashlib.sha256(relative.encode()).hexdigest()[:16], "path": str(path), "relative_path": relative,
                            "filename": item["file_name"], "origin": origin, "frame": frame_number(origin),
                            "width": width, "height": height, "boxes": boxes, "pixel_sha256": pixel_hash, "file_sha256": sha256(path)})
    if not resolved_names or not sum(len(r["boxes"]) for r in records):
        raise ValueError(f"No target boxes for {target_names}. Annotated category counts: {dict(categories)}")
    return records, {"category_annotation_counts_before_dedup": dict(categories), "target_category_names": sorted(resolved_names), "duplicates_removed": dropped_duplicates, "box_repairs": rejected,
                     "unique_images": len(records), "negative_images": sum(not r["boxes"] for r in records)}


def split_records(records, config):
    if len(records) < 4:
        raise ValueError("At least four unique images are required for two disjoint splits.")
    rng = random.Random(config["seed"])
    frame_based = all(r["frame"] is not None for r in records)
    if frame_based:
        frame_values = sorted({r["frame"] for r in records})
        pivot = frame_values[len(frame_values)//2]
        validation_pool = [r for r in records if r["frame"] < pivot]
        test_pool = [r for r in records if r["frame"] >= pivot + config["frame_gap"]]
        strategy = {"name": "blocked_temporal", "pivot_frame": pivot, "gap_frames": config["frame_gap"],
                    "limitation": "Temporal separation is not an independent camera/scene split. Frame numbers are inferred from names; verify the displayed samples."}
    else:
        groups = defaultdict(list)
        for r in records:
            groups[r["origin"]].append(r)
        keys = sorted(groups)
        rng.shuffle(keys)
        middle = len(keys)//2
        validation_pool = [r for key in keys[:middle] for r in groups[key]]
        test_pool = [r for key in keys[middle:] for r in groups[key]]
        strategy = {"name": "source_image_groups", "limitation": "No reliable video IDs. Derived copies/exact duplicates excluded; near-duplicate scene leakage may remain."}
    def sample(pool, count):
        pool = sorted(pool, key=lambda r: r["relative_path"])
        chosen = rng.sample(pool, min(count, len(pool)))
        return sorted(chosen, key=lambda r: r["relative_path"])
    validation = sample(validation_pool, config["validation_images"])
    test = sample(test_pool, config["test_images"])
    for name, split in [("validation", validation), ("test", test)]:
        if not split or not sum(len(r["boxes"]) for r in split):
            raise ValueError(f"{name} is empty or has no target boxes. Inspect frame numbering/category mapping; lower frame_gap only before measuring anything.")
    assert not {r["pixel_sha256"] for r in validation} & {r["pixel_sha256"] for r in test}
    assert not {r["origin"] for r in validation} & {r["origin"] for r in test}
    strategy.update({"validation_pool": len(validation_pool), "test_pool": len(test_pool), "validation_images": len(validation), "test_images": len(test),
                     "unused_images": len(records)-len(validation)-len(test), "original_dataset_splits_reassigned": True,
                     "reason": "No new model training. Validation/test are rebuilt to reduce adjacent-frame leakage."})
    return validation, test, strategy


def prepare(config, output_dir):
    root, acquisition = acquire(config)
    files = discover_annotations(root, config)
    records, audit = load_records(root, files, config["target_category_names"])
    validation, test, split_info = split_records(records, config)
    audit.update({"acquisition": acquisition, "annotations": [{"file": p.relative_to(root).as_posix(), "sha256": sha256(p)} for p in files], "split": split_info})
    output_dir = Path(output_dir)
    write_json(output_dir / "data_audit.json", audit)
    write_json(output_dir / "split_manifest.json", {"validation": validation, "test": test})
    print(f"검증 {len(validation)}장 / 최종 평가 {len(test)}장 / 대상 {audit['target_category_names']}", flush=True)
    return validation, test, audit
