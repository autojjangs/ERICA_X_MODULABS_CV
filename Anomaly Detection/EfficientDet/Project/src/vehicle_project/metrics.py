"""Single-class, one-to-one box metrics; no confidence-as-accuracy shortcut."""
import numpy as np


def iou_matrix(boxes_a, boxes_b):
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    intersection_wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = intersection_wh.prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_image(gt_boxes, predictions, threshold=0.0, match_iou=0.5, max_detections=100):
    """Greedy matching by descending confidence to the best unmatched GT."""
    ordered = sorted([p for p in predictions if p["score"] >= threshold], key=lambda p: -p["score"])[:max_detections]
    overlaps = iou_matrix([p["box"] for p in ordered], gt_boxes)
    used = set()
    matched = []
    for index, prediction in enumerate(ordered):
        available = [(float(v), j) for j, v in enumerate(overlaps[index]) if j not in used and v >= match_iou]
        best = max(available, default=None)
        gt_index = best[1] if best else None
        if gt_index is not None:
            used.add(gt_index)
        matched.append({"prediction": prediction, "gt_index": gt_index, "tp": int(gt_index is not None)})
    return matched, sorted(set(range(len(gt_boxes))) - used)


def evaluate(records, predictions, threshold, match_iou=0.5, max_detections=100):
    if not records:
        raise ValueError("Cannot evaluate an empty split.")
    total_tp = total_fp = total_fn = 0
    details = []
    for record in records:
        matches, missed = match_image(record["boxes"], predictions.get(record["id"], []), threshold, match_iou, max_detections)
        tp = sum(m["tp"] for m in matches)
        fp, fn = len(matches) - tp, len(missed)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        details.append({"image_id": record["id"], "gt_count": len(record["boxes"]), "pred_count": len(matches), "tp": tp, "fp": fp, "fn": fn, "count_error": len(matches) - len(record["boxes"]), "matched_gt": [m["gt_index"] for m in matches if m["tp"]], "missed_gt": missed})
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    small_gt = small_tp = 0
    for record, detail in zip(records, details):
        # Relative area avoids confusing native image sizes with model input sizes.
        small = {i for i, (x1, y1, x2, y2) in enumerate(record["boxes"]) if (x2-x1)*(y2-y1)/(record["width"]*record["height"]) <= .01}
        small_gt += len(small)
        small_tp += len(small.intersection(detail["matched_gt"]))
    return {"images": len(records), "gt_boxes": total_tp + total_fn, "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": precision, "recall": recall, "f1": f1,
            "count_mae": float(np.mean([abs(x["count_error"]) for x in details])),
            "small_gt": small_gt, "small_recall": small_tp / small_gt if small_gt else None}, details


def ap50(records, predictions, match_iou=0.5, max_detections=100):
    """101-point interpolated AP for one class, no crowd/ignore, one IoU."""
    positives = sum(len(r["boxes"]) for r in records)
    if positives == 0:
        return None
    ranked = []
    for record in records:
        matches, _ = match_image(record["boxes"], predictions.get(record["id"], []), 0, match_iou, max_detections)
        ranked.extend((m["prediction"]["score"], m["tp"]) for m in matches)
    ranked.sort(key=lambda x: -x[0])
    if not ranked:
        return 0.0
    tp = np.cumsum([v[1] for v in ranked])
    fp = np.cumsum([1-v[1] for v in ranked])
    recall = tp / positives
    precision = tp / (tp + fp)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    values = [float(envelope[recall >= r].max()) if np.any(recall >= r) else 0.0 for r in np.linspace(0, 1, 101)]
    return float(np.mean(values))


def coco_ap50(records, predictions, max_detections=100):
    """Independent check using official pycocotools; keep its distinct label."""
    import contextlib
    import io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, annotations, detections = [], [], []
    ann_id = 1
    for image_id, record in enumerate(records, 1):
        images.append({"id": image_id, "width": record["width"], "height": record["height"]})
        for x1, y1, x2, y2 in record["boxes"]:
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": 1, "bbox": [x1,y1,x2-x1,y2-y1], "area": (x2-x1)*(y2-y1), "iscrowd": 0})
            ann_id += 1
        for p in predictions.get(record["id"], []):
            x1,y1,x2,y2 = p["box"]
            detections.append({"image_id": image_id, "category_id": 1, "bbox": [x1,y1,x2-x1,y2-y1], "score": p["score"]})
    if not annotations:
        return None
    if not detections:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"info": {}, "images": images, "annotations": annotations, "categories": [{"id": 1, "name": "car"}]}
        gt.createIndex()
        dt = gt.loadRes(detections)
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.iouThrs = np.array([.5])
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
    values = evaluator.eval["precision"][0, :, 0, 0, -1]
    return float(values[values > -1].mean())
