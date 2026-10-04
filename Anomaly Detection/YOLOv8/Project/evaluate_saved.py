"""저장된 예측으로 confidence별 평가 수치를 다시 계산한다.

실행: python evaluate_saved.py
모델 추론과 mAP 계산은 포함하지 않는다.
"""
import json
from pathlib import Path


def pair_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_detections(gt, predictions, threshold):
    """클래스가 같고 IoU가 0.5 이상인 정답과 예측을 한 번씩 연결한다.

    정답: [class, x1, y1, x2, y2], 예측: 같은 구조 뒤에 confidence가 붙는다.
    점수가 높은 예측부터 확인하며, 이미 연결한 정답은 다시 세지 않는다.
    """
    predictions = sorted((p for p in predictions if p[5] >= threshold), key=lambda p: -p[5])
    used, tp = set(), 0
    for p in predictions:
        candidates = [(pair_iou(p[1:5], g[1:5]), j) for j, g in enumerate(gt)
                      if j not in used and int(p[0]) == int(g[0])]
        overlap, j = max(candidates, default=(0.0, -1))
        if overlap >= 0.5:
            used.add(j)
            tp += 1
    return tp, len(predictions)-tp, len(gt)-tp


def evaluate(records, predictions, threshold):
    tp = fp = fn = 0
    for record in records:
        a, b, c = match_detections(record['boxes'], predictions[record['id']], threshold)
        tp, fp, fn = tp+a, fp+b, fn+c
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f2 = 5*precision*recall/(4*precision+recall) if 4*precision+recall else 0.0
    return dict(confidence=threshold, TP=tp, FP=fp, FN=fn,
                precision=precision, recall=recall, F2=f2)


def main():
    artifacts = Path(__file__).resolve().parent/'artifacts'
    records = json.loads((artifacts/'manifest.json').read_text(encoding='utf-8'))
    val_predictions = json.loads((artifacts/'validation_predictions.json').read_text(encoding='utf-8'))
    val_records = [r for r in records if r['split'] == 'val']
    validation = [evaluate(val_records, val_predictions, t) for t in [0.15, 0.25, 0.50]]
    chosen = sorted(validation, key=lambda r: (-r['F2'], r['FP'], -r['confidence']))[0]['confidence']
    test_predictions = json.loads((artifacts/'test_predictions.json').read_text(encoding='utf-8'))
    test = evaluate([r for r in records if r['split'] == 'test'], test_predictions, chosen)
    expected = json.loads((artifacts/'test_summary.json').read_text(encoding='utf-8'))['operating_point']
    assert all(abs(test[k]-expected[k]) < 1e-10 for k in test), (test, expected)
    print(json.dumps(dict(validation=validation, test=test, verified=True), indent=2))


if __name__ == '__main__':
    main()
