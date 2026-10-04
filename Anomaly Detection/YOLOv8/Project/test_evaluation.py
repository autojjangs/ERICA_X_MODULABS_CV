"""간단한 박스 예시로 중복 예측, 클래스 오류, 경계값을 확인한다."""
import unittest
from evaluate_saved import match_detections


class MatchingTests(unittest.TestCase):
    def test_duplicate_predictions_are_false_positives(self):
        gt = [[0, 0, 0, 10, 10]]
        pred = [[0, 0, 0, 10, 10, .9], [0, 0, 0, 10, 10, .8]]
        self.assertEqual(match_detections(gt, pred, .5), (1, 1, 0))

    def test_wrong_class_is_both_false_positive_and_missed(self):
        self.assertEqual(match_detections([[0, 0, 0, 10, 10]], [[1, 0, 0, 10, 10, .9]], .5), (0, 1, 1))

    def test_confidence_boundary_is_inclusive(self):
        gt = [[0, 0, 0, 10, 10]]
        self.assertEqual(match_detections(gt, [[0, 0, 0, 10, 10, .5]], .5), (1, 0, 0))
        self.assertEqual(match_detections(gt, [[0, 0, 0, 10, 10, .49]], .5), (0, 0, 1))

    def test_iou_boundary_is_inclusive(self):
        self.assertEqual(match_detections([[0, 0, 0, 10, 10]], [[0, 0, 0, 20, 10, .9]], .5), (1, 0, 0))

    def test_low_overlap_does_not_claim_ground_truth(self):
        gt = [[0, 0, 0, 10, 10]]
        pred = [[0, 0, 0, 30, 10, .9], [0, 0, 0, 10, 10, .8]]
        self.assertEqual(match_detections(gt, pred, .5), (1, 1, 0))


if __name__ == '__main__':
    unittest.main()
