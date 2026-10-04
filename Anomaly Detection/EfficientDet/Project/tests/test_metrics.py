import unittest
from vehicle_project.metrics import ap50, evaluate, iou_matrix, match_image


class BoxMetricsTests(unittest.TestCase):
    def record(self, boxes, image_id="a"):
        return {"id": image_id, "boxes": boxes, "width": 100, "height": 100}

    def pred(self, box, score=.9):
        return {"box":box,"score":score}

    def test_iou_and_empty_shapes(self):
        self.assertAlmostEqual(iou_matrix([[0,0,10,10]],[[5,0,15,10]])[0,0],1/3)
        self.assertEqual(iou_matrix([],[[0,0,1,1]]).shape,(0,1))
        self.assertEqual(iou_matrix([[0,0,1,1]],[]).shape,(1,0))

    def test_duplicate_predictions_are_false_positives(self):
        record=self.record([[0,0,10,10]])
        predictions={"a":[self.pred([0,0,10,10],.9),self.pred([0,0,10,10],.8)]}
        metrics,_=evaluate([record],predictions,.1)
        self.assertEqual((metrics["tp"],metrics["fp"],metrics["fn"]),(1,1,0))
        self.assertAlmostEqual(metrics["precision"],.5)
        self.assertAlmostEqual(metrics["f1"],2/3)
        self.assertEqual(metrics["count_mae"],1)

    def test_match_uses_best_unmatched_box(self):
        gt=[[0,0,10,10],[1,0,11,10]]
        matched,missed=match_image(gt,[self.pred([0,0,10,10],.9),self.pred([0,0,10,10],.8)])
        self.assertEqual(sum(x["tp"] for x in matched),2)
        self.assertEqual(missed,[])

    def test_threshold_keeps_misses_in_denominator(self):
        record=self.record([[0,0,10,10],[30,30,40,40]])
        metrics,_=evaluate([record],{"a":[self.pred([0,0,10,10],.2)]},.3)
        self.assertEqual((metrics["tp"],metrics["fp"],metrics["fn"]),(0,0,2))
        self.assertEqual(metrics["f1"],0)

    def test_negative_image_prediction_counts_as_false_positive(self):
        records=[self.record([[0,0,10,10]]),self.record([],"b")]
        metrics,_=evaluate(records,{"a":[self.pred([0,0,10,10])],"b":[self.pred([0,0,10,10])]},.3)
        self.assertEqual(metrics["precision"],.5)
        self.assertEqual(metrics["recall"],1)

    def test_perfect_and_missing_ap(self):
        records=[self.record([[0,0,10,10]])]
        self.assertEqual(ap50(records,{"a":[self.pred([0,0,10,10])]}),1)
        self.assertEqual(ap50(records,{}),0)
        self.assertIsNone(ap50([self.record([])],{}))

    def test_ap_penalizes_early_false_positives(self):
        records=[self.record([[0,0,10,10]])]
        predictions={"a":[self.pred([20,20,30,30],.99),self.pred([0,0,10,10],.8)]}
        self.assertEqual(ap50(records,predictions),.5)

    def test_no_small_objects_is_not_zero_recall(self):
        metrics,_=evaluate([self.record([[0,0,100,100]])],{},.3)
        self.assertIsNone(metrics["small_recall"])

    def test_empty_split_fails(self):
        with self.assertRaises(ValueError):
            evaluate([],{},.3)


if __name__ == "__main__":
    unittest.main()
