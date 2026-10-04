import json
import unittest
from pathlib import Path
from PIL import Image
from vehicle_project.data import discover_annotations, load_records, source_name, split_records
from test_helpers import test_directory


class DataTests(unittest.TestCase):
    def test_derived_names_share_origin(self):
        self.assertEqual(source_name("frame_001_jpg.rf.abc.jpg"),source_name("frame_001_jpg.rf.def.jpg"))

    def test_coco_load_and_temporal_gap(self):
        with test_directory() as temporary:
            root=Path(temporary)
            folder=root/"No_Apply_Grayscale"/"Vehicles.v8i.coco"/"train"
            folder.mkdir(parents=True)
            images=[]
            annotations=[]
            for i in range(12):
                filename=f"frame_{i*100:05d}_jpg.rf.fixture.jpg"
                Image.new("RGB",(40,20),(i*20,50,100)).save(folder/filename)
                images.append({"id":i,"file_name":filename,"width":40,"height":20})
                annotations.append({"id":i,"image_id":i,"category_id":2,"bbox":[1,2,10,5]})
            path=folder/"_annotations.coco.json"
            path.write_text(json.dumps({"images":images,"annotations":annotations,"categories":[{"id":2,"name":"Car"}]}),encoding="utf-8")
            config={"dataset_variant":"No_Apply_Grayscale","seed":42,"frame_gap":30,"validation_images":4,"test_images":4}
            files=discover_annotations(root,config)
            records,audit=load_records(root,files,["Car"])
            self.assertEqual(len(records),12)
            self.assertEqual(records[0]["boxes"],[[1.,2.,11.,7.]])
            validation,test,strategy=split_records(records,config)
            self.assertGreaterEqual(min(r["frame"] for r in test)-max(r["frame"] for r in validation),30)
            self.assertEqual((len(validation),len(test)),(4,4))
            self.assertEqual(strategy["name"],"blocked_temporal")
            self.assertEqual(split_records(records,config)[0],validation)
            with self.assertRaises(ValueError):
                load_records(root,files,["Bus"])

    def test_exact_duplicates_removed(self):
        with test_directory() as temporary:
            root=Path(temporary)
            for name in ["one.png","two.png"]:
                Image.new("RGB",(20,20),(60,80,90)).save(root/name)
            path=root/"_annotations.coco.json"
            path.write_text(json.dumps({"categories":[{"id":1,"name":"car"}],"images":[{"id":i,"file_name":name,"width":20,"height":20} for i,name in enumerate(["one.png","two.png"])],"annotations":[{"id":i,"image_id":i,"category_id":1,"bbox":[0,0,10,10]} for i in range(2)]}),encoding="utf-8")
            records,audit=load_records(root,[path],["Car"])
            self.assertEqual(len(records),1)
            self.assertEqual(audit["duplicates_removed"],1)


if __name__ == "__main__":
    unittest.main()
