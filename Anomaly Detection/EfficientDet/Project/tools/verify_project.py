"""Offline checks, with explicit separation from a real model experiment."""
import ast
import base64
import hashlib
import io
import json
import os
import sys
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from vehicle_project.common import read_json, validate_config, write_json


def verify():
    config=read_json(ROOT/"configs/experiment.json")
    validate_config(config)
    python_files=[p for directory in ["src","tests","tools"] for p in (ROOT/directory).rglob("*.py")]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    notebook=read_json(ROOT/"notebooks/vehicle_detection_project.ipynb")
    assert notebook["nbformat"]==4
    assert len({c["id"] for c in notebook["cells"]})==len(notebook["cells"])
    code_cells=0
    embedded=None
    for cell in notebook["cells"]:
        source="".join(cell["source"])
        assert "\ufffd" not in source
        if cell["cell_type"]=="code":
            code_cells+=1
            tree=ast.parse(source,filename=cell["id"])
            assert cell["execution_count"] is None or isinstance(cell["execution_count"], int)
            assert isinstance(cell["outputs"], list)
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SOURCE_BUNDLE_BASE64" for t in node.targets):
                    embedded=ast.literal_eval(node.value)
    assert embedded
    raw=base64.b64decode(embedded)
    assert hashlib.sha256(raw).hexdigest()==notebook["metadata"]["vehicle_project"]["source_sha256"]
    embedded_files=0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            path=(ROOT/name).resolve()
            assert path.is_relative_to(ROOT)
            assert archive.read(name)==path.read_bytes(),f"Stale notebook source: {name}. Rebuild."
            embedded_files+=1
    scratch=os.environ.get("VEHICLE_TEST_SCRATCH",str(ROOT/".cache/test-fixtures"))
    os.environ["VEHICLE_TEST_SCRATCH"]=scratch
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    print(log.getvalue())
    checks={"checked_utc":datetime.now(timezone.utc).isoformat(),"python_files_parsed":len(python_files),"notebook_cells":len(notebook["cells"]),"notebook_code_cells_parsed":code_cells,
            "embedded_source_files_match":embedded_files,"unit_tests_run":result.testsRun,"unit_test_failures":len(result.failures),"unit_test_errors":len(result.errors),
            "offline_checks_passed":result.wasSuccessful(),
            "public_submission_verified":False,"note":"This command checks syntax, embedded source and offline fixtures. GPU measurements and publication are outside its scope; saved Colab evidence is recorded in LOCAL_CHECKS.md."}
    write_json(ROOT/"verification/checks.json",checks)
    (ROOT/"verification/test_log.txt").write_text(log.getvalue(),encoding="utf-8")
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=="__main__":
    verify()
