"""Refresh the notebook source bundle while preserving prose and measured output."""
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def bundle_source():
    memory = io.BytesIO()
    paths = [ROOT / "requirements.txt", ROOT / "README.md"]
    for directory in ["src", "configs", "tests", "docs"]:
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix in {".py", ".json", ".md"})
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    raw = memory.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


def build():
    path = ROOT / "notebooks/vehicle_detection_project.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    embedded, digest = bundle_source()
    targets = [cell for cell in notebook["cells"]
               if cell["cell_type"] == "code" and
               re.search(r"^SOURCE_BUNDLE_BASE64 =", "".join(cell["source"]), re.M)]
    if len(targets) != 1:
        raise ValueError("Expected exactly one source bootstrap cell.")
    cell = targets[0]
    source = "".join(cell["source"])
    for key, value in [("SOURCE_BUNDLE_BASE64", embedded), ("SOURCE_BUNDLE_SHA256", digest)]:
        source, count = re.subn(rf"^{key} = .+$", f'{key} = "{value}"', source, flags=re.M)
        if count != 1:
            raise ValueError(f"Missing or duplicate assignment: {key}")
    if source != "".join(cell["source"]):
        cell["source"] = source.splitlines(keepends=True)
        # An old hash printed by a previous bootstrap is no longer current.
        cell["outputs"] = []
        cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    notebook["metadata"].setdefault("vehicle_project", {})["source_sha256"] = digest
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Source bundle refreshed; markdown and measurement outputs preserved: {digest}")


if __name__ == "__main__":
    build()
