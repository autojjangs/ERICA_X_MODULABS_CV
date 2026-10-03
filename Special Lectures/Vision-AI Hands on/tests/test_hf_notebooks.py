"""Exercise notebook behavior when Colab CLI continues after a cell error."""
from pathlib import Path

import nbformat
import pytest


def test_completed_training_report_blocks_every_following_cell(tmp_path):
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "hf_colab_gpu/notebooks/02_gpu_finetuning.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    settings_index = next(
        i for i, cell in enumerate(cells)
        if 'if (OUTPUT_DIR / "report.json").exists():' in cell.source
    )
    existing = {
        "report.json": b'{"status":"completed","previous_run":true}',
        "training.csv": b"previous,training\n",
        "predictions.csv": b"previous,predictions\n",
        "learning_curves.png": b"previous plot",
        "confusion_comparison.png": b"previous confusion",
        "finetuned_model/model.safetensors": b"previous checkpoint",
    }
    for name, contents in existing.items():
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
    namespace = {"OUTPUT_DIR": tmp_path, "TRAINING_ALLOWED": True}
    with pytest.raises(FileExistsError):
        exec(cells[settings_index].source, namespace)
    assert namespace["TRAINING_ALLOWED"] is False

    # Match the CLI behavior: attempt every later cell despite the first error.
    for cell in cells[settings_index + 1:]:
        with pytest.raises(RuntimeError, match="학습 준비가 완료되지"):
            exec(cell.source, namespace)
    observed = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*") if path.is_file()
    }
    assert observed == existing
