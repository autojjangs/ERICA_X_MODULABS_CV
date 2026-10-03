"""Distribute only the HF course and its three reading companions."""
import importlib.util
from pathlib import Path
import stat
from zipfile import ZipFile, ZipInfo

import pytest


spec = importlib.util.spec_from_file_location("student_distribution", Path(__file__).parents[1] / "scripts/package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def fixture_root(tmp_path):
    root = tmp_path / "course"
    for name in package.REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    return root


def test_zip_keeps_hf_course_and_compact_evidence_only(tmp_path):
    root = fixture_root(tmp_path)
    excluded = [
        ".local-state/colab-cli/token.json", ".venv/bin/python", ".cache/download.bin",
        "runs/gpu/report.json", "results/gpu/report.json", "custom_images/test/cup.jpg",
        "vision_lab/model.py", "pyproject.toml", "presentation/output/old.pptx",
        "validation/old/report.json", "validation/direct/gpu/report.json",
        "validation/direct/gpu/finetuned_checkpoint.npz", "assets/pretrained/config.json",
        "notebook_sources/training.py", "docs/lecture.md", "scripts/build_notebooks.py",
        "scripts/build_final_guide.py", "tests/test_direct_training.py",
        "requirements/codespaces-jax.txt", "requirements/gpu.txt", "requirements/tpu.txt",
        ".agents/skills/vision-colab/token.yaml", "scripts/credentials.py",
        "notebooks/03_tpu_and_compare.ipynb", "notebooks/02_gpu_finetuning_output.ipynb",
        "scripts/__pycache__/cached.py", "hf_colab_gpu/build_prepare_notebook.py",
        "hf_colab_gpu/models/deit-tiny/pytorch_model.bin",
        "hf_colab_gpu/models/deit-tiny/config.json", "hf_colab_gpu/data/train.npz",
        "hf_colab_gpu/.cache/huggingface/token", "hf_colab_gpu/results/report.json",
        "hf_colab_gpu/notebooks/01_gpu_inference_output.ipynb",
        "hf_colab_gpu/validation/gpu/finetuned_model.safetensors",
        "hf_colab_gpu/validation/gpu/model.zip", "hf_colab_gpu/validation/gpu/token.json",
        "hf_colab_gpu/validation/gpu/finetuned_model/config.json",
        "hf_colab_gpu/validation/gpu/finetuned_model/report.json",
        "hf_colab_gpu/validation/gpu/raw.log", "hf_colab_gpu/credentials.md",
        "hf_colab_gpu/validation/gpu/01_gpu_inference_output.ipynb",
        "hf_colab_gpu/validation/codespaces/00_hf_download_and_data.ipynb",
        "hf_colab_gpu/validation/old/report.json",
    ]
    for name in excluded:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private or obsolete")
    evidence = {
        "hf_colab_gpu/validation/gpu/report.json": b'{"status": "completed"}',
        "hf_colab_gpu/validation/gpu/training.csv": b"epoch,loss\n1,0.5\n",
        "hf_colab_gpu/validation/gpu/learning_curves.png": b"PNG fixture",
        "hf_colab_gpu/validation/codespaces/completed/report.json": b'{"passed": true}',
    }
    for name, data in evidence.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    artifact = tmp_path / "students.zip"
    result = package.build_distribution(artifact, root)
    with ZipFile(artifact) as archive:
        names = {name.removeprefix("vision_ai_notbook/") for name in archive.namelist()}
        for name, data in evidence.items():
            assert archive.read(f"vision_ai_notbook/{name}") == data
    assert package.REQUIRED <= names
    assert not names.intersection(excluded)
    assert {name for name in names if name.endswith(".ipynb")} == {
        f"hf_colab_gpu/notebooks/{name}" for name in package.HF_NOTEBOOKS
    } | {f"hf_colab_gpu/explanations/{name}" for name in package.HF_EXPLANATIONS}
    assert result["files"] == len(names)


@pytest.mark.parametrize("notebook", package.HF_NOTEBOOKS)
def test_missing_hf_notebook_aborts_distribution(tmp_path, notebook):
    root = fixture_root(tmp_path)
    (root / "hf_colab_gpu/notebooks" / notebook).unlink()
    with pytest.raises(FileNotFoundError, match=notebook):
        package.build_distribution(tmp_path / "students.zip", root)


@pytest.mark.parametrize("notebook", package.HF_EXPLANATIONS)
def test_missing_explanation_notebook_aborts_distribution(tmp_path, notebook):
    root = fixture_root(tmp_path)
    (root / "hf_colab_gpu/explanations" / notebook).unlink()
    with pytest.raises(FileNotFoundError, match=notebook):
        package.build_distribution(tmp_path / "students.zip", root)


def test_symlinked_optional_evidence_is_excluded(tmp_path):
    root = fixture_root(tmp_path)
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "report.json").write_text("do not distribute")
    (outside / "environment.json").write_text("do not distribute")
    validation = root / "hf_colab_gpu/validation"
    validation.mkdir()
    (validation / "gpu").symlink_to(outside, target_is_directory=True)
    (validation / "environment.json").symlink_to(outside / "environment.json")
    files = {path.relative_to(root).as_posix() for path in package.collect_files(root)}
    assert "hf_colab_gpu/validation/gpu/report.json" not in files
    assert "hf_colab_gpu/validation/environment.json" not in files


def test_symlinked_required_source_aborts_distribution(tmp_path):
    root = fixture_root(tmp_path)
    outside = tmp_path / "external.py"
    outside.write_text("do not distribute")
    source = root / "scripts/doctor.py"
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(FileNotFoundError, match="scripts/doctor.py"):
        package.collect_files(root)


@pytest.mark.parametrize("extra", [
    "vision_ai_notbook/notebooks/03_tpu_and_compare.ipynb",
    "vision_ai_notbook/hf_colab_gpu/validation/gpu/01_gpu_inference_output.ipynb",
    "vision_ai_notbook/notebook_sources/training.py",
    "vision_ai_notbook/validation/direct/gpu/report.json",
    "vision_ai_notbook/hf_colab_gpu/validation/gpu/token.json",
    "vision_ai_notbook/hf_colab_gpu/explanations/unreviewed_output.ipynb",
    "vision_ai_notbook/../outside.py",
    "/vision_ai_notbook/README.md",
])
def test_archive_validation_rejects_unapproved_paths(tmp_path, extra):
    root = fixture_root(tmp_path)
    artifact = tmp_path / "students.zip"
    package.build_distribution(artifact, root)
    with ZipFile(artifact, "a") as archive:
        archive.writestr(extra, "not allowed")
    with pytest.raises(RuntimeError):
        package.validate_archive(artifact)


def test_archive_validation_rejects_symlink_entries(tmp_path):
    root = fixture_root(tmp_path)
    artifact = tmp_path / "students.zip"
    package.build_distribution(artifact, root)
    entry = ZipInfo("vision_ai_notbook/hf_colab_gpu/validation/gpu/report.json")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(artifact, "a") as archive:
        archive.writestr(entry, "/private/report.json")
    with pytest.raises(RuntimeError, match="심볼릭 링크"):
        package.validate_archive(artifact)
