"""Bundle three practice notebooks, three reading companions and support files.

This ZIP distributes files; it does not build or install a Python package.
"""
import argparse
import hashlib
import json
from pathlib import Path
import stat
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
HF_NOTEBOOKS = (
    "00_hf_download_and_data.ipynb", "01_gpu_inference.ipynb", "02_gpu_finetuning.ipynb",
)
HF_EXPLANATIONS = tuple(name.replace('.ipynb', '_explained.ipynb') for name in HF_NOTEBOOKS)
HF_FILES = (
    "hf_colab_gpu/README.md", "hf_colab_gpu/requirements-torch.txt",
    "hf_colab_gpu/requirements-gpu.txt", "hf_colab_gpu/handson.md",
    "hf_colab_gpu/final-guide.md", "hf_colab_gpu/test-results.md",
    "hf_colab_gpu/codespaces-verification.md", "hf_colab_gpu/quantization-reference.md",
    "hf_colab_gpu/inference_exercises.py", "hf_colab_gpu/finetuning_exercises.py",
    "hf_colab_gpu/ai-coding-workshop.md", "hf_colab_gpu/build_gpu_notebooks.py",
    "hf_colab_gpu/build_final_guide.py",
    "hf_colab_gpu/pipeline_inference.py", "hf_colab_gpu/run_pipeline_colab.sh",
    "hf_colab_gpu/pipeline-guide.md", "hf_colab_gpu/pipeline-demo-results.md",
    "hf_colab_gpu/explanations/README.md",
    *(f"hf_colab_gpu/notebooks/{name}" for name in HF_NOTEBOOKS),
    *(f"hf_colab_gpu/explanations/{name}" for name in HF_EXPLANATIONS),
)
FILES = (
    "README.md", "THIRD_PARTY.md", "pytest.ini", ".vision-lab-root", ".gitignore",
    ".dockerignore", ".devcontainer/Dockerfile", ".devcontainer/devcontainer.json",
    ".vscode/settings.json", "requirements/codespaces.txt", "scripts/setup.sh",
    "scripts/doctor.py", "scripts/check_student_cells.py", "scripts/package.py",
    "tests/test_distribution.py", "tests/test_environment.py",
    "tests/test_hf_notebooks.py", "tests/test_student_cells.py",
    "tests/test_pipeline_inference.py", "tests/test_pipeline_wrapper.py",
    "tests/test_explanation_notebooks.py",
    ".agents/skills/vision-colab/SKILL.md",
    ".agents/skills/vision-colab/agents/openai.yaml",
    ".agents/skills/vision-pipeline-inference/SKILL.md",
    ".agents/skills/vision-pipeline-inference/agents/openai.yaml",
    *HF_FILES,
)
REQUIRED = set(FILES)
# Keep compact, reviewed evidence at its known path. Executed notebook copies,
# checkpoints, model downloads and old validation trees are not student materials.
HF_EVIDENCE_FILES = {
    "hf_colab_gpu/validation/environment.json",
    "hf_colab_gpu/validation/preparation.json",
    "hf_colab_gpu/validation/test-summary.json",
    "hf_colab_gpu/validation/codespaces/verification.json",
    "hf_colab_gpu/validation/codespaces/environment.json",
    "hf_colab_gpu/validation/codespaces/report.json",
    "hf_colab_gpu/validation/codespaces/cleanup.json",
    "hf_colab_gpu/validation/codespaces/completed/report.json",
    "hf_colab_gpu/validation/codespaces/cold-start/report.json",
    "hf_colab_gpu/validation/gpu/report.json",
    "hf_colab_gpu/validation/gpu/inference_report.json",
    "hf_colab_gpu/validation/gpu/training.csv",
    "hf_colab_gpu/validation/gpu/predictions.csv",
    "hf_colab_gpu/validation/gpu/cleanup.json",
    "hf_colab_gpu/validation/gpu/verification.json",
    "hf_colab_gpu/validation/gpu/pretrained_top5.png",
    "hf_colab_gpu/validation/gpu/learning_curves.png",
    "hf_colab_gpu/validation/gpu/confusion_comparison.png",
    "hf_colab_gpu/validation/pipeline-demo/result.json",
    "hf_colab_gpu/validation/pipeline-demo/cleanup.json",
    "hf_colab_gpu/validation/pipeline-demo/public-input.json",
    "hf_colab_gpu/validation/pipeline-demo/input.png",
}
ALLOWED_FILES = REQUIRED | HF_EVIDENCE_FILES
EXCLUDED_PARTS = {
    "__pycache__", ".venv", ".cache", ".local-state", ".pytest_cache",
    ".ipynb_checkpoints", "runs", "results", "outputs", "custom_images", "presentation",
    "models", "finetuned_model",
}
PRIVATE_NAMES = {"token", "tokens", "credential", "credentials", "session", "sessions",
                 "auth", "oauth", "oauth2", "secret", "secrets"}


def eligible(path, root):
    """Never follow symlinks or include local state among source files."""
    relative = path.relative_to(root)
    if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name.startswith(".env") or path.stem.lower() in PRIVATE_NAMES:
        return False
    return not any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent)


def collect_files(root=ROOT):
    root = Path(root).resolve()
    files = {root / name for name in ALLOWED_FILES if eligible(root / name, root)}
    missing = REQUIRED - {path.relative_to(root).as_posix() for path in files}
    if missing:
        raise FileNotFoundError("배포 필수 파일 누락: " + ", ".join(sorted(missing)))
    return sorted(files)


def validate_archive(path):
    with ZipFile(path) as archive:
        if archive.testzip():
            raise RuntimeError("배포 ZIP CRC 검증 실패")
        names = archive.namelist()
        relative = {name.removeprefix("vision_ai_notbook/") for name in names}
        if len(names) != len(set(names)) or REQUIRED - relative:
            raise RuntimeError("배포 ZIP 필수 파일 또는 중복 경로 검증 실패")
        for info in archive.infolist():
            name = info.filename
            parts = Path(name).parts
            if not name.startswith("vision_ai_notbook/") or ".." in parts:
                raise RuntimeError(f"배포 ZIP 경로 오류: {name}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise RuntimeError(f"배포 ZIP 심볼릭 링크 발견: {name}")
            if any(part in EXCLUDED_PARTS for part in parts):
                raise RuntimeError(f"배포 제외 경로 발견: {name}")
            if name.removeprefix("vision_ai_notbook/") not in ALLOWED_FILES:
                raise RuntimeError(f"허용하지 않은 배포 파일: {name}")
        notebooks = {name for name in relative if name.endswith(".ipynb")}
        expected = {f"hf_colab_gpu/notebooks/{name}" for name in HF_NOTEBOOKS}
        expected |= {f"hf_colab_gpu/explanations/{name}" for name in HF_EXPLANATIONS}
        if notebooks != expected:
            raise RuntimeError("배포 ZIP에는 실습 노트북 3개와 코드 해설 노트북 3개만 포함해야 합니다.")
    return len(names)


def build_distribution(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output)
    files = collect_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".tmp")
    try:
        with ZipFile(partial, "w", ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, Path("vision_ai_notbook") / path.relative_to(root))
        count = validate_archive(partial)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return {"file": str(output), "files": count, "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "format": "HF CLI + Colab GPU workshop; three practice and three explanation notebooks; no project-package installation",
            "excludes": ["credentials", "private images", "data downloads", "run state",
                         "virtual environments", "raw logs", "validation checkpoints",
                         "executed notebook copies", "legacy JAX/TPU course",
                         "legacy Python package", "legacy slides"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "dist/vision_ai_notbook_hf_colab_gpu.zip")
    args = parser.parse_args()
    print(json.dumps(build_distribution(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
