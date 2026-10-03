"""Guard credential persistence, CPU setup and manifest checks without cloud access."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("lab_doctor", Path(__file__).parents[1] / "scripts" / "doctor.py")
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)


def test_state_link_is_private_and_idempotent(tmp_path):
    root, user_home = tmp_path / "repo", tmp_path / "home"
    root.mkdir()
    result = doctor.configure_state(root, user_home)
    target = root / ".local-state" / "colab-cli"
    assert result["status"] == "ok"
    assert (user_home / ".config" / "colab-cli").resolve() == target
    assert target.stat().st_mode & 0o777 == 0o700
    assert doctor.configure_state(root, user_home)["status"] == "ok"


@pytest.mark.parametrize("foreign_symlink", [False, True])
def test_existing_credentials_are_never_overwritten(tmp_path, foreign_symlink):
    root, user_home = tmp_path / "repo", tmp_path / "home"
    root.mkdir()
    link = user_home / ".config" / "colab-cli"
    link.parent.mkdir(parents=True)
    original = tmp_path / "external" if foreign_symlink else link
    original.mkdir()
    credential = original / "token.json"
    credential.write_text("private existing value")
    if foreign_symlink:
        link.symlink_to(original)
    assert doctor.configure_state(root, user_home)["status"] == "warning"
    assert credential.read_text() == "private existing value"
    assert link.is_symlink() == foreign_symlink


def test_workspace_state_cannot_escape_via_symlink(tmp_path):
    root, foreign = tmp_path / "repo", tmp_path / "other"
    root.mkdir()
    foreign.mkdir()
    (root / ".local-state").symlink_to(foreign)
    with pytest.raises(ValueError, match="outside"):
        doctor.configure_state(root, tmp_path / "home")


def test_split_hash_detects_corruption(tmp_path):
    directory = tmp_path / "data" / "prepared"
    directory.mkdir(parents=True)
    splits = {}
    for name in ("train", "validation", "test"):
        path = directory / f"{name}.npz"
        path.write_bytes(b"pretend archive for hash test")
        splits[name] = {"file": path.name, "count": 1, "sha256": doctor.sha256(path)}
    (directory / "manifest.json").write_text(json.dumps({"schema_version": 1, "classes": ["cup"], "splits": splits}))
    assert "hashes verified" in doctor.check_data(tmp_path)
    (directory / "validation.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="validation: SHA-256 mismatch"):
        doctor.check_data(tmp_path)


def test_manifest_paths_cannot_escape_data_directory(tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    (tmp_path / "secret").write_text("secret")
    with pytest.raises(ValueError, match="escapes"):
        doctor.contained_file(directory, "../secret")


def test_doctor_rejects_incompatible_renamed_kernel_api(monkeypatch):
    incompatible = SimpleNamespace(JupyterKernelClient=object)
    monkeypatch.setattr(doctor.importlib, "import_module", lambda _: incompatible)
    with pytest.raises(RuntimeError, match="jupyter-kernel-client==0.9.0"):
        doctor.check_cli_kernel_api()


def test_installed_cli_kernel_api_is_compatible_without_connection():
    assert "no connection opened" in doctor.check_cli_kernel_api()


@pytest.mark.parametrize("require_data, expected", [(False, "skipped"), (True, "error")])
def test_notebook_prepares_data_after_environment_setup(tmp_path, monkeypatch, require_data, expected):
    """첫 환경 설정은 데이터 없이 가능하지만 --data는 실제 파일을 요구합니다."""
    monkeypatch.setattr(doctor, "DEPENDENCIES", {})
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    checks = doctor.run_checks(tmp_path, require_data=require_data)
    prepared = next(check for check in checks if check["name"] == "prepared data")
    assert prepared["status"] == expected
    if require_data:
        assert "FileNotFoundError" in prepared["detail"]


def test_basic_environment_does_not_require_jax_or_downloaded_model(tmp_path, monkeypatch):
    """기본 환경은 JAX 설치나 모델 다운로드 없이 점검할 수 있습니다."""
    assert not {"jax", "optax", "torch", "transformers"} & doctor.DEPENDENCIES.keys()
    monkeypatch.setattr(doctor, "DEPENDENCIES", {})
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    checks = doctor.run_checks(tmp_path)
    assert not any(check["name"] == "pretrained assets" for check in checks)


@pytest.mark.parametrize("command", [["bash", "scripts/setup.sh"], [sys.executable, "scripts/doctor.py"]])
def test_removed_jax_option_is_rejected_before_setup(command):
    result = subprocess.run(command + ["--with-jax"], cwd=doctor.ROOT,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert "--with-jax" in result.stderr
