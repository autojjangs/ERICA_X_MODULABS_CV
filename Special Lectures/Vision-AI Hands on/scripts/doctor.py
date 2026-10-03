#!/usr/bin/env python3
"""Codespaces의 HF CLI·Colab CLI 준비 환경을 점검합니다. 로그인·원격 자원 할당은 하지 않습니다."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
DEPENDENCIES = {
    "pyarrow": ("pyarrow", "20.0.0"),
    "numpy": ("numpy", "2.5.2"),
    "Pillow": ("PIL", "12.3.0"),
    "huggingface-hub": ("huggingface_hub", "0.36.0"),
    "matplotlib": ("matplotlib", "3.11.1"),
    "google-colab-cli": ("colab_cli", "0.6.0"),
    "jupyter-kernel-client": ("jupyter_kernel_client", "0.9.0"),
    "ipykernel": ("ipykernel", "7.3.0"),
    "nbformat": ("nbformat", "5.10.4"),
    "nbclient": ("nbclient", "0.10.4"),
    "pytest": ("pytest", "9.1.1"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_file(directory: Path, filename: str) -> Path:
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"File path escapes its directory: {filename}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def check_cli_kernel_api() -> str:
    """원격 런타임을 만들기 전에 CLI 실행 API의 호환성을 확인합니다."""
    client = importlib.import_module("jupyter_kernel_client")
    if not callable(getattr(client, "KernelClient", None)):
        raise RuntimeError("Colab CLI 0.6.0 requires jupyter-kernel-client==0.9.0 (KernelClient); rerun setup.sh")
    for method in ("start", "execute", "stop"):
        if not callable(getattr(client.KernelClient, method, None)):
            raise RuntimeError(f"KernelClient.{method} is unavailable")
    if not hasattr(client.JupyterSubprotocol, "DEFAULT"):
        raise RuntimeError("JupyterSubprotocol.DEFAULT is unavailable")
    return "KernelClient start/execute/stop and default WebSocket protocol are available (no connection opened)"


def check_data(root: Path) -> str:
    directory = root / "data" / "prepared"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported data manifest schema_version")
    classes = manifest["classes"]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Manifest must contain distinct class names")
    counts = []
    for split in ("train", "validation", "test"):
        metadata = manifest["splits"][split]
        path = contained_file(directory, metadata["file"])
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"{split}: SHA-256 mismatch")
        if metadata["count"] < 1:
            raise ValueError(f"{split}: no samples")
        counts.append(f"{split}={metadata['count']}")
    return f"{len(classes)} classes; " + ", ".join(counts) + "; all split hashes verified"


def configure_state(root: Path, user_home: Path | None = None) -> dict[str, str]:
    """기존 인증 정보를 이동하지 않고 제외된 작업 폴더에 CLI 상태를 보존합니다."""
    user_home = user_home or Path.home()
    target = root / ".local-state" / "colab-cli"
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(".local-state resolves outside this workspace; refusing to use it")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.parent.chmod(0o700)
    target.chmod(0o700)
    config_home = user_home / ".config"
    link = config_home / "colab-cli"
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return {"status": "ok", "detail": "Colab CLI workspace persistence is configured."}
        return {"status": "warning", "detail": "Existing ~/.config/colab-cli symlink points elsewhere; kept unchanged. Resolve it manually before relying on rebuild persistence."}
    if link.exists():
        return {"status": "warning", "detail": "Existing ~/.config/colab-cli directory or file was kept unchanged. Existing credentials were not copied. Resolve it manually before relying on rebuild persistence."}
    config_home.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    return {"status": "ok", "detail": "Colab CLI auth/history will persist under ignored .local-state/colab-cli (mode 700)."}


def run_checks(root: Path, require_data: bool = False) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def check(name: str, function) -> None:
        try:
            detail = function()
            checks.append({"name": name, "status": "ok", "detail": str(detail)})
        except Exception as error:
            checks.append({"name": name, "status": "error", "detail": f"{type(error).__name__}: {error}"})

    def python_check() -> str:
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("Use .venv/bin/python (Python 3.12) for the Codespaces workshop")
        return f"{sys.version.split()[0]}; executable={sys.executable}"

    check("python", python_check)
    for distribution, (module, expected) in DEPENDENCIES.items():
        def dependency_check(distribution=distribution, module=module, expected=expected):
            version = importlib.metadata.version(distribution)
            if expected and version != expected:
                raise RuntimeError(f"Expected {expected}, found {version}")
            imported = importlib.import_module(module)
            return f"{version}; import {module}; file={getattr(imported, '__file__', '(namespace package)')}"
        check(distribution, dependency_check)

    def cli_check() -> str:
        local_cli = root / ".venv" / "bin" / "colab"
        executable = str(local_cli) if local_cli.is_file() else shutil.which("colab")
        if not executable:
            raise FileNotFoundError("colab executable not found")
        # 0.6.0의 version 명령은 홈 폴더에 로그를 씁니다.
        # --help는 해당 초기화를 건너뛰므로 인증·로그·네트워크 없이 설치만 확인합니다.
        result = subprocess.run([executable, "--help"], check=True, text=True,
                                capture_output=True, timeout=20)
        output = result.stdout.strip()
        if "Colab CLI" not in output or "sessions" not in output:
            raise RuntimeError("The discovered executable does not expose the Colab CLI interface")
        return f"{executable}; --help works (distribution version checked separately)"

    def hf_cli_check() -> str:
        local_cli = root / ".venv" / "bin" / "hf"
        executable = str(local_cli) if local_cli.is_file() else shutil.which("hf")
        if not executable:
            raise FileNotFoundError("hf executable not found; rerun setup.sh")
        result = subprocess.run([executable, "--help"], check=True, text=True,
                                capture_output=True, timeout=20)
        if "download" not in result.stdout or "auth" not in result.stdout:
            raise RuntimeError("The discovered executable does not expose the Hugging Face CLI interface")
        return f"{executable}; --help works (no model downloaded or authentication attempted)"

    def kernel_check() -> str:
        path = root / ".venv" / "share" / "jupyter" / "kernels" / "codespaces-vision-ai" / "kernel.json"
        spec = json.loads(path.read_text())
        if Path(spec["argv"][0]).absolute() != (root / ".venv" / "bin" / "python").absolute():
            raise ValueError("Kernel points to another Python environment; rerun setup.sh")
        return spec["display_name"]

    check("colab executable", cli_check)
    check("Hugging Face executable", hf_cli_check)
    check("Colab execution API compatibility", check_cli_kernel_api)
    check("notebook kernel", kernel_check)
    if require_data:
        check("prepared data", lambda: check_data(root))
    else:
        checks.append({"name": "prepared data", "status": "skipped", "detail": "Run notebook 00 to prepare data, then run python scripts/doctor.py --data to verify it."})

    target = root / ".local-state" / "colab-cli"
    link = Path.home() / ".config" / "colab-cli"
    linked = link.is_symlink() and link.resolve() == target.resolve()
    checks.append({"name": "credential persistence", "status": "ok" if linked else "warning",
                   "detail": "Workspace symlink is configured." if linked else "Workspace symlink is not configured; existing home credentials were not inspected. Run setup.sh in the new Codespace."})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--data", action="store_true", help="Also require and verify data prepared by notebook 00")
    data_mode.add_argument("--skip-data", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--configure-state-only", action="store_true", help="Safely configure workspace persistence; no authentication")
    args = parser.parse_args()
    if args.configure_state_only:
        try:
            result = configure_state(ROOT)
        except Exception as error:
            result = {"status": "error", "detail": str(error)}
        checks = [{"name": "credential persistence", **result}]
    else:
        checks = run_checks(ROOT, require_data=args.data)
    ok = not any(check["status"] == "error" for check in checks)
    report = {"ok": ok, "root": str(ROOT), "checks": checks,
              "scope": "Offline local checks only; Google authentication, Colab allocation and GPU execution were not tested. Notebook 00 downloads and checks the model files."}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for check in checks:
            print(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
        print(report["scope"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
