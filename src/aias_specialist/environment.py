from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGES = [
    "faster-whisper",
    "huggingface-hub",
    "jiwer",
    "pandas",
    "python-docx",
    "pyyaml",
]


def _command(command: list[str], cwd: Path) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        suffix = ".exe" if os.name == "nt" else ""
        candidate = Path(sys.executable).parent / f"{command[0]}{suffix}"
        if candidate.exists():
            executable = str(candidate)
    if executable is None:
        return {"available": False, "command": command[0]}
    result = subprocess.run(
        [executable, *command[1:]],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def collect_environment(project_root: Path) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    gpu_count = 0
    try:
        import ctranslate2

        gpu_count = ctranslate2.get_cuda_device_count()
    except Exception:
        pass

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "cuda_device_count": gpu_count,
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "git": _command(["git", "status", "--short", "--branch"], project_root),
        "git_remote": _command(["git", "remote", "-v"], project_root),
        "hf_cli": _command(["hf", "version"], project_root),
        "nvidia_smi": _command(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            project_root,
        ),
    }


def doctor(settings: Any) -> dict[str, Any]:
    environment = collect_environment(settings.project_root)
    environment["inputs"] = {
        "manifest_exists": settings.paths.manifest.exists(),
        "manifest": str(settings.paths.manifest),
        "domain_terms_exists": settings.paths.domain_terms.exists(),
        "domain_terms": str(settings.paths.domain_terms),
        "model_lock_exists": settings.paths.model_lock.exists(),
        "model_lock": str(settings.paths.model_lock),
    }
    environment["ready"] = bool(
        environment["inputs"]["manifest_exists"] and environment["inputs"]["domain_terms_exists"]
    )
    return environment
