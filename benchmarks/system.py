from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def environment_info(repository_root: Path) -> dict[str, Any]:
    git_status = _git_output(repository_root, "status", "--porcelain")
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "byte_order": sys.byteorder,
        "git_commit": _git_output(repository_root, "rev-parse", "HEAD"),
        "git_dirty": None if git_status is None else bool(git_status),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def current_rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (IndexError, OSError, ValueError):
            pass

    return None


def peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 if sys.platform == "darwin" else 1_024
    return int(peak * multiplier)


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cpu_utilization_percent(cpu_seconds: float, wall_seconds: float) -> float | None:
    if wall_seconds <= 0:
        return None
    return (cpu_seconds / wall_seconds) * 100


def _git_output(repository_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return completed.stdout.strip()
