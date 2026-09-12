"""Inspect the local environment without network calls or package installation."""

import argparse
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["analysis", "mac", "cuda"], default="analysis")
    parser.add_argument("--require-research", action="store_true")
    arguments = parser.parse_args()
    failures = []
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Repository: {Path(__file__).resolve().parents[1]}")
    print(f"Git executable: {shutil.which('git') or 'not found'}")
    if sys.version_info < (3, 11):
        failures.append("Python 3.11+ is required for the research environment.")
    if platform.system() == "Darwin":
        memory = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if memory.returncode == 0 and memory.stdout.strip().isdigit():
            print(f"Physical memory: {int(memory.stdout) / 1024 ** 3:.1f} GiB")
    packages = ["numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "torch", "transformers", "accelerate", "datasets", "huggingface-hub", "safetensors"]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
            if arguments.require_research:
                failures.append(f"Missing research package: {package}")
        print(f"{package}: {version}")
    if arguments.backend in {"mac", "cuda"}:
        try:
            import torch

            available = torch.backends.mps.is_available() if arguments.backend == "mac" else torch.cuda.is_available()
            print(f"Requested accelerator available: {available}")
            if not available:
                failures.append(f"Requested {arguments.backend} accelerator is unavailable.")
        except ImportError:
            failures.append("PyTorch is required to inspect the requested accelerator.")
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    print("Offline inspection only. Model compatibility, throughput, and experiment feasibility remain untested.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
