"""Save local package and platform metadata without credentials or network access."""

import datetime
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "artifacts" / "environment" / stamp
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    packages = sorted(
        [{"name": distribution.metadata.get("Name", "unknown"), "version": distribution.version}
         for distribution in importlib.metadata.distributions()],
        key=lambda package: package["name"].lower(),
    )
    metadata = {
        "created_at_utc": stamp,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "packages": packages,
        "scope": "Local environment metadata, not model or experiment validation",
    }
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if freeze.returncode != 0:
        raise SystemExit("pip freeze failed; environment metadata was saved, but no package freeze was written.")
    (output / "requirements.freeze.txt").write_text(freeze.stdout)
    print(f"Saved local environment to {output}")


if __name__ == "__main__":
    main()
