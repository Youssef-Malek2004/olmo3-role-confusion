#!/usr/bin/env python
"""Resolve immutable revisions for the configured models/dataset, record them, and download.

Usage:
    python scripts/pin_and_download.py --config configs/pilot.json [--stages sft rlvr] [--dry-run]

Writes configs/revisions.json (tracked) with repo -> commit sha. Downloads go to HF_HOME, which
defaults here to <repo>/.cache/huggingface (ignored by Git). Requires network for the first run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

from role_confusion.io import read_json, write_json  # noqa: E402

MODEL_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py", "*.md"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--stages", nargs="*", default=None,
                    help="stages to download; default = required stages in config")
    ap.add_argument("--revisions", default="configs/revisions.json")
    ap.add_argument("--dry-run", action="store_true", help="resolve and record shas only")
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    cfg = read_json(REPO_ROOT / args.config)
    rev_path = REPO_ROOT / args.revisions
    revisions = read_json(rev_path) if rev_path.exists() else {"models": {}, "dataset": {}}

    stages = args.stages or [m["stage"] for m in cfg["models"] if m.get("required")]
    models = [m for m in cfg["models"] if m["stage"] in stages]

    for m in models:
        repo = m["repo_id"]
        pinned = revisions["models"].get(repo, {}).get("sha")
        info = api.model_info(repo, revision=pinned)
        sha = info.sha
        if pinned and pinned != sha:
            print(f"WARNING: pinned {pinned} != resolved {sha} for {repo}")
        revisions["models"][repo] = {
            "stage": m["stage"], "sha": sha,
            "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "safetensors_files": sorted(s.rfilename for s in info.siblings
                                        if s.rfilename.endswith(".safetensors")),
        }
        print(f"{m['stage']:>5}  {repo}  @ {sha}")

    ds = cfg["dataset"]
    ds_pinned = revisions["dataset"].get("sha")
    ds_info = api.dataset_info(ds["repo_id"], revision=ds_pinned)
    revisions["dataset"] = {"repo_id": ds["repo_id"], "sha": ds_info.sha,
                            "configuration": ds["configuration"], "source_split": ds["source_split"],
                            "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(f"dataset {ds['repo_id']}  @ {ds_info.sha}")
    write_json(rev_path, revisions)
    print(f"wrote {rev_path}")

    if args.dry_run:
        return 0

    for m in models:
        repo = m["repo_id"]
        sha = revisions["models"][repo]["sha"]
        t0 = time.time()
        path = snapshot_download(repo, revision=sha, allow_patterns=MODEL_PATTERNS)
        print(f"downloaded {repo} -> {path} in {time.time() - t0:.0f}s")

    from datasets import load_dataset
    t0 = time.time()
    d = load_dataset(ds["repo_id"], ds["configuration"], split=ds["source_split"],
                     revision=revisions["dataset"]["sha"])
    print(f"dataset rows={len(d)} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
