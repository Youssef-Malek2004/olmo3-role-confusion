# Setup

## Requirements

- Python 3.11 or newer. The runs used Python 3.13.15, torch 2.14.0, transformers 4.57.6.
- For generation: an Apple-silicon Mac with 48 GB unified memory (MPS, bf16, one 7B checkpoint at a time) or a CUDA GPU. The A100 runs used vLLM for the large compliance matrices and HF transformers for anything that needed steering hooks or activations.
- About 15 GB of disk per checkpoint. Six checkpoints were used in total (`configs/revisions.json`).

## Install

```bash
bash scripts/setup.sh --install mac        # or: --install cuda | --install analysis
source .venv/bin/activate
python scripts/doctor.py --backend mac --require-research
```

`--install analysis` installs only numpy, scipy, scikit-learn, pandas, matplotlib and is enough to run the unit tests, re-summarise saved runs, and regenerate the figures.

## Download the pinned checkpoints and MMLU

```bash
export HF_HOME="$PWD/.cache/huggingface"
python scripts/pin_and_download.py --config configs/pilot.json
```

The scripts default to `HF_HUB_OFFLINE=1`, so downloads must be run with that variable unset. Revisions are pinned in `configs/revisions.json`; every run manifest records the revision it used.

## Environment variables

`.env.example` documents the variables the scripts read. Nothing sources `.env` automatically. For a rented GPU, `scripts/remote.sh` reads the SSH target from `.env` (`GPU_HOST`/`GPU_PORT`/`GPU_USER`/`GPU_KEY`, or `GPU_ALIAS` for an entry in `~/.ssh/config`).

## Memory notes for the Mac

The generation code uses a preallocated static KV cache, clears the MPS cache after every batch, and sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.75` so a too-large batch fails instead of swapping. Batch 6 at a 4,096-token cap peaks near 27 GB. Sustained decoding for several hours throttles the laptop GPU by 2 to 4x; the chain scripts in `scripts/` insert cooling rests and cap items per process for that reason.
