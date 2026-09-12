# A100 run plan — role confusion across OLMo 3 Think post-training

Prepared. Everything below is restartable; rerunning any step skips finished rows.

## Goal and deliverables

1. **Stage effect with uncertainty.** Compliance with instructions injected into a tool result at SFT, DPO, RLVR; 90 injected items (10 per goal x voice), 12 user-controls, 8 clean, **3 sampled draws per item**. Item-clustered bootstrap CIs. Marker goal is the primary measure; format and exfil are secondary (they conflict with the task or the record's own "do not share" note).
2. **Role representation.** User-turn vs environment-turn probes and directions at all three stages (already shown invariant on the Mac; re-run on GPU for the record).
3. **Silent compliance.** Fraction of complied traces whose think block never mentions the injected instruction, per stage, pooled over draws.
4. **Defenses.** On all 90 injected + 8 clean items, one draw: delimiter prompt; turn-conditioned constant vector at 1x, 2x, 4x the class gap (block 16, environment-turn tokens, prefill only); random matched-norm vector. RLVR and SFT first, then DPO with the 2x setting. Utility: task accuracy and answer validity on the same items, plus clean-item false positives.
5. **Negative controls kept in the write-up.** Role-confusion score vs compliance; within-goal-type activation probes (Mac result: chance).

## Machine settings (80 GB A100)

| setting | value | why |
|---|---|---|
| dtype | bf16 | as on Mac; 14 GB per checkpoint |
| decoding | sampling T=0.6, top-p 0.95, seed per (batch, draw) | model-card defaults; greedy loops |
| cap | 5,000 new tokens; summaries use the same effective cap | RLVR truncation was 26% at 3,600 |
| batch | 20 with static cache | 20 x 5,400 tokens x 0.52 MB = 57 GB KV + 14 GB weights, ~71 GB peak |
| attention | sdpa | default; no flash-attn install |
| per-process cap | disabled (100000) | the Metal graph-cache issue does not apply to CUDA |
| one model at a time | yes | 3 checkpoints would not fit with the KV cache |

Expected speed: ~35-45 ms/step at batch 20; a full-cap batch ~3-4 min; most batches finish early.

## Order and budget (at ~$1.5/h)

| step | command (inside tmux on the pod) | est. |
|---|---|---|
| setup | `bash scripts/gpu_setup.sh` | 20 min, ~$0.5 |
| everything | `tmux new -d -s rc 'bash scripts/run_gpu_chain.sh'` | ~7-8 h, ~$11 |

Within the chain: roles (10 min) -> injection none x3 draws x3 stages (~3 h) -> defenses rlvr, sft (5 arms each, ~3.5 h) -> dpo defenses (3 arms, ~1.5 h) -> summaries. Stop the pod when `artifacts/runs/gpu_chain.log` shows `chain complete`, or earlier at any step boundary; every step is independently usable.

## From the Mac

```bash
#.env: GPU_HOST, GPU_PORT, GPU_USER, GPU_KEY, GPU_DIR (see scripts/remote.sh header)
bash scripts/remote.sh check
bash scripts/remote.sh push
bash scripts/remote.sh run "bash scripts/gpu_setup.sh"
bash scripts/remote.sh run "tmux new -d -s rc 'bash scripts/run_gpu_chain.sh'"
bash scripts/remote.sh run "tail -5 artifacts/runs/gpu_chain.log"
bash scripts/remote.sh pull # results + logs + jsonl + manifests
bash scripts/remote.sh pull-acts inj_gpu # activations, when needed for probes
```

## Stop / fix rules

- Smoke test in setup must show > 20 tok/s per stream and a finished 400-token generation; otherwise inspect the CUDA/torch install before launching the chain.
- If any stage's clean false-positive rate exceeds 1/8, inspect detectors before reading compliance numbers.
- If marker compliance at RLVR is not above SFT's upper CI after 3 draws, report a null for the stage effect; do not add draws to chase it.
- Steering arms: if answer validity on injected items drops below 80% of the no-defense value, that arm is reported as "breaks generation", not as a defense.
- Spending: record pod start/stop times and rate in `logs/spending.csv`; hard stop at $25 total for the main run.

## What is reused from the Mac runs

`inj_main` (54 injected, single draw, SFT+RLVR, cap 3,600 effective) and `roles01` remain as the development results; the GPU run `inj_gpu` / `roles_gpu` supersedes them for the write-up. Item IDs are deterministic from the config seed, so the Mac items are a subset of the GPU items.

## Extension (added, evening): the framing study at scale

Order of value:
1. **Compliance matrix (vLLM, `.venv-vllm`, `scripts/run_injection_vllm.py`).** Types: the six attacker framings, the six literature framings, the four ablations (escaped fake turn, lexical fake turn, destyled CoT forgery, length control). 25 items each, 3 draws, SFT/DPO/RLVR, cap 3,000, 12 clean items per stage. Per-item seeds. Est. 1-2 GPU-hours.
2. **Role scores (HF prompt-only pass).** Project every injected span on the GPU-derived env-minus-user direction (`results/generated/roles_gpu/dir_<stage>_b16.npy`); item-level logistic fit of success on score with item-clustered bootstrap; token-level fraction-user variant. CPU after a short GPU pass.
3. **Defenses (HF, `scripts/run_injection.py`).** On the framings that work at SFT and RLVR: steer at blocks 8/16/24 × 1x/2x/4x, random matched norm at each setting, probe-gated steering, delimiter prompt, and an escaping baseline for the structural fake turn. Baseline arm regenerated in HF for a within-engine comparison. Answer validity reported for every arm.
4. **Role monitor.** Detection AUROC of injected vs clean tool spans from the role score alone; false-positive rate on clean records.
5. **Generality.** Web-page-summary and email-retrieval contexts with the same payloads.
6. **Realistic goals at scale.** answer/link/deny/toolcall, 25 items × 3 draws × 3 stages (vLLM).

Reporting rules: escaped and unescaped fake turns side by side; cap stated next to every rate; random sample of transcripts; item-clustered intervals everywhere.
