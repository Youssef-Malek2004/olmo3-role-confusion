# Results

Every run directory under `generated/` is the summary the analysis scripts wrote from the raw generations; the raw per-item text (`artifacts/runs/<run>/`) is not tracked here. `rows_*.jsonl` files carry one detection row per item (no generated text). `injection_summary.json` files carry per-condition counts, per-framing tables, and item-clustered bootstrap intervals.

| run | what | produced by |
|---|---|---|
| `inj_gpu` | A100 main result: marker/format/exfil x 3 voices, 3 draws, SFT/DPO/RLVR; defense arms on marker items (delimiter, random, steer 1x/2x/4x); `silent_compliance.json`, `role_monitor.json`, a 30-trace RLVR audit sample | `scripts/run_injection.py`, `scripts/summarize_injection.py` |
| `inj_matrix` | 16-framing tool-call compliance matrix (25 items x up to 3 draws x 3 stages, vLLM) and the injected-span projections on the role and own-voice directions (`span_scores_*`) | `scripts/run_injection_vllm.py`, `scripts/score_spans.py` |
| `inj_matrix_real` | realistic goals at scale (answer manipulation, tool call, link, deny), 3 stages | `scripts/run_injection_vllm.py` |
| `inj_frame_def`, `inj_frame_def_style`, `inj_frame_def_sum` | defense arms against the four working imitation framings: role vector 2x/4x, own-voice vector, sum, random at matched norms; `DEFENSE_TABLE.md` | `scripts/run_injection.py`, `scripts/analyze_frame_def.py` |
| `inj_style_def` | first own-voice steering pilot vs CoT forgery (SFT) | `scripts/run_injection.py` |
| `inj_legit` | utility check: correct answer key in the record, with and without steering | `scripts/run_injection.py` |
| `ct01`, `ct_smoke` | copy-vs-trust: can a steered model still read a code that exists only in the tool result? SFT arms none / steer 4x / random 4x | `scripts/run_copy_vs_trust.py` |
| `roles_gpu`, `roles01` | user-turn vs tool-turn role probes and difference-of-means directions (`dir_<stage>_b<block>.npy`) for the Think line; A100 and Mac captures agree | `scripts/capture_roles.py`, `scripts/fit_role_probe.py` |
| `roles_mac` | the same for the Instruct SFT/DPO/RL line | same |
| `voice01` | own-voice directions (model think-style vs prose, both inside the tool turn) and payload projections | `scripts/capture_voice.py` |
| `agentic` | tabulation of every agentic run (Instruct and Think, web page and email, stage comparison, planted facts, reasoning confound) | `scripts/summarize_agentic.py` |
| `agentic_pilot` | fetch-initiation probes and the first agentic span scores | `scripts/probe_fetch_init.py`, `scripts/score_agentic_spans.py` |
| `inj_main`, `inj_pilot`, `inj_frame`, `inj_frame2`, `inj_real` | Mac and A100 pilots that preceded the scaled runs; `inj_main/INTERIM_RESULTS.md` is the first write-up | `scripts/run_injection.py` |
| `cue01`, `pilot01` | the abandoned hint-monitor project: cue-channel smoke test and the 40-question pilot | `scripts/run_generation.py`, `scripts/summarize_run.py` |

Every table needs its denominator and token cap next to it. Compliance rates are over finished items; truncation counts are in the summaries.
