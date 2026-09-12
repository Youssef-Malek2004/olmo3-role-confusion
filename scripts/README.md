# Scripts

Python entry points (run from the repo root with the venv active; each writes a manifest with model revision, decoding settings, cap, seed rule and code state):

| script | purpose |
|---|---|
| `pin_and_download.py`, `prepare_questions.py` | pin revisions, download checkpoints and MMLU, build the frozen 440-question table |
| `run_injection.py` | tool-result injection eval (HF), one checkpoint at a time; `--defense none|delimiter|steer|random`, span-mean activations per item |
| `run_injection_vllm.py` | the same items and detectors on vLLM for the large matrices (no hooks) |
| `run_agentic.py` | agentic version: the model calls `fetch_page`/`read_email` itself; three batched rounds; `--scripted`, `--no-think`, `--force-think`, `--reason tool`, `--final-tool` |
| `run_copy_vs_trust.py` | copy-vs-trust conditions (clean / retrieve_print / retrieve_send / attack) under an arm; `--report` tabulates |
| `capture_roles.py`, `fit_role_probe.py` | same text in user turn vs tool turn; token and mean probes; difference-of-means directions; cross-stage transfer |
| `capture_voice.py` | own-voice direction: think-style vs prose inside the tool turn |
| `score_spans.py`, `score_agentic_spans.py` | project injected spans onto a direction (prompt-only pass) |
| `summarize_injection.py`, `analyze_frame_def.py`, `summarize_agentic.py`, `select_alpha.py` | re-detect from raw text and tabulate; defense tables; agentic tables; pick a steering scale from a dev sweep |
| `make_figures.py` | the four figures in `figures/` from saved results |
| `run_generation.py`, `summarize_run.py`, `analyze_pilot.py`, `dump_audit.py` | the abandoned hint-monitor pilot |
| `probe_toolcalls.py`, `probe_fetch_init.py` | small behavioural probes (does the model emit a call; does it initiate a fetch) |
| `doctor.py`, `record_environment.py`, `setup.sh`, `gpu_setup.sh`, `remote.sh` | environment |

Shell chains are the exact unattended sequences that produced the runs, kept as the run record. `run_gpu_chain.sh`, `run_day2_*.sh`, `run_vllm_*.sh`, `gated_*.sh` ran on the A100; `run_rc_chain*.sh`, `run_mac_*.sh`, `run_realistic_chain.sh`, `run_ct_*.sh`, `run_pilot_*.sh` ran on the Mac. Later `rc_chain` numbers exist because the laptop GPU throttled and the queue was reordered; the research log explains each.
