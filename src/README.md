# `role_confusion` package

| module | contents |
|---|---|
| `injection.py` | tool-result injection items: the `lookup_question_record` scaffold, three base goals (marker, format, exfil), realistic goals (toolcall, answer, link, deny), 16 framings, three voices, the `send_summary` tool, and detectors that strip echoed records before matching |
| `agentic.py` | web-page and email contexts where the model calls `fetch_page` / `read_email` itself; round rendering, native and JSON call parsing, `reason()` and `record_final()` tools |
| `copy_trust.py` | copy-vs-trust conditions with a per-question verification code that exists only in the tool result |
| `generation.py` | model loading (MPS/CUDA, bf16), batched sampled generation with a static KV cache, prompt-end and span-mean activation capture, and `Steer`: a forward hook adding `alpha * unit_vector` at tool-turn token positions during prefill |
| `probe.py` | regularised logistic probes, shuffled-label controls, difference-of-means directions |
| `questions.py`, `prompts.py`, `parsing.py`, `labels.py`, `io.py` | question table, prompt rendering, final-answer parsing from generated text only, switching labels for the abandoned hint pilot, JSONL helpers |
