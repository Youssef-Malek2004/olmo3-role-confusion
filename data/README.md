# Data

`processed/questions.jsonl` is the frozen question table: 440 MMLU test questions from ten subjects (astronomy, college medicine, conceptual physics, high-school biology / chemistry / macroeconomics / statistics, logical fallacies, philosophy, professional psychology), 4 / 20 / 10 / 10 per subject for pilot / train / validation / test, built by `scripts/prepare_questions.py` from the pinned `cais/mmlu` revision. 105 exact duplicates were dropped dataset-wide before splitting. Each row carries a content-hash ID, the answer key, and a planted wrong letter seeded from the question ID only. `marker12_item_ids.txt` is the fixed 12-item set used for the Mac defense arms; `neutral_control_ids.json` is the neutral-sentence subset of the abandoned hint pilot.

Raw dataset and model caches go under the ignored `.cache/`.
