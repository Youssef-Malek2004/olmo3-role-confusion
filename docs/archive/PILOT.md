# Pilot: establish feasibility before scaling

Current permission is scaffolding only. These are future steps, not commands already run.

## First experiment

1. Review `configs/pilot.json` and record model/dataset revision hashes when downloads are authorized.
2. Select 40 pilot questions with a fixed subject mix. Keep them out of main train/validation/test splits.
3. Assign a seeded wrong option independent of all model responses. Reuse it across checkpoints.
4. Run clean and hinted versions on SFT and final RLVR, initially using greedy decoding and the same 2,048-token cap. This is 160 responses, not 40.
5. Save rendered prompts, generated text, final answer, token count, completion status, and prompt-end vectors.
6. Manually check parsing, at least several positive/negative cases, and every failure category. Read randomly selected complete traces.
7. Use a small fixed subset for neutral-sentence controls; check repeated greedy runs for reproducibility.
8. Record throughput, class counts, valid-pair coverage, rough main-run cost, and active time. Do not judge probe reliability on this tiny development set.

## If the model always rejects the wrong hint

This is a genuine risk, not an implementation nuisance to hide.

| Observed behaviour | Consequence |
|---|---|
| Neither stage switches | There is no viable switching-detection task in this setup. |
| SFT switches but final RLVR never switches | Resistance may have improved on these inputs. Target recall/AUROC cannot be estimated with no positives; report false alarms if useful. |
| Both stages sometimes switch | The detector study may be feasible if the eventual sample sizes are sufficient. |

Use moderately difficult questions the model can often solve unhinted. Very easy questions may produce no switches; impossibly difficult questions increase guessing. Make at most one deliberate pilot-only adjustment to subject difficulty or hint format, then reassess. A professor suggestion and answer-key metadata are two candidates. Freeze the chosen setup before touching held-out test outcomes.

Do not use “ignore the question and output C” as a silent replacement: that changes the construct toward direct instruction following. Do not select positive test cases or continuously strengthen hints until a preferred effect appears.

If resistance persists, stop expanding the detector study. Measuring changes in answer probabilities is a possible separate question, but immediate answer logits may not represent a thinking model's eventual answer; it requires a new protocol. A failed pilot is not automatically a strong final application.

## Decision record

Before the main run, write one entry in `docs/DECISION_LOG.md` stating:

- Valid pairs and switch-positive counts for each checkpoint.
- Parser and truncation failures and their resolution.
- Whether the neutral intervention shows comparable instability.
- Expected positive counts under the proposed main sample allocation.
- Measured generation rate and expected main cost.
- Final task, hint format, output cap, model revisions, and reason to proceed or stop.

Rough planning floors in the full plan are 40 positive source training examples and 20 positive target test examples. They are not power guarantees. A zero-switch 40-question pilot does not prove immunity, and a handful of switches does not validate a probe.
