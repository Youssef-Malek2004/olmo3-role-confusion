# [AI-DRAFT] Experiment map: Think line vs Instruct line

Review before sharing. Status as of Sept 7, 2026, 21:30 local. "Done" means results exist in `results/generated/` or `artifacts/runs/`; "queued" means in a running pod chain; "backlog" means not scheduled.

## The question at the root

Do post-trained language models treat text that arrives from a tool as if a user had said it? Where in the model does that decision live, how does it change across post-training, and can the internal signal be used to detect or block it?

## Mind graph

```mermaid
mindmap
  root((Tool output treated as instruction?))
    Shared scaffolding
      MMLU items, 440 q, 10 subjects
      Payload set: 3 base goals, 4 realistic goals, 16 framings, 4 ablations
      Detector: visible-answer compliance, send to attacker, key sent, echo stripping
      Controls: clean floors, length-matched benign text, random vector, prompt warning
    Think line: SFT, DPO, RLVR
      Behaviour across stages
        Base goals, 3 draws, 3 stages: done
        Framing matrix: SFT, RLVR done; DPO running
        Realistic goals: SFT done; RLVR, DPO queued
        Silent compliance and user attribution in think blocks: done
      Representation
        User vs tool probes, 5 blocks, 3 stages: done
        Cross-stage direction stability: done
        Role score predicts item compliance: SFT, RLVR done; DPO queued
        Own-voice direction from think openers: SFT, RLVR done; DPO queued
        Two-axis type map: done
      Intervention
        Role steering vs marker, 3 stages, dose response: done
        Copy-vs-trust: 4x makes the tool turn unreadable, random does not; attack unblocked at 1x and 2x: done at SFT
        Role vs style vs sum vs imitation framings: done
        Random and prompt controls: done, incl. random at norm 16.6
        Utility on legitimate tool use: done, accuracy unchanged under 4x
        Role 2x additivity baseline: done, inert alone; sum superadditive at SFT
      Context dependence
        Web page and email, scripted call: fake turn about half, imitation payloads near zero
        Diagnostics: pipeline, wording, user sentence ruled out; form vs tool identity running on the Mac
      Reasoning removed
        Empty think prefill: compliance falls on every payload, forgery 35 to 10, important 43 to 8
      Not suited
        Self-initiated tool use: 0 of 40 fetches
        Agentic multi-round design
    Instruct line: released checkpoint
      Tool policy
        Fetch initiation probe: 4 of 4, stops and waits: done
        Native call syntax name k equals v: handled
      Agentic injection
        Web page and email, self-fetched result: pilot done 4 of 4 exfil
        7 payloads x 6, web page, Mac: 19 of 42 exfil, forgery 6 of 6
        At scale, 8 payloads, 2 contexts, 2 draws: done, forgery 100 percent, destyled 8 to 16
      Reasoning confound
        reason() tool after fetch: voice attacks unchanged, important block and planted fact dented inconsistently
        record_final tool raises important-block compliance from near 0 to 83 percent
      Not suited
        Own-voice axis, no think block
        Silent compliance analysis
      Representation
        User vs tool probe: AUROC 0.9999, done
        Cross-line cosine with Think 0.79 to 0.58 by block: done
        Role score on agentic items: own dir 0.70, Think dir 0.83: done
      Backlog
        Framing matrix with scripted call, for a direct Think comparison
        Steering on Instruct
        SFT, DPO, RL stage comparison on Instruct: done at 12 per payload; impersonal payloads fall at RL, voice at ceiling; within-line cosine 0.94 to 0.999
```

## Which line answers which question

| question | line | why | status |
|---|---|---|---|
| Does compliance with tool-borne instructions change across post-training stages? | Think | Three pinned checkpoints of one lineage, reasoning RL is the interesting transition | done, DPO framings running |
| Does the model know who is speaking, and does that change? | Think | Probes and cross-stage cosine need the same lineage at several stages | done |
| Does the internal role score predict which injections succeed? | Think | Uses the matrix plus prefill activations | done for SFT, RLVR; DPO queued |
| Does the model mention the instruction, and whom does it credit? | Think | Requires the think block | done |
| Is there a second, own-voice axis behind forged reasoning? | Think | Direction is built from the model's think openers | done, 4x norm confound pending |
| Can steering along the role direction block the attacks, at what dose, with what side effects? | Think | Directions exist, HF steering hooks, all three stages | done; utility and random at 16.6 queued |
| Does a model trained to use tools get subverted inside a result it fetched itself? | Instruct | Only line with a trained stop-and-wait tool policy | pilot done, scale queued |
| Do the same payloads work when the tool exchange is self-initiated rather than scripted? | Instruct, compared with Think scripted | Same payloads and detector on both | queued |
| Does the fake-turn attack depend on the model having a tool policy at all? | Both | Think shows pattern completion without a policy; Instruct shows a trained policy being subverted | pilot evidence, scale queued |
| Are role directions the same on a tool-trained model? | Instruct | Same base model, directions comparable | done: cosine 0.67 at block 16 across lines; 0.94 to 0.999 within Instruct stages |
| Does compliance change across Instruct's own stages? | Instruct | Three checkpoints, agentic contexts | done: impersonal payloads drop at RL, voice payloads at ceiling |
| Does role steering work on a tool-trained model? | Instruct | Needs HF steering time | backlog |
| Is the Instruct-Think gap due to the reasoning block? | Both | Add native reasoning to Instruct, remove it from Think | done: no; the block is Think's attack surface, not a defense |
| Does role steering remove the injected span's authority, or break copying from the tool turn? | Think SFT | Verification code exists only in the tool result; user asks for it under each dose | done at SFT (1x, 2x, 4x, random 4x): copying collapses at 4x, attack unblocked below 4x; the defense is a channel trade-off. RLVR not run |

## How the two lines relate in the write-up

- Think is the primary study: stage trend, stable role representation, role score as predictor, two orthogonal directions, steering with controls, honest limits.
- Instruct is the generality check for the tool-call claims: the attacks that produce actions in a scaffolded Think setting also produce them in a genuinely agentic Instruct setting, on a model with trained tool use.
- The training-data finding ties them together: the Think SFT mix has no tool-use data, which explains both the missing initiation in Think and why the Instruct comparison is needed.

## Findings that belong to neither line alone

- Fake-turn text works whether or not the delimiter tokens are real, so it is a model-level failure, not a serialisation bug.
- Steering scrubs the injected span's specifics before it removes the action: exfil goes to zero while a placeholder send can survive. Report both metrics.
- Tool-call rates need the spontaneous-call floor from clean records beside them.
