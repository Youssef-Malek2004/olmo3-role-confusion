# Related work and references

Links marked (verify) were cited from memory or from indexed excerpts during the project and should be checked before being quoted.

## Directly related

- **Prompt Injection as Role Confusion** (ICML 2026). Diagnoses indirect prompt injection as the model perceiving the *source* of text by how it sounds; role probes predict attack success on gpt-oss and o4-mini; leaves defenses open. This project reproduces the role-score result on an open model with a public pipeline and tests the turn-conditioned steering defense. (verify)
- **AgentDojo** ([arXiv 2406.13352](https://arxiv.org/abs/2406.13352)). Source of the "important instructions" injection style used as the `tc2_important` framing.
- **CoT Forgery** (style-matched fake reasoning; destyling collapses the attack). Source of the `tc2_cot_forgery` / `tc3_cot_destyled` pair. (verify)
- **ChatInject / special-token fake turns** and **fake-completion dialogues**. Source of the `tc2_fake_turn`, `tc3_fake_turn_escaped`, `tc3_fake_turn_lexical`, and `tc2_fake_completion` framings. (verify)
- **OLMo 3** model card and [technical report](https://arxiv.org/abs/2512.13961); checkpoints [allenai/Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT), `-Think-DPO`, `-Think`, and the `-Instruct` SFT/DPO/RL line. The Dolci-Think-SFT dataset card documents that the Think SFT mix contains no tool-use data, which explains the fetch-initiation result in the research log.
- **MMLU** ([cais/mmlu](https://huggingface.co/datasets/cais/mmlu), [original repository](https://github.com/hendrycks/test)).

## Background for the abandoned first project (hint-monitor transfer)

- [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410). Paired hint intervention; influence versus acknowledgment.
- [Monitor drift](https://github.com/vaiyr/monitor-drift). Detector transfer across post-training.
- [Length Penalties Make Chain-of-Thought Less Monitorable](https://arxiv.org/abs/2607.09786); [Monitorability as a Free Gift](https://arxiv.org/abs/2602.03978); [Faithfulness as Information Flow](https://arxiv.org/abs/2605.24286); [FACE-Eval](https://arxiv.org/abs/2608.29464); [Lie to Me](https://arxiv.org/abs/2603.22582).
- Exploratory OLMo prompt-end probes (intentionallydense.github.io) and the Second Look OLMo checkpoint comparison were read as indexed excerpts, not reproduced.
