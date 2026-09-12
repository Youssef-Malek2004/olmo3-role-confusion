"""Final-answer extraction from model-generated text only.

The parser must never see the prompt. Callers pass the decoded continuation, not the full
sequence. ``finished`` is whether generation stopped on an end-of-sequence token rather than
the token cap; truncated outputs are unscorable even if a FINAL line appears somewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FINAL_RE = re.compile(r"FINAL\s*:\s*\**\s*\(?\s*([ABCD])\b", re.IGNORECASE)
# Fallback tier, added during the pilot after observing "**Final Answer: D**" without the FINAL: tag.
# Only consulted when no FINAL: match exists in the answer region. Counted separately via ``pattern``.
FALLBACK_RE = re.compile(r"final\s+answer\s*(?:is|:)?\s*\**\s*\(?\s*([ABCD])\b(?![a-z])", re.IGNORECASE)
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class ParsedAnswer:
    letter: str | None
    valid: bool
    reason: str  # "ok", "truncated", "missing_final", "empty"
    n_matches: int
    distinct_letters: tuple[str, ...]
    answer_region: str  # "after_think" or "whole"
    pattern: str = "final_tag"  # "final_tag" | "final_answer_fallback" | "none"


def extract_final_answer(generated_text: str, finished: bool) -> ParsedAnswer:
    text = generated_text or ""
    if not text.strip():
        return ParsedAnswer(None, False, "empty", 0, (), "whole", "none")

    # Prefer the visible answer after the last closing think tag when one exists.
    if THINK_CLOSE in text:
        region_text = text.rsplit(THINK_CLOSE, 1)[1]
        region = "after_think"
    else:
        region_text = text
        region = "whole"

    pattern = "final_tag"
    matches = [m.group(1).upper() for m in FINAL_RE.finditer(region_text)]
    if not matches and region == "after_think":
        # Fallback tier: "Final Answer: X" in the visible answer only (never inside the think block).
        matches = [m.group(1).upper() for m in FALLBACK_RE.finditer(region_text)]
        if matches:
            pattern = "final_answer_fallback"
    if not matches and region == "after_think":
        # Fall back to the whole text only for diagnostics; still record where it came from.
        matches = [m.group(1).upper() for m in FINAL_RE.finditer(text)]
        region = "whole"

    distinct = tuple(sorted(set(matches)))
    if not finished:
        return ParsedAnswer(None, False, "truncated", len(matches), distinct, region, pattern if matches else "none")
    if not matches:
        return ParsedAnswer(None, False, "missing_final", 0, (), region, "none")
    # Prespecified rule: the last matching line in the answer region wins.
    return ParsedAnswer(matches[-1], True, "ok", len(matches), distinct, region, pattern)
