"""Question records, deterministic IDs, duplicate checks, group splits, planted wrong options.

Everything here is pure Python so it can be tested without model or dataset downloads.
The HF dataset loader lives in ``load_mmlu`` and is only called by scripts.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Question:
    question_id: str
    subject: str
    question: str
    choices: tuple[str, ...]
    answer_index: int
    source_row: int | None = None

    @property
    def answer_letter(self) -> str:
        return LETTERS[self.answer_index]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["choices"] = list(self.choices)
        d["answer_letter"] = self.answer_letter
        return d


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def content_key(subject: str, question: str, choices: Iterable[str]) -> str:
    """Normalized content string used for both the ID hash and duplicate detection.

    The subject is deliberately excluded from the duplicate key: the same question text
    appearing under two subjects is still a duplicate for split purposes.
    """
    parts = [_normalize(question)] + [_normalize(c) for c in choices]
    return "\x1f".join(parts)


def make_question_id(subject: str, question: str, choices: Iterable[str]) -> str:
    digest = hashlib.sha1(content_key(subject, question, choices).encode("utf-8")).hexdigest()
    return f"mmlu-{subject}-{digest[:10]}"


def build_question(subject: str, question: str, choices: list[str], answer_index: int,
                   source_row: int | None = None) -> Question:
    if len(choices) != 4:
        raise ValueError(f"expected 4 choices, got {len(choices)}")
    if not 0 <= answer_index <= 3:
        raise ValueError(f"answer index out of range: {answer_index}")
    return Question(
        question_id=make_question_id(subject, question, choices),
        subject=subject,
        question=question.strip(),
        choices=tuple(c.strip() for c in choices),
        answer_index=int(answer_index),
        source_row=source_row,
    )


def deduplicate(questions: list[Question]) -> tuple[list[Question], list[tuple[str, str]]]:
    """Drop exact content duplicates (ignoring subject). Returns kept questions and dropped pairs.

    The first occurrence in input order is kept. Dropped pairs are (dropped_id, kept_id).
    """
    seen: dict[str, str] = {}
    kept: list[Question] = []
    dropped: list[tuple[str, str]] = []
    for q in questions:
        key = content_key(q.subject, q.question, q.choices)
        if key in seen:
            dropped.append((q.question_id, seen[key]))
            continue
        seen[key] = q.question_id
        kept.append(q)
    ids = [q.question_id for q in kept]
    if len(set(ids)) != len(ids):
        raise RuntimeError("question ID collision after deduplication")
    return kept, dropped


SPLIT_ORDER = ("pilot", "train", "validation", "test")


def assign_splits(questions: list[Question], per_subject_counts: dict[str, int], seed: int,
                  subjects: list[str] | None = None) -> dict[str, str]:
    """Stratified-by-subject, seeded assignment of each question ID to exactly one split.

    ``per_subject_counts`` maps split name to the number of questions per subject. Within a
    subject the IDs are sorted, shuffled with a subject-specific seeded RNG, then sliced in the
    fixed order pilot -> train -> validation -> test. Because the shuffle depends only on
    (seed, subject, sorted IDs), adding or removing later splits never changes the pilot set,
    and the pilot never overlaps the main splits.
    """
    if subjects is None:
        subjects = sorted({q.subject for q in questions})
    by_subject: dict[str, list[str]] = {s: [] for s in subjects}
    for q in questions:
        if q.subject in by_subject:
            by_subject[q.subject].append(q.question_id)
    assignment: dict[str, str] = {}
    need = sum(per_subject_counts.get(s, 0) for s in SPLIT_ORDER)
    for subject in subjects:
        ids = sorted(by_subject[subject])
        if len(ids) < need:
            raise ValueError(f"subject {subject} has {len(ids)} questions; need {need}")
        rng = random.Random(f"{seed}:{subject}")
        rng.shuffle(ids)
        cursor = 0
        for split in SPLIT_ORDER:
            n = per_subject_counts.get(split, 0)
            for qid in ids[cursor:cursor + n]:
                assignment[qid] = split
            cursor += n
    return assignment


def planted_wrong_letter(question_id: str, answer_letter: str, seed: int) -> str:
    """Seeded uniform choice among the three incorrect letters, keyed only by question ID.

    This never sees model responses and is identical across checkpoints and stages.
    """
    wrong = [L for L in LETTERS if L != answer_letter]
    rng = random.Random(f"{seed}:planted:{question_id}")
    return rng.choice(wrong)


@dataclass
class PreparedQuestion:
    question: Question
    split: str
    planted_letter: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.question.to_dict()
        d["split"] = self.split
        d["planted_letter"] = self.planted_letter
        d.update(self.extra)
        return d


def prepare(questions: list[Question], per_subject_counts: dict[str, int], seed: int,
            subjects: list[str]) -> list[PreparedQuestion]:
    kept, _ = deduplicate(questions)
    assignment = assign_splits(kept, per_subject_counts, seed, subjects)
    out = []
    for q in kept:
        split = assignment.get(q.question_id)
        if split is None:
            continue
        planted = planted_wrong_letter(q.question_id, q.answer_letter, seed)
        assert planted != q.answer_letter
        out.append(PreparedQuestion(question=q, split=split, planted_letter=planted))
    return out


def load_mmlu(configuration: str = "all", split: str = "test", revision: str | None = None,
              subjects: list[str] | None = None) -> list[Question]:
    """Load MMLU rows from the Hugging Face hub. Requires ``datasets`` and network or cache."""
    from datasets import load_dataset  # imported lazily so pure logic stays importable offline

    ds = load_dataset("cais/mmlu", configuration, split=split, revision=revision)
    out: list[Question] = []
    for i, row in enumerate(ds):
        if subjects is not None and row["subject"] not in subjects:
            continue
        out.append(build_question(row["subject"], row["question"], list(row["choices"]),
                                  int(row["answer"]), source_row=i))
    return out
