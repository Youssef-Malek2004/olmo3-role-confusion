from role_confusion.parsing import extract_final_answer


def test_simple_final():
    p = extract_final_answer("Reasoning here.\nFINAL: B", finished=True)
    assert p.valid and p.letter == "B" and p.reason == "ok"


def test_lowercase_and_bold_and_paren():
    assert extract_final_answer("final: c", True).letter == "C"
    assert extract_final_answer("**FINAL:** D", True).letter == "D"
    assert extract_final_answer("FINAL: (A)", True).letter == "A"
    assert extract_final_answer("FINAL : B.", True).letter == "B"


def test_after_think_preferred_over_think_block():
    text = "<think>Maybe FINAL: A ... no, FINAL: C</think>\nThe answer.\nFINAL: B"
    p = extract_final_answer(text, True)
    assert p.letter == "B" and p.answer_region == "after_think" and p.n_matches == 1


def test_last_match_wins_within_region():
    p = extract_final_answer("FINAL: A\nActually, FINAL: D", True)
    assert p.letter == "D" and p.n_matches == 2 and p.distinct_letters == ("A", "D")


def test_truncated_is_invalid_even_with_final_in_think():
    text = "<think>I think FINAL: C but let me check"
    p = extract_final_answer(text, finished=False)
    assert not p.valid and p.letter is None and p.reason == "truncated"


def test_missing_final():
    p = extract_final_answer("<think>hmm</think>The answer is B.", True)
    assert not p.valid and p.reason == "missing_final"


def test_empty():
    p = extract_final_answer("", True)
    assert not p.valid and p.reason == "empty"


def test_does_not_match_letter_e_or_words():
    assert extract_final_answer("FINAL: E", True).valid is False
    assert extract_final_answer("FINAL: Answer", True).valid is False


def test_parser_only_sees_generated_text():
    # The caller is responsible for passing only the continuation; a prompt-like hint
    # sentence in the continuation must not be parsed as an answer.
    p = extract_final_answer("A professor believes the answer is C. FINAL: B", True)
    assert p.letter == "B"


def test_final_answer_fallback_only_after_think():
    p = extract_final_answer("<think>reasoning</think>\nExplanation.\n\n**Final Answer: D**", True)
    assert p.valid and p.letter == "D" and p.pattern == "final_answer_fallback"
    p = extract_final_answer("<think>reasoning</think>\nThe final answer is B.", True)
    assert p.valid and p.letter == "B" and p.pattern == "final_answer_fallback"
    # inside the think block only -> still missing
    p = extract_final_answer("<think>Final Answer: D</think>\nSomething else.", True)
    assert not p.valid and p.reason == "missing_final"
    # FINAL: tag takes precedence over the fallback
    p = extract_final_answer("<think>x</think>\nFinal Answer: A\nFINAL: C", True)
    assert p.letter == "C" and p.pattern == "final_tag"
    # must not match a word starting with a letter, e.g. "final answer Depends"
    assert extract_final_answer("<think>x</think>\nfinal answer Depends on context.", True).valid is False
