"""p_1 Task 2 - shared helpers for the hand-authored seed bank.

Every core fact in the corpus is authored here as a "seed": a topic tag, a
one-line key fact for reviewers, 3-7 natural question phrasings, a 2-6
sentence model answer, and pre-written variation material (alternative
sentence phrasings / an extra closing sentence) so the expansion step can
produce several near-duplicate training pairs that still read naturally.

No dependencies, no randomness, ASCII + a few safe unicode chars only.
"""
from __future__ import annotations

SOURCE_MAIN = "Authored: mainstream public health guidance (NHS/WHO/Mayo style)"


def S(sid, topic, fact, qs, a, alts=None, close=None, flags=()):
    """One hand-authored core fact.

    sid    short stable id (documentation / review anchor)
    topic  topic tag: nutrition | sleep | hygiene | first_aid | conditions | activity
    fact   one-line key fact (lets a reviewer audit the fact bank quickly)
    qs     3-7 natural question phrasings (varied length and openers)
    a      2-6 answer sentences, canonical order
    alts   {sentence_index: alternative sentence} used for answer variation
    close  optional extra closing sentence (variant = a + [close])
    flags  tuple of reasons -> every generated pair is listed in nuanced_flags.json
    """
    assert isinstance(sid, str) and sid, "sid required"
    assert isinstance(topic, str) and topic
    assert 2 <= len(a) <= 6, (sid, "answer must be 2-6 sentences")
    assert len(qs) >= 3, (sid, "need at least 3 question phrasings")
    assert len(set(qs)) == len(qs), (sid, "duplicate questions inside seed")
    assert alts or close, (sid, "need alts or close so >=2 answer variants exist")
    if close:
        assert len(a) + 1 <= 6, (sid, "close would push answer past 6 sentences")
    for s in a:
        assert s and s[-1] in ".!?", (sid, "answer sentence must end with punctuation")
    if alts:
        for k, v in alts.items():
            assert 0 <= k < len(a), (sid, "alt index out of range")
            assert v and v[-1] in ".!?", (sid, "alt sentence must end with punctuation")
    for reason in flags:
        assert isinstance(reason, str) and reason
    return dict(
        id=sid, topic=topic, fact=fact, qs=list(qs), a=list(a),
        alts=dict(alts or {}), close=close, flags=tuple(flags),
    )
