#!/usr/bin/env python3
"""Standalone syllable counting for custom pipeline rate_syl computation.

This module is intentionally decoupled from the original project cfg/db modules.
It tries higher-quality resources when available and falls back gracefully.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# Optional resources.
_CMU_DICT: Optional[Dict[str, List[List[str]]]] = None
_HYPHENATE = None


def _init_optional_resources() -> None:
    global _CMU_DICT, _HYPHENATE

    # CMU pronouncing dictionary via NLTK (best quality when available).
    try:
        import nltk  # type: ignore

        _CMU_DICT = nltk.corpus.cmudict.dict()
    except Exception:
        _CMU_DICT = None

    # Secondary fallback if installed.
    try:
        import hyphenate  # type: ignore

        _HYPHENATE = hyphenate
    except Exception:
        _HYPHENATE = None


_init_optional_resources()


def _syllables_heuristic(word: str) -> int:
    """Simple fallback: vowel groups with minor English-specific adjustment."""
    groups = re.findall(r"[aeiouy]+", word)
    syllables = len(groups)
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def count_syllables(text: str) -> int:
    """Count syllables in text with CMU-dict -> hyphenate -> heuristic fallback."""
    total = 0
    for raw in text.split(" "):
        word = raw.strip().lower()

        # Keep apostrophes and ? for handling contractions/unknown tokens.
        word = re.sub(r"[^a-z?']", "", word)

        # Preprocessing aligned with original project behavior.
        if len(word) > 0 and word[-1] == "-":
            word = word[:-1]
        if len(word) > 1 and word.endswith("'s"):
            if _CMU_DICT is None or word not in _CMU_DICT:
                word = word[:-2]

        if len(word) == 0:
            continue

        # Unknown/unintelligible markers.
        if "?" in word:
            total += word.count("?")
            continue

        # Best case: dictionary lookup.
        if _CMU_DICT is not None and word in _CMU_DICT:
            pron = _CMU_DICT[word][0]
            total += sum(1 for p in pron if p and p[-1].isdigit())
            continue

        # Second fallback: hyphenate package if present.
        if _HYPHENATE is not None:
            try:
                total += len(_HYPHENATE.hyphenate_word(word))
                continue
            except Exception:
                pass

        # Final fallback: heuristic.
        total += _syllables_heuristic(word)

    return total
