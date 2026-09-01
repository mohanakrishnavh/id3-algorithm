"""Tests for id3_algorithm.entropy.entropy against hand-worked values.

Each expected value below is derived independently from the standard
binary-entropy formula H(p) = -p*log2(p) - (1-p)*log2(1-p), not by
calling the implementation under test, so these actually check
correctness rather than just guard against regressions.
"""

from __future__ import annotations

import math

import pytest

from id3_algorithm.entropy import entropy


def test_entropy_of_pure_positive_set_is_zero():
    # No negatives at all: there is nothing left to be uncertain about.
    assert entropy(positive_count=5, negative_count=0) == 0.0


def test_entropy_of_pure_negative_set_is_zero():
    assert entropy(positive_count=0, negative_count=5) == 0.0


def test_entropy_of_empty_set_is_zero():
    assert entropy(positive_count=0, negative_count=0) == 0.0


def test_entropy_of_perfectly_balanced_set_is_one():
    # p = n = 0.5 is the maximum-uncertainty case for a binary class:
    # H(0.5) = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0 bit, exactly.
    assert entropy(positive_count=2, negative_count=2) == pytest.approx(1.0)
    assert entropy(positive_count=4, negative_count=4) == pytest.approx(1.0)


def test_entropy_matches_the_binary_entropy_formula():
    # 1 positive, 3 negatives: p = 0.25.
    p = 0.25
    expected = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    assert entropy(positive_count=1, negative_count=3) == pytest.approx(expected)
    assert expected == pytest.approx(0.8112781244591328)

    # 3 positives, 5 negatives: p = 0.375.
    p = 3 / 8
    expected = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    assert entropy(positive_count=3, negative_count=5) == pytest.approx(expected)
    assert expected == pytest.approx(0.954434002924965)


def test_entropy_is_symmetric_in_positive_and_negative_counts():
    # Entropy only depends on the class *distribution*, not on which
    # class is labelled "positive" — swapping the counts must not
    # change the result.
    assert entropy(3, 7) == pytest.approx(entropy(7, 3))
