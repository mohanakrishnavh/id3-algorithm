"""Tests for id3_algorithm.entropy._information_gain against hand-worked
splits.

`_information_gain` picks the attribute that maximises information
gain (i.e. the entropy reduction from splitting on it). Each dataset
below is constructed so the winning attribute is known by hand
construction, independent of the implementation:

- test_information_gain_prefers_the_perfectly_discriminating_attribute:
  attribute "A" perfectly separates the two classes (gain = 1.0 bit,
  the maximum possible for a binary root with a 50/50 class split),
  while attribute "B" is uncorrelated with the class (gain = 0.0).
- test_information_gain_prefers_higher_gain_over_lower_gain: a milder
  version of the same idea with an attribute that only partially
  separates the classes, to check it isn't just detecting "perfect".
"""

from __future__ import annotations

import pandas as pd

from id3_algorithm.entropy import _information_gain, entropy


def _perfectly_separated_dataset() -> pd.DataFrame:
    # A perfectly predicts Class (A=0 -> Class=0, A=1 -> Class=1).
    # B is split 50/50 within each value of A, so it carries no
    # information about Class at all: IG(B) = 0.
    return pd.DataFrame(
        {
            "A": [0, 0, 0, 0, 1, 1, 1, 1],
            "B": [0, 0, 1, 1, 0, 0, 1, 1],
            "Class": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_information_gain_prefers_the_perfectly_discriminating_attribute():
    dataset = _perfectly_separated_dataset()
    total_entropy = entropy(
        positive_count=int((dataset["Class"] == 1).sum()),
        negative_count=int((dataset["Class"] == 0).sum()),
    )
    assert total_entropy == 1.0  # 4 positives, 4 negatives: max uncertainty

    best_attribute = _information_gain(total_entropy, dataset, ["A", "B"])
    assert best_attribute == "A"

    # Order in the attribute list must not matter.
    best_attribute_reordered = _information_gain(total_entropy, dataset, ["B", "A"])
    assert best_attribute_reordered == "A"


def test_information_gain_prefers_higher_gain_over_lower_gain():
    # C only partially separates the classes (3/4 correct within each
    # branch) while D is exactly as uninformative as B above. C's gain
    # is strictly between 0 and 1, but it must still beat D's gain of 0.
    dataset = pd.DataFrame(
        {
            "C": [0, 0, 0, 0, 1, 1, 1, 1],
            "D": [0, 0, 1, 1, 0, 0, 1, 1],
            "Class": [0, 0, 0, 1, 1, 1, 1, 0],
        }
    )
    total_entropy = entropy(
        positive_count=int((dataset["Class"] == 1).sum()),
        negative_count=int((dataset["Class"] == 0).sum()),
    )
    assert total_entropy == 1.0  # still 4/4

    best_attribute = _information_gain(total_entropy, dataset, ["C", "D"])
    assert best_attribute == "C"
