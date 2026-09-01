"""End-to-end tests for id3_algorithm.entropy.construct_tree /
calculate_accuracy.

Note: the original plan for this file was "train on the bundled data,
assert accuracy on a held-out split doesn't regress" — but this repo
doesn't actually ship any training/validation/test CSVs (only
`reports/*.txt` output from a prior run). `run_entropy()` / the CLI
expect the caller to supply dataset paths; there's nothing bundled to
train on here. These tests build small synthetic datasets in-memory
instead, with hand-known correct trees, which lets them assert on the
*actual* resulting structure rather than just "accuracy didn't drop".
"""

from __future__ import annotations

import pandas as pd

from id3_algorithm.entropy import calculate_accuracy, construct_tree


def test_construct_tree_on_a_perfectly_separable_dataset():
    # Same dataset as in test_information_gain.py: A alone perfectly
    # predicts Class, so the tree should be exactly one split deep,
    # with A at the root and two pure leaves.
    dataset = pd.DataFrame(
        {
            "A": [0, 0, 0, 0, 1, 1, 1, 1],
            "B": [0, 0, 1, 1, 0, 0, 1, 1],
            "Class": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    decision_tree = construct_tree(dataset)
    root = decision_tree.get_root()

    assert root is not None
    assert root.value == "A"
    assert root.left is not None and root.left.value == "0"
    assert root.right is not None and root.right.value == "1"
    # Both leaves: no further children.
    assert root.left.left is None and root.left.right is None
    assert root.right.left is None and root.right.right is None


def test_calculate_accuracy_is_perfect_when_the_tree_fits_the_data():
    dataset = pd.DataFrame(
        {
            "A": [0, 0, 0, 0, 1, 1, 1, 1],
            "B": [0, 0, 1, 1, 0, 0, 1, 1],
            "Class": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    decision_tree = construct_tree(dataset)
    assert calculate_accuracy(dataset, decision_tree) == 100.0


def test_calculate_accuracy_on_unseen_data_with_the_same_pattern():
    training = pd.DataFrame(
        {
            "A": [0, 0, 0, 0, 1, 1, 1, 1],
            "B": [0, 0, 1, 1, 0, 0, 1, 1],
            "Class": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    decision_tree = construct_tree(training)

    # A held-out set following the same A -> Class rule the tree
    # learned; B is irrelevant to the label so its values don't matter.
    held_out = pd.DataFrame(
        {
            "A": [0, 1, 0, 1],
            "B": [1, 1, 0, 0],
            "Class": [0, 1, 0, 1],
        }
    )
    assert calculate_accuracy(held_out, decision_tree) == 100.0


def test_construct_tree_falls_back_to_the_majority_label_with_no_attributes():
    # With no attributes to split on (this exercises the same
    # `if not attributes:` branch that recursion hits once every
    # attribute has been used up) and a class column that still
    # mixes 0s and 1s, the node should be labelled with the majority
    # class rather than raising.
    dataset = pd.DataFrame({"Class": [1, 1, 0]})
    decision_tree = construct_tree(dataset)
    root = decision_tree.get_root()
    assert root is not None
    assert root.value == "1"  # 2 ones vs 1 zero
