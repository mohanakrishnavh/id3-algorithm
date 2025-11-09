"""Variance impurity driven ID3 implementation with optional post-pruning."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from . import tree


TARGET_COLUMN = "Class"


@dataclass
class VarianceRunResult:
    """Container storing metrics for a variance impurity training run."""

    tree: tree.BTree
    pruned_tree: tree.BTree
    accuracy_before_pruning: float
    accuracy_after_pruning: float


def variance_impurity(positive_count: int, negative_count: int) -> float:
    """Return the variance impurity for the supplied class counts."""

    total = positive_count + negative_count
    if total == 0:
        return 0.0
    if positive_count == 0 or negative_count == 0:
        return 0.0
    probability_pos = positive_count / total
    probability_neg = negative_count / total
    return probability_pos * probability_neg


def _variance_gain(
    total_variance: float, dataset: pd.DataFrame, attributes: Sequence[str]
) -> str:
    """Return the attribute that maximises variance gain."""

    gains: List[float] = []
    for attribute in attributes:
        subtotal = 0.0
        for value in (0, 1):
            subset = dataset.loc[dataset[attribute] == value]
            negative = len(subset.loc[subset[TARGET_COLUMN] == 0])
            positive = len(subset.loc[subset[TARGET_COLUMN] == 1])
            impurity = variance_impurity(positive, negative)
            subtotal += ((negative + positive) / len(dataset)) * impurity
        gains.append(total_variance - subtotal)
    max_index = gains.index(max(gains))
    return attributes[max_index]


def _create_node(
    decision_tree: tree.BTree,
    parent_idx: int | None,
    branch_value: int | None,
    value: str,
    zeroes: int,
    ones: int,
) -> int:
    if parent_idx is None:
        if decision_tree.get_root() is None:
            return decision_tree.add_root(value, zeroes, ones)
        raise ValueError("Root already exists")
    if branch_value == 0:
        return decision_tree.add_left_child(parent_idx, value, zeroes, ones)
    if branch_value == 1:
        return decision_tree.add_right_child(parent_idx, value, zeroes, ones)
    raise ValueError("Branch value must be 0 or 1 for non-root nodes")


def _build_tree(
    dataset: pd.DataFrame,
    attributes: Sequence[str],
    decision_tree: tree.BTree,
    parent_idx: int | None = None,
    branch_value: int | None = None,
) -> None:
    ones = len(dataset.loc[dataset[TARGET_COLUMN] == 1])
    zeroes = len(dataset.loc[dataset[TARGET_COLUMN] == 0])

    if ones == 0:
        _create_node(decision_tree, parent_idx, branch_value, "0", zeroes, ones)
        return
    if zeroes == 0:
        _create_node(decision_tree, parent_idx, branch_value, "1", zeroes, ones)
        return
    if not attributes:
        label = "1" if ones >= zeroes else "0"
        _create_node(decision_tree, parent_idx, branch_value, label, zeroes, ones)
        return

    total_variance = variance_impurity(ones, zeroes)
    best_attribute = _variance_gain(total_variance, dataset, attributes)

    current_idx = _create_node(
        decision_tree, parent_idx, branch_value, best_attribute, zeroes, ones
    )

    remaining_attributes = [attribute for attribute in attributes if attribute != best_attribute]
    for value in (0, 1):
        subset = dataset.loc[dataset[best_attribute] == value]
        if subset.empty:
            label = "1" if ones >= zeroes else "0"
            _create_node(decision_tree, current_idx, value, label, zeroes, ones)
        else:
            _build_tree(subset, remaining_attributes, decision_tree, current_idx, value)


def construct_tree(dataset: pd.DataFrame) -> tree.BTree:
    """Train an ID3 decision tree using variance impurity as the split criterion."""

    attributes = list(dataset.drop(columns=[TARGET_COLUMN]).columns)
    decision_tree = tree.BTree()
    _build_tree(dataset, attributes, decision_tree)
    return decision_tree


def calculate_accuracy(dataset: pd.DataFrame, decision_tree: tree.BTree) -> float:
    """Return the classification accuracy of ``decision_tree`` on ``dataset``."""

    predictors = dataset.drop(columns=[TARGET_COLUMN])
    root = decision_tree.get_root()
    if root is None:
        raise ValueError("Decision tree has no root node")

    matches = 0
    for index, row in predictors.iterrows():
        prediction = decision_tree.traverse(root, row)
        if prediction == dataset[TARGET_COLUMN].iloc[index]:
            matches += 1
    return (matches / len(dataset)) * 100


def post_pruning(
    l: int,
    k: int,
    validation: pd.DataFrame,
    base_tree: tree.BTree,
    print_tree: bool = False,
) -> tree.BTree:
    """Apply randomized post-pruning to a variance based decision tree."""

    best_tree = copy.deepcopy(base_tree)
    current_best = calculate_accuracy(validation, best_tree)
    for _ in range(max(l, 1)):
        candidate = copy.deepcopy(base_tree)
        for _ in range(max(random.randint(1, max(k, 1)), 1)):
            non_leaves = candidate.get_non_leaves()
            if not non_leaves:
                break
            node_to_prune = random.choice(non_leaves)
            node_to_prune.left = None
            node_to_prune.right = None
            node_to_prune.value = "1" if node_to_prune.ones >= node_to_prune.zeroes else "0"
        candidate_accuracy = calculate_accuracy(validation, candidate)
        if candidate_accuracy > current_best:
            best_tree = candidate
            current_best = candidate_accuracy
    if print_tree:
        print("Pruned variance tree:")
        best_tree.print_btree()
    return best_tree


def run_variance(
    training_path: str,
    validation_path: str,
    test_path: str,
    l: int,
    k: int,
    print_tree: bool = False,
) -> VarianceRunResult:
    """Train, prune, and evaluate a variance impurity based ID3 tree."""

    training = pd.read_csv(training_path)
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)

    decision_tree = construct_tree(training)
    if print_tree:
        print("Variance tree before pruning:")
        decision_tree.print_btree()

    accuracy_before = calculate_accuracy(test, decision_tree)
    pruned_tree = post_pruning(l, k, validation, decision_tree, print_tree)
    accuracy_after = calculate_accuracy(test, pruned_tree)

    return VarianceRunResult(
        tree=decision_tree,
        pruned_tree=pruned_tree,
        accuracy_before_pruning=accuracy_before,
        accuracy_after_pruning=accuracy_after,
    )