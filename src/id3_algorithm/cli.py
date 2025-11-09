"""Command line interface for running the ID3 decision tree experiments."""

from __future__ import annotations

import argparse
from typing import Literal

from . import entropy, variance


def _positive_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"yes", "y", "true", "1"}:
        return True
    if lowered in {"no", "n", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError("Expected yes or no")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ID3 decision tree experiments with entropy and variance heuristics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("L", type=int, help="L parameter for randomized post-pruning")
    parser.add_argument("K", type=int, help="K parameter for randomized post-pruning")
    parser.add_argument("training_set", help="Path to the training CSV file")
    parser.add_argument("validation_set", help="Path to the validation CSV file")
    parser.add_argument("test_set", help="Path to the test CSV file")
    parser.add_argument(
        "print_tree",
        type=_positive_flag,
        help="Print the trees before and after pruning (yes/no)",
    )
    parser.add_argument(
        "--metric",
        choices=["entropy", "variance", "both"],
        default="both",
        help="Which heuristic(s) to run",
    )
    return parser


def _print_result(title: str, result: entropy.EntropyRunResult | variance.VarianceRunResult) -> None:
    print(title)
    print(f"Accuracy before pruning : {result.accuracy_before_pruning:.2f}%")
    print(f"Accuracy after pruning  : {result.accuracy_after_pruning:.2f}%")


def run(
    l: int,
    k: int,
    training_set: str,
    validation_set: str,
    test_set: str,
    print_tree: bool,
    metric: Literal["entropy", "variance", "both"] = "both",
) -> dict[str, float]:
    """Execute the selected experiments and return a summary of accuracies."""

    summary: dict[str, float] = {}

    if metric in {"entropy", "both"}:
        entropy_result = entropy.run_entropy(
            training_set,
            validation_set,
            test_set,
            l,
            k,
            print_tree,
        )
        _print_result("Entropy heuristic", entropy_result)
        summary.update({
            "entropy_accuracy_before_pruning": entropy_result.accuracy_before_pruning,
            "entropy_accuracy_after_pruning": entropy_result.accuracy_after_pruning,
        })

    if metric in {"variance", "both"}:
        variance_result = variance.run_variance(
            training_set,
            validation_set,
            test_set,
            l,
            k,
            print_tree,
        )
        _print_result("Variance heuristic", variance_result)
        summary.update({
            "variance_accuracy_before_pruning": variance_result.accuracy_before_pruning,
            "variance_accuracy_after_pruning": variance_result.accuracy_after_pruning,
        })

    return summary


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(
        l=args.L,
        k=args.K,
        training_set=args.training_set,
        validation_set=args.validation_set,
        test_set=args.test_set,
        print_tree=args.print_tree,
        metric=args.metric,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()