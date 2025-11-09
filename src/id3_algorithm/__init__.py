"""Core package exposing helpers to train ID3 decision trees."""

from . import cli, entropy, variance, tree

__all__ = [
    "cli",
    "entropy",
    "variance",
    "tree",
]
