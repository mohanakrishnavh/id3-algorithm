"""Utility classes for representing and traversing binary decision trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Node:
    """Single node inside a binary decision tree."""

    value: Optional[str]
    idx: int
    zeroes: int
    ones: int
    left: Optional["Node"] = None
    right: Optional["Node"] = None


class BTree:
    """Binary tree tailored for ID3 style decision trees."""

    def __init__(self) -> None:
        self.root: Optional[Node] = None
        self._counter: int = 0

    def add_root(self, value: Optional[str], zeroes: int, ones: int) -> int:
        """Create the root node and return its index."""

        if self.root is not None:
            raise ValueError("Root node already exists")
        self.root = Node(value=value, idx=self._counter, zeroes=zeroes, ones=ones)
        return self._counter

    def add_left_child(
        self, parent_idx: int, value: Optional[str], zeroes: int, ones: int
    ) -> int:
        """Attach a node as the left child of the provided parent."""

        parent = self._find_node(parent_idx)
        self._counter += 1
        parent.left = Node(value=value, idx=self._counter, zeroes=zeroes, ones=ones)
        return parent.left.idx

    def add_right_child(
        self, parent_idx: int, value: Optional[str], zeroes: int, ones: int
    ) -> int:
        """Attach a node as the right child of the provided parent."""

        parent = self._find_node(parent_idx)
        self._counter += 1
        parent.right = Node(value=value, idx=self._counter, zeroes=zeroes, ones=ones)
        return parent.right.idx

    def set_node_value(self, node_idx: int, value: str, zeroes: int, ones: int) -> None:
        """Overwrite the label stored at a node."""

        node = self._find_node(node_idx)
        node.value = value
        node.zeroes = zeroes
        node.ones = ones

    def get_root(self) -> Optional[Node]:
        """Return the root node for traversal."""

        return self.root

    def num_of_non_leaves(self, node: Optional[Node]) -> int:
        """Count the number of non-leaf nodes anchored at ``node``."""

        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 0
        return (
            1
            + self.num_of_non_leaves(node.left)
            + self.num_of_non_leaves(node.right)
        )

    def get_non_leaves(self) -> List[Node]:
        """Return a list with all non-leaf nodes in the tree."""

        nodes: List[Node] = []
        self._collect_non_leaves(self.root, nodes)
        return nodes

    def _collect_non_leaves(self, node: Optional[Node], nodes: List[Node]) -> None:
        if node is None:
            return
        if node.left is not None or node.right is not None:
            nodes.append(node)
        self._collect_non_leaves(node.left, nodes)
        self._collect_non_leaves(node.right, nodes)

    def print_btree(self) -> None:
        """Pretty print the tree using the classic ID3 indentation style."""

        if self.root is None:
            print("<empty tree>")
            return
        if self.root.left is None and self.root.right is None:
            print(self.root.value)
            return
        self._print_subtree(self.root, depth=0)

    def _print_subtree(self, node: Node, depth: int) -> None:
        indent = "| " * depth
        if node.left is not None:
            if node.left.left is None and node.left.right is None:
                print(f"{indent}{node.value} = 0 : {node.left.value}")
            else:
                print(f"{indent}{node.value} = 0 :")
                self._print_subtree(node.left, depth + 1)
        if node.right is not None:
            if node.right.left is None and node.right.right is None:
                print(f"{indent}{node.value} = 1 : {node.right.value}")
            else:
                print(f"{indent}{node.value} = 1 :")
                self._print_subtree(node.right, depth + 1)

    def traverse(self, node: Node, sample) -> int:
        """Return the predicted label for ``sample`` starting at ``node``."""

        if node.left is None and node.right is None:
            if node.value is None:
                raise ValueError("Leaf nodes must carry a classification label")
            return int(node.value)
        branch = sample[node.value]  # type: ignore[index]
        if branch == 0:
            if node.left is None:
                raise ValueError("Tree is missing a left branch for value 0")
            return self.traverse(node.left, sample)
        if node.right is None:
            raise ValueError("Tree is missing a right branch for value 1")
        return self.traverse(node.right, sample)

    def _find_node(self, node_idx: int) -> Node:
        node = self._find_node_recursive(self.root, node_idx)
        if node is None:
            raise ValueError(f"Node with index {node_idx} not found")
        return node

    def _find_node_recursive(
        self, node: Optional[Node], node_idx: int
    ) -> Optional[Node]:
        if node is None:
            return None
        if node.idx == node_idx:
            return node
        left = self._find_node_recursive(node.left, node_idx)
        if left is not None:
            return left
        return self._find_node_recursive(node.right, node_idx)