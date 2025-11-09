## ID3 Algorithm Playground

This repository offers a clean implementation of the ID3 decision tree algorithm
with two heuristics: classic information gain (entropy) and variance impurity.
It also demonstrates the randomized post-pruning strategy often used in
assignments to avoid overfitting. The codebase has been reorganised into a
Python package so that it can be reused from scripts or invoked directly from
the command line.

### Project layout

```
src/
	id3_algorithm/
		cli.py          # Command-line entry point (python -m id3_algorithm.cli)
		entropy.py      # Entropy-based training routine
		variance.py     # Variance impurity training routine
		tree.py         # General binary tree helpers
reports/
	...              # Sample accuracy reports generated during experimentation
```

### Algorithm overview

The learner expects binary attributes and a binary target column named
`Class`. For a dataset $S$ and an attribute $A$ the entropy-based gain is

$$
Gain(S, A) = Entropy(S) - \sum_{v \in \{0, 1\}} \frac{|S_v|}{|S|} \cdot Entropy(S_v)
$$

with $Entropy(S) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)$ and $p_1$, $p_0$ the
class proportions. The variance impurity variant replaces entropy with

$$
Variance(S) = p_1 \cdot p_0.
$$

Trees are grown recursively until either the examples are pure, no attributes
remain, or a branch becomes empty, in which case the majority label of the
parent node is used. Post-pruning repeats the standard randomized procedure:

1. Clone the tree produced on the training data.
2. Repeat $L$ times:
	 - Copy the current tree and randomly replace up to $M \sim \mathcal{U}(1, K)$
		 non-leaf nodes with leaves labelled by the majority class at that node.
	 - Keep the candidate if it improves accuracy on the validation split.

### Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Running experiments

The CLI mirrors the original script but now supports selecting which heuristic
to run. The positional arguments are:

- `L` and `K`: integers controlling the number of rounds and candidate prunes.
- `training_set`, `validation_set`, `test_set`: CSV files with binary features
	and a `Class` column.
- `print_tree`: whether to print the tree before and after pruning (`yes`/`no`).

Example:

```
python -m id3_algorithm.cli 55 5 data_sets1/training_set.csv \
		data_sets1/validation_set.csv data_sets1/test_set.csv no
```

To run only the variance impurity heuristic:

```
python -m id3_algorithm.cli 100 6 data_sets2/training_set.csv \
		data_sets2/validation_set.csv data_sets2/test_set.csv yes --metric variance
```

Both heuristics print their accuracy on the test set before and after pruning.
If `print_tree=yes`, the unpruned tree and the best pruned tree are displayed.

### Reports

Historical experiment logs are preserved under `reports/` for reference. They
match the output format produced by the new CLI for reproducibility.

### Next steps

- Extend the tree printer to export Graphviz `.dot` files for visualisation.
- Add automated tests that validate the pruning routine using synthetic data.
