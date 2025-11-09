## ID3 Algorithm Playground

This repository offers a clean implementation of the ID3 decision tree algorithm
with two heuristics: classic information gain (entropy) and variance impurity.
It also demonstrates the randomized post-pruning strategy often used in
assignments to avoid overfitting. The codebase has been reorganised into a
Python package so that it can be reused from scripts or invoked directly from
the command line.

📚 **New to ID3?** Check out [ALGORITHM_GUIDE.md](./ALGORITHM_GUIDE.md) for a comprehensive visual guide with step-by-step examples and diagrams!

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

### Visual explanation

#### How ID3 builds a decision tree

```
Step 1: Start with training data
┌─────────────────────────────────┐
│  Attr_A  Attr_B  Attr_C  Class  │
│    0       1       0       1    │
│    1       0       1       0    │
│    0       1       1       1    │
│    1       1       0       1    │
│    ...    ...     ...     ...   │
└─────────────────────────────────┘
         ↓
Calculate entropy/variance for each attribute
         ↓
Step 2: Select best attribute (highest information gain)
         ↓
    ┌────────┐
    │ Attr_A │  ← Root node (best split)
    └───┬────┘
        │
   ┌────┴────┐
   ↓         ↓
 A=0       A=1
```

#### Recursive splitting

```
         Root
          │
    ┌─────┴─────┐
    │           │
  A = 0       A = 1
    │           │
    ↓           ↓
Split on    Split on
Attr_B      Attr_C
    │           │
┌───┴───┐   ┌───┴───┐
↓       ↓   ↓       ↓
B=0    B=1  C=0    C=1
│       │    │      │
0       1    1      0  ← Leaf nodes (predictions)
```

#### Example decision tree output

```
Weather = Sunny :
| Humidity = High : No
| Humidity = Low : Yes
Weather = Overcast : Yes
Weather = Rainy :
| Wind = Strong : No
| Wind = Weak : Yes
```

This tree structure means:
- If weather is sunny AND humidity is high → predict No
- If weather is sunny AND humidity is low → predict Yes
- If weather is overcast (regardless of other factors) → predict Yes
- If weather is rainy AND wind is strong → predict No
- If weather is rainy AND wind is weak → predict Yes

#### Post-pruning visualization

```
Before Pruning:              After Pruning:
    Root                         Root
     │                            │
  ┌──┴──┐                     ┌───┴───┐
  A     B                     A       1 ← Subtree replaced
  │     │                     │         with leaf
 ┌┴┐   ┌┴┐                   ┌┴┐
 C D   E F                   C D
 │ │   │ │                   │ │
 0 1   1 0                   0 1

The pruned tree is simpler and may generalize better!
```

#### Entropy vs Variance comparison

```
Dataset: [++++++----]  (6 positive, 4 negative)

Entropy approach:
  H(S) = -0.6·log₂(0.6) - 0.4·log₂(0.4)
       ≈ 0.971 bits

Variance approach:
  V(S) = 0.6 × 0.4
       = 0.24

Both measure impurity, but entropy is theoretically
more principled while variance is computationally simpler.

Pure sets have 0 impurity:
  [++++++++++] → H=0, V=0
  [----------] → H=0, V=0

Maximum impurity at 50/50 split:
  [+++++-----] → H=1.0, V=0.25
```

#### Information gain illustrated

```
Parent node: [+++++-----]  (H = 0.971)
                │
        Split on Attribute X
                │
       ┌────────┴────────┐
       ↓                 ↓
    X = 0             X = 1
  [++++--]          [+---]
  H = 0.918        H = 0.811
  (6 samples)      (4 samples)

Weighted average = (6/10)·0.918 + (4/10)·0.811 = 0.875

Information Gain = 0.971 - 0.875 = 0.096

Higher gain = better split!
```### Installation

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
