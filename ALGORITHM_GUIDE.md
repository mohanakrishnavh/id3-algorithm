# ID3 Algorithm: Visual Guide

## Table of Contents
1. [Decision Tree Basics](#decision-tree-basics)
2. [ID3 Algorithm Steps](#id3-algorithm-steps)
3. [Entropy Calculation](#entropy-calculation)
4. [Variance Impurity](#variance-impurity)
5. [Building the Tree](#building-the-tree)
6. [Post-Pruning Strategy](#post-pruning-strategy)
7. [Complete Example](#complete-example)

## Decision Tree Basics

A decision tree is a flowchart-like structure where:
- **Internal nodes** represent tests on attributes
- **Branches** represent outcomes of tests
- **Leaf nodes** represent class labels

```
         ┌─────────┐
         │  Root   │  ← Decision node
         │(Outlook)│
         └────┬────┘
              │
     ┌────────┼────────┐
     │        │        │
  Sunny    Overcast  Rainy
     │        │        │
     ↓        ↓        ↓
 ┌───────┐   Yes   ┌───────┐
 │Humid? │         │ Wind? │
 └───┬───┘         └───┬───┘
     │                 │
  ┌──┴──┐          ┌───┴───┐
High  Low       Strong   Weak
  │     │          │       │
  ↓     ↓          ↓       ↓
 No    Yes        No      Yes
```

## ID3 Algorithm Steps

```
┌─────────────────────────────────────────────────────────┐
│                    ID3 Algorithm                        │
└─────────────────────────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────┐
        │  Base Cases (Stopping Criteria) │
        └─────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
   All same         No attributes     Empty subset
   class?           remaining?         (edge case)
      │                  │                  │
      ↓                  ↓                  ↓
  Return leaf      Return majority    Return parent
  with class       class leaf         majority
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          ↓
                    Not base case?
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │  Calculate gain for each attribute  │
        └─────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │   Select attribute with best gain   │
        └─────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │  Create node with that attribute    │
        └─────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │ For each value of the attribute:    │
        │   - Partition data by value         │
        │   - Recursively build subtree       │
        └─────────────────────────────────────┘
```

## Entropy Calculation

Entropy measures the impurity or disorder in a dataset.

### Formula
```
H(S) = -Σ p(c) × log₂(p(c))

where p(c) = proportion of class c in dataset S
```

### Visual Examples

#### Example 1: Pure dataset
```
Dataset: [+, +, +, +, +]  (all positive)

p(+) = 5/5 = 1.0
p(-) = 0/5 = 0.0

H(S) = -(1.0 × log₂(1.0)) - (0.0 × log₂(0.0))
     = -(1.0 × 0) - 0
     = 0

┌─────────────────────┐
│ Entropy = 0         │  ← Perfectly pure!
│ No uncertainty      │
└─────────────────────┘
```

#### Example 2: Evenly split dataset
```
Dataset: [+, +, +, -, -, -]  (3 positive, 3 negative)

p(+) = 3/6 = 0.5
p(-) = 3/6 = 0.5

H(S) = -(0.5 × log₂(0.5)) - (0.5 × log₂(0.5))
     = -(0.5 × -1) - (0.5 × -1)
     = 0.5 + 0.5
     = 1.0

┌─────────────────────┐
│ Entropy = 1.0       │  ← Maximum uncertainty!
│ Can't predict       │
└─────────────────────┘
```

#### Example 3: Mixed dataset
```
Dataset: [+, +, +, +, +, +, -, -]  (6 positive, 2 negative)

p(+) = 6/8 = 0.75
p(-) = 2/8 = 0.25

H(S) = -(0.75 × log₂(0.75)) - (0.25 × log₂(0.25))
     = -(0.75 × -0.415) - (0.25 × -2)
     ≈ 0.311 + 0.500
     ≈ 0.811

┌─────────────────────┐
│ Entropy ≈ 0.811     │  ← Some uncertainty
│ Mostly positive     │
└─────────────────────┘
```

### Entropy Scale
```
0.0 ─────────────────────────────────── 1.0
 │                                        │
Pure                                 Maximum
dataset                            uncertainty
 │                                        │
 ↓                                        ↓
[+++++++]                         [+++---]
[-------]                         50/50 split

Lower entropy = Better for prediction!
```

## Variance Impurity

An alternative to entropy that's simpler to compute.

### Formula
```
V(S) = p(+) × p(-)

For binary classification with classes 0 and 1
```

### Comparison Chart
```
Class Distribution    Entropy    Variance
─────────────────    ───────    ────────
100% positive        0.000      0.000    ← Pure
90% positive         0.469      0.090
80% positive         0.722      0.160
70% positive         0.881      0.210
60% positive         0.971      0.240
50% positive         1.000      0.250    ← Max impurity
40% positive         0.971      0.240
30% positive         0.881      0.210
20% positive         0.722      0.160
10% positive         0.469      0.090
0% positive          0.000      0.000    ← Pure

Both reach minimum at pure nodes
Both reach maximum near 50/50 split
```

### Visual Comparison
```
Entropy curve:           Variance curve:
    1.0│      ╱╲              0.25│      ╱╲
       │     ╱  ╲                 │     ╱  ╲
    0.5│    ╱    ╲             0.15│    ╱    ╲
       │   ╱      ╲                │   ╱      ╲
    0.0│__╱________╲__          0.0│__╱________╲__
       0%   50%   100%             0%   50%   100%
```

## Building the Tree

### Step-by-step example

#### Dataset
```
┌─────┬─────┬─────┬───────┐
│  A  │  B  │  C  │ Class │
├─────┼─────┼─────┼───────┤
│  0  │  0  │  1  │   1   │
│  0  │  1  │  0  │   0   │
│  0  │  1  │  1  │   1   │
│  1  │  0  │  0  │   0   │
│  1  │  0  │  1  │   1   │
│  1  │  1  │  0  │   0   │
└─────┴─────┴─────┴───────┘

Overall: 3 positive, 3 negative
Entropy(S) = 1.0 (maximum uncertainty)
```

#### Step 1: Calculate gain for each attribute

**Attribute A:**
```
A = 0: [1, 0, 1]  → 2 pos, 1 neg, H = 0.918
A = 1: [0, 1, 0]  → 1 pos, 2 neg, H = 0.918

Weighted avg = (3/6)×0.918 + (3/6)×0.918 = 0.918
Gain(A) = 1.0 - 0.918 = 0.082
```

**Attribute B:**
```
B = 0: [1, 0, 1]  → 2 pos, 1 neg, H = 0.918
B = 1: [0, 1, 0]  → 1 pos, 2 neg, H = 0.918

Gain(B) = 1.0 - 0.918 = 0.082
```

**Attribute C:**
```
C = 0: [0, 0, 0]  → 0 pos, 3 neg, H = 0.0
C = 1: [1, 1, 1]  → 3 pos, 0 neg, H = 0.0

Weighted avg = (3/6)×0.0 + (3/6)×0.0 = 0.0
Gain(C) = 1.0 - 0.0 = 1.0  ← Best!
```

#### Step 2: Split on C
```
         C
         │
    ┌────┴────┐
    │         │
  C=0       C=1
    │         │
 [0,0,0]   [1,1,1]
    │         │
    ↓         ↓
    0         1

Done! Both branches are pure.
```

### Information Gain Visualization
```
Parent Node
[+++++-----]  H = 0.971
     │
     │ Try splitting on different attributes
     │
     ├─── Split A ───┐
     │               │
  Gain=0.2        Gain=0.4  ← Better!
     │               │
     ├─── Split B ───┤
                     │
                 Pick Split B
                     │
                     ↓
              ┌──────────┐
              │ Node: B  │
              └──────────┘
                     │
            ┌────────┴────────┐
            │                 │
         B = 0             B = 1
      [++++--]          [+---]
```

## Post-Pruning Strategy

### Why prune?
```
Overfitted tree:              Pruned tree:
      Root                        Root
       │                           │
    ┌──┴──┐                     ┌──┴──┐
    A     B                     A     1
    │     │                     │
   ┌┴┐   ┌┴┐                   ┌┴┐
   C D   E F                   C D
   │ │   │ │                   │ │
  ┌┴┐│  ┌┴┐│                  0 1
  0 1│  0 1│
     │     │
   ┌─┴┐  ┌┴─┐
   0  1  0  1

Training: 100%              Training: 95%
Test: 75%                   Test: 85%  ← Better!
```

### Randomized post-pruning algorithm
```
┌────────────────────────────────────────────┐
│  Input: Tree T, parameters L and K         │
│  Output: Best pruned tree                  │
└────────────────────────────────────────────┘
                    │
                    ↓
        D_best ← copy of T
                    │
                    ↓
        ┌───────────────────────┐
        │   Repeat L times:     │
        └───────────────────────┘
                    │
                    ↓
        D' ← copy of T
                    │
                    ↓
        M ← random(1, K)
                    │
                    ↓
        ┌───────────────────────┐
        │   Repeat M times:     │
        │                       │
        │ - Pick random         │
        │   non-leaf node N     │
        │                       │
        │ - Replace N with      │
        │   leaf labeled by     │
        │   majority class      │
        └───────────────────────┘
                    │
                    ↓
        ┌───────────────────────┐
        │ If accuracy(D') >     │
        │    accuracy(D_best)   │
        │ then D_best ← D'      │
        └───────────────────────┘
                    │
                    ↓
        Return D_best
```

### Pruning example iteration
```
Iteration 1:
  Original → Prune 2 nodes → Accuracy: 82%
                                   ↓
                              Keep as best

Iteration 2:
  Original → Prune 1 node → Accuracy: 80%
                                   ↓
                            Discard (worse)

Iteration 3:
  Original → Prune 3 nodes → Accuracy: 85%
                                   ↓
                            New best! ✓

...continue for L iterations
```

## Complete Example

### Training data
```
┌─────────┬─────────┬────────┬───────┐
│ Outlook │ Temp    │ Humid  │ Play? │
├─────────┼─────────┼────────┼───────┤
│ Sunny   │ Hot     │ High   │  No   │
│ Sunny   │ Hot     │ High   │  No   │
│ Overcast│ Hot     │ High   │  Yes  │
│ Rain    │ Mild    │ High   │  Yes  │
│ Rain    │ Cool    │ Normal │  Yes  │
│ Rain    │ Cool    │ Normal │  No   │
│ Overcast│ Cool    │ Normal │  Yes  │
│ Sunny   │ Mild    │ High   │  No   │
│ Sunny   │ Cool    │ Normal │  Yes  │
│ Rain    │ Mild    │ Normal │  Yes  │
│ Sunny   │ Mild    │ Normal │  Yes  │
│ Overcast│ Mild    │ High   │  Yes  │
│ Overcast│ Hot     │ Normal │  Yes  │
│ Rain    │ Mild    │ High   │  No   │
└─────────┴─────────┴────────┴───────┘

Total: 9 Yes, 5 No
Entropy = 0.940
```

### Tree construction process
```
Step 1: Root node
────────────────
Calculate gain for all attributes:
  Gain(Outlook) = 0.246  ← Highest!
  Gain(Temp) = 0.029
  Gain(Humid) = 0.151

Select Outlook as root:

         Outlook
            │
    ┌───────┼───────┐
    │       │       │
  Sunny  Overcast Rain
    │       │       │
   [2,3]  [4,0]   [3,2]

Step 2: Overcast branch
────────────────────────
All 4 examples are "Yes"
→ Create leaf with "Yes"

         Outlook
            │
    ┌───────┼───────┐
    │       │       │
  Sunny  Overcast Rain
    │       │       │
   [2,3]   Yes    [3,2]

Step 3: Sunny branch
─────────────────────
Calculate gain for remaining attributes:
  Gain(Temp) = 0.571
  Gain(Humid) = 0.971  ← Highest!

Split on Humid:

         Outlook
            │
    ┌───────┼───────┐
    │       │       │
  Sunny  Overcast Rain
    │       │       │
  Humid    Yes    [3,2]
    │
  ┌─┴─┐
High Normal
  │     │
 [0,3] [2,0]
  │     │
  No   Yes

Step 4: Rain branch
────────────────────
Similar process...

Final tree:

         Outlook
            │
    ┌───────┼───────┐
    │       │       │
  Sunny  Overcast  Rain
    │       │       │
  Humid    Yes    Wind
    │              │
  ┌─┴─┐          ┌─┴─┐
High Norm      Strong Weak
  │    │         │     │
  No  Yes        No   Yes
```

### Tree output format
```
Outlook = Sunny :
| Humidity = High : No
| Humidity = Normal : Yes
Outlook = Overcast : Yes
Outlook = Rain :
| Wind = Strong : No
| Wind = Weak : Yes
```

### Making predictions
```
Test sample: [Outlook=Sunny, Temp=Cool, Humid=High, Wind=Weak]

1. Start at root: Outlook?
   → Outlook = Sunny, go left

2. At Humidity node: Humidity?
   → Humidity = High, go left

3. At leaf: No

Prediction: Don't play (No)
```

---

## Key Takeaways

1. **ID3 is a greedy algorithm**: Makes locally optimal choices at each step
2. **Entropy measures uncertainty**: Lower is better for classification
3. **Information gain guides splits**: Pick attribute that reduces entropy most
4. **Pruning prevents overfitting**: Simpler trees often generalize better
5. **Both heuristics work**: Entropy is theoretically better, variance is simpler

## Further Reading

- Original ID3 paper: Quinlan, J. R. (1986)
- C4.5 algorithm (ID3 successor)
- Random Forests (ensemble of decision trees)
- CART (Classification and Regression Trees)
