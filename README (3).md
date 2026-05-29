# Match Outcome Prediction — Logistic Regression

A from-scratch logistic regression model for predicting football match results using historical match data.

## Overview

This notebook implements a custom logistic regression classifier (no scikit-learn) to predict whether a team wins a given match. It trains on pre-2022 match data and evaluates on matches from 2022 onward.

## Dataset

**File:** `TIPE2/matches.csv`

Expected columns include:

| Column | Description |
|---|---|
| `date` | Match date (used for train/test split) |
| `venue_code` | Encoded home/away venue |
| `opp_code` | Encoded opponent identifier |
| `hour` | Kickoff hour |
| `day_code` | Day of the week (encoded) |
| `target` | Binary label — `1` = win, `0` = not win |
| `team` | Team name |
| `opponent` | Opponent name |
| `result` | Raw result string |

## Model

A logistic regression classifier built from scratch using NumPy:

- **Activation:** Sigmoid function
- **Optimization:** Gradient descent
- **Hyperparameters:** Learning rate `lr=0.01`, `n_iters=1000`
- **Weights:** Initialized to zero

## Features (Predictors)

```
venue_code, opp_code, hour, day_code
```

## Train / Test Split

| Split | Condition |
|---|---|
| Train | `date < 2022-01-01` |
| Test | `date >= 2022-01-01` |

## Usage

1. Place `matches.csv` in a `TIPE2/` directory relative to the notebook.
2. Run all cells in order.
3. The final cell outputs a DataFrame of actual vs. predicted results merged with match metadata.

```python
# Example output columns
# actual | predicted | date | team | opponent | result
```

## Requirements

```
pandas
numpy
matplotlib
```

Install with:

```bash
pip install pandas numpy matplotlib
```

## Output

- **Accuracy score** printed to console after evaluation
- **`combined_predictions`** DataFrame with columns: `actual`, `predicted`, `date`, `team`, `opponent`, `result`
