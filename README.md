# FuzzyLLI – A fuzzy probabilistic model of vague temporal adverbials

This repository accompanies the paper **“A Fuzzy Probabilistic Model of Human Interpretation of Vague Temporal Adverbials”**, 
which investigates how people interpret expressions such as *just*, *recently*, *some time ago*, and *a long time ago* in 
relation to different types of events.

The paper introduces **FuzzyLLI**, a compositional fuzzy–probabilistic framework that combines fuzzy semantics for vague temporal adverbials with probabilistic models of event-dependent temporal uncertainty grounded in cognitive memory theory. 

In this branch FuzzyLLI is not used for vague temporal adverbials but for vague spatial expressions *close to*, *moderately far*, *far away*.
For this the training dataset has changed including now spatial distances, two object types *Small* and *Big* and membership values for the three expressions.

This repository provides: 

- an installable Python package (`fuzzylli`) for predictions (distance or adverbial)
- research scripts to reproduce training and plots

---

## Requirements

- **Python 3.10**
- Recommended: [`uv`](https://github.com/astral-sh/uv)

---

## Installation

### 1) Install the prediction package (minimal runtime)
This installs the `fuzzylli` package from `src/fuzzylli/`.

```bash
uv venv
uv pip install -e . 
uv pip install . # for non-editable install
```

### 2) Install with research dependencies (training/eval/plots)
To run the full pipeline you need the `dev` extras
```bash
uv pip install -e ".[dev]"
```
---

## Quick Start: Package Usage (Predictions with Random Forest)
The package loads fitted artifacts bundled under `src/fuzzylli/resources/` (no training required).
```python
from fuzzylli import (
    predict_distance_fuzzylli,
    predict_adverbial_fuzzylli,
)

# Memberships for each vague adverbial given the distance
adverbial_memberships = predict_adverbial_fuzzylli("Small", 20)
print(adverbial_memberships)

# Interval (distance) where this object most likely would be described as close to
upper, lower = predict_distance_fuzzylli("Small", "close to")
print(upper, lower)
```

## Research Pipeline (train, plot)
It is driven by the repo-root `main.py` and uses code under `scripts/`
```bash
# Full pipeline (default behaviour)
uv run python main.py

# Train only
uv run python main.py train

# Plot (Random Forest configuration by default)
uv run python main.py plot --adverbial "close to"

# Example prediction demo
uv run python main.py predict \
  --object "Big" \
  --adverbial "close to" \
  --distance 20
```
---

## Repository layout
- `src/fuzzylli` — installable package.
- `scripts/` — training/plotting.
- `main.py` — CLI entrypoint (full pipeline + subcommands).
- `datasets/` — training dataset
- `results/` — generated fits/plots

---

## Notes
- If you retrain models and want the installed package to use the new RF/Json artifacts copy the udpated files into `src/fuzzylli/resources` and reinstall:
```bash
uv pip install -e .
```
