# FuzzyLLI – A fuzzy probabilistic model of vague temporal adverbials

This repository accompanies the paper **“A Fuzzy Probabilistic Model of Human Interpretation of Vague Temporal Adverbials”**, 
which investigates how people interpret expressions such as *just*, *recently*, *some time ago*, and *a long time ago* in 
relation to different types of events.

The paper introduces **FuzzyLLI**, a compositional fuzzy–probabilistic framework that combines fuzzy semantics for vague temporal adverbials with probabilistic models of event-dependent temporal uncertainty grounded in cognitive memory theory. 
By integrating intrinsic event properties such as richness, frequency, and importance, 
the model captures how perceived temporal distance depends not only on elapsed time but also on how events are experienced and remembered.

Check out the paper [here (Paper currently under review)](PATH_OR_URL_TO_PAPER) if you are interested in the details.

This repository provides: 

- an installable Python package (`fuzzylli`) for random-forest based predictions (time interval or adverbial)
- research scripts to reproduce training, evaluation, and plots

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
    predict_time_frame_random_forest,
    predict_adverbial_random_forest,
)

event_properties = {"Richness": 5, "Frequency": 1, "Importance": 5}

# Memberships for each vague adverbial given minutes_ago
adverbial_memberships = predict_adverbial_random_forest(event_properties, minutes_ago=120)
print(adverbial_memberships)

# Interval (minutes ago) where an event with these properties most likely occurred
upper, lower = predict_time_frame_random_forest(event_properties, adverbial="just", min_prob=0.6)
print(upper, lower)
```

## Research Pipeline (train, eval, plot)
It is driven by the repo-root `main.py` and uses code under `scripts/`
```bash
# Full pipeline (default behaviour)
uv run python main.py

# Train only
uv run python main.py train

# Evaluate (regenerate predictions with --generate-new-predictions)
uv run python main.py evaluate --scope training --generate-new-predictions
uv run python main.py evaluate --scope test --generate-new-predictions

# Plot (Random Forest configuration by default)
uv run python main.py plot --adverbial "long time ago"

# Example prediction demo
uv run python main.py predict \
  --event-nl "I was at the hospital" \
  --adverbial "some time ago" \
  --minutes-ago 120 \
  --properties '{"Richness":5,"Frequency":1,"Importance":5}'
```
---

## Repository layout
- `src/fuzzylli` — installable package.
- `scripts/` — training/evaluation/plotting.
- `main.py` — CLI entrypoint (full pipeline + subcommands).
- `datasets/` — training and test datasets
- `results/` — generated fits/evaluation/plots

---

## Notes
- Training may call GPT (incurs cost/latency).
- If you retrain models and want the installed package to use the new RF/Json artifacts copy the udpated files into `src/fuzzylli/resources` and reinstall:
```bash
uv pip install -e .
```
