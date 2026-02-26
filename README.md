# FuzzyLLI – A fuzzy probabilistic model of vague temporal adverbials

This repository accompanies the paper **“A Fuzzy Probabilistic Model of Human Interpretation of Vague Temporal Adverbials”**, 
which investigates how people interpret expressions such as *just*, *recently*, *some time ago*, and *a long time ago* in 
relation to different types of events.

The paper introduces **FuzzyLLI**, a compositional fuzzy–probabilistic framework that combines fuzzy semantics for vague temporal adverbials 
with probabilistic models of event-dependent temporal uncertainty grounded in cognitive memory theory. 
By integrating intrinsic event properties such as richness, frequency, and importance, 
the model captures how perceived temporal distance depends not only on elapsed time but also on how events are experienced and remembered.

Check out the paper [here (Paper currently under review)](PATH_OR_URL_TO_PAPER) if you are interested in the details.

This repository provides code, data, and evaluation resources to reproduce the experiments, analyze the model’s behavior, making predictions 
and extend the approach to new events or temporal expressions.


## Quick start (uv)

- Python 3.10 recommended
- Install [`uv`](https://github.com/astral-sh/uv)
- Create/sync the environment from `pyproject.toml` / `uv.lock`:
  ```bash
  uv sync
- Set OpenAI credentials for property extraction: `export OPENAI_API_KEY=...` (model version in `src/config.py`)
- Run the full pipeline: `uv run python -m fuzzylli.main`

## What you can do
- Train all FuzzyLLI variants and get the event property predictions via gpt
- Evaluate on the training (via leave-one-out) and test dataset
- Plot fitted event/adverbial probabilities/membership functions.
- Make predictions for:
  - event + happening time -> membership values of the vague adverbials
  - adverbial + event type -> time interval where the event most probable has taken place


## Repository layout
- `main.py` — CLI entrypoint (full pipeline + subcommands).
- `src/train.py` — GPT property extraction + training (core FuzzyLLI and the 3 configurations).
- `src/predictions.py` — Inference.
- `src/evaluation_test_dataset.py`, `src/evaluation_training_dataset.py` — Evaluation of training (leave-one-out) and test dataset.
- `src/plot.py` — Visualisations.
- `src/config.py` — paths, model settings, and function choices.

## Data + outputs
- Input data: `fuzzylli/datasets/training/<event>/cleanedData_minutes.json` and `fuzzylli/datasets/test/...`.
- GPT-derived properties: `fuzzylli/datasets/training/event_properties.json`, `fuzzylli/datasets/test/event_properties.json`.
- Fits: `results/fits/`.
- Evaluations: `results/evaluation/training_dataset/`, `results/evaluation/test_dataset/`.
- Plots: `results/plots/`.

## CLI usage
Here all configurations of FuzzyLLI are used for training and evaluation. 
If you want to use only a specific configuration comment out the others in the `main.py`. 
If you want to train/eval/plot other events change the DEFAULT_EVENTS and DEFAULT_EVENTS_NL in the `fuzzylli/main.py`

- Full pipeline (train →  eval training dataset → eval test dataset → plots → demo prediction):
  ```bash
  uv run python -m fuzzylli.main
  ```

- Train only (The Event Properties (GPT-4) are not determined new to avoid cost):
  ```bash
  uv run python -m fuzzylli.main train
  ```

- Evaluate (regenerate predictions with `--generate-new-predictions`):
  - Training dataset (via leave-one-out): `uv run python -m fuzzylli.main evaluate --scope training --generate-new-predictions`
  - Test dataset: `uv run python -m fuzzylli.main evaluate --scope test --generate-new-predictions`

- Plot (Random Forest configuration of FuzzyLLI by default):
  ```bash
  uv run python -m fuzzylli.main plot --adverbial "long time ago"
  ```

- Exemplary Prediction: The most probable interval where the event has taken place (give the adverbial) and all adverbial 
membership values are predicted for each configuration of FuzzyLLI. 
  ```bash
  uv run python -m fuzzylli.main predict \
    --event-nl "I was at the hospital" \
    --adverbial "some time ago" \
    --minutes-ago 120 \
    --properties '{"Richness":5,"Frequency":1,"Importance":5}'
  ```

## Configuration tips
- Adjust paths, model choices, and GPT model in `fuzzylli/src/config.py`.
- Change default events/demo inputs in `fuzzylli/main.py`.
- To limit training/eval to specific configurations, comment out the calls inside `fuzzylli/main.py`.

## Notes
- Training may call GPT (incurs cost/latency).
