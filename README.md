# FuzzyLLI – A fuzzy probabilistic model of vague temporal adverbials

This repository accompanies the paper **“A Fuzzy Probabilistic Model of Human Interpretation of Vague Temporal Adverbials”**, 
which investigates how people interpret expressions such as *just*, *recently*, *some time ago*, and *a long time ago* in 
relation to different types of events.

The paper introduces **FuzzyLLI**, a compositional fuzzy–probabilistic framework that combines fuzzy semantics for vague temporal adverbials 
with probabilistic models of event-dependent temporal uncertainty grounded in cognitive memory theory. 
By integrating intrinsic event properties such as richness, frequency, and importance, 
the model captures how perceived temporal distance depends not only on elapsed time but also on how events are experienced and remembered.

Check out the paper [here (Paper currently under review)](PATH_OR_URL_TO_PAPER) if you are interested in the details.

This repository provides code, data, and evaluation resources to reproduce the experiments, analyze the model’s behavior, 
and extend the approach to new events or temporal expressions.

## Repository layout
- `main.py` — CLI entrypoint (full pipeline + subcommands).
- `src/train.py` — GPT property extraction + training (core FuzzyLLI and the 3 configurations).
- `src/predictions.py` — Inference.
- `src/evaluation_test_dataset.py`, `src/evaluation_training_dataset.py` — Evaluation of training (leave-one-out) and test dataset.
- `src/plot.py` — Visualisations.
- `src/config.py` — paths, model settings, and function choices.

## Setup
1) Python 3.9+ recommended. Create and activate a virtual environment.
2) Install dependencies: `pip install -r requirements.txt`.
3) Set OpenAI credentials for property extraction (e.g., `export OPENAI_API_KEY=...`). GPT versions are configurable in `src/config.py`.

## Data expectations
- Training data under `dataset/training`.
- Test data under `datasets/test`.
- Model fitting parameters and evaluation metrics are written to `results/...` (see constants in `src/config.py`).

## Usage (CLI)
Here all configurations of FuzzyLLI are used for training and evaluation. 
If you want to use only a specific configuration comment out the others in the main.py
If you want to train/eval/plot other events change the DEFAULT_EVENTS and DEFAULT_EVENTS_NL in the main.py
- Full pipeline (train → eval training dataset → eval test dataset → plots → demo prediction):
  ```
  python3 main.py
  ```
- Train only (The Event Properties (GPT-4) are determined at the beginning of the training function):
  ```
  python3 main.py train
  ```
- Evaluate:
  - Training dataset (via leave-one-out): `python3 main.py evaluate --scope training --generate-new-predictions`
  - Test dataset: `python3 main.py evaluate --scope test --generate-new-predictions`
- Plot fitted functions:
  The used configuration for plotting is Random Forest. 
Change the configuration when calling `plot_results(...)` in the main.py. 
  ```
  python3 main.py plot --adverbial "long time ago"
  ```
- Exemplary Prediction: 
The most probable interval where the event has taken place (give the adverbial) and all adverbial membership values are 
predicted for each configuration of FuzzyLLI.
  ```
  python3 main.py predict \
    --event-nl "I was at the hospital" \
    --adverbial "some time ago" \
    --minutes-ago 120 \
    --properties '{"Richness":5,"Frequency":1,"Importance":5}'
  ```

## Configuration
- Edit defaults, paths, and function choices in `src/config.py`.
- Default events and the demo prediction live in `main.py`; the CLI uses these when you omit arguments.

## Outputs
- Fits: `results/fits/`.
- Evaluation metrics/predictions: `results/evaluation/`.
- Plots: `results/plots/`.
- GPT predicted properties: `data/with_event_properties/event_properties.json` and `data/evaluation_survey/event_properties.json`.

## Notes
- Training and evaluation will generate/overwrite files under `results/` and may call GPT (cost + latency).
- Some functionality depends on scikit-learn/xgboost; ensure system libraries meet their requirements.
