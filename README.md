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

## Quick start
- Python 3.8+ recommended. Create and activate a virtual environment.
- Install deps: `pip install -r requirements.txt`
- Set OpenAI credentials for property extraction: `export OPENAI_API_KEY=...` (model version in `src/config.py`)
- Run the full pipeline: `python3 main.py`

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
- Input data: `datasets/training/<event>/cleanedData_minutes.json` and `datasets/test/...`.
- GPT-derived properties: `datasets/training/event_properties.json`, `datasets/training/event_properties.json`.
- Fits: `results/fits/`.
- Evaluations: `results/evaluation/training_dataset/`, `results/evaluation/test_dataset/`.
- Plots: `results/plots/`.

## CLI usage
Here all configurations of FuzzyLLI are used for training and evaluation. 
If you want to use only a specific configuration comment out the others in the `main.py`. 
If you want to train/eval/plot other events change the DEFAULT_EVENTS and DEFAULT_EVENTS_NL in the main.py

- Full pipeline (train →  eval training dataset → eval test dataset → plots → demo prediction):
  ```bash
  python3 main.py
  ```

- Train only (The Event Properties (GPT-4) are determined at the beginning of the training function):
  ```bash
  python3 main.py train
  ```

- Evaluate (regenerate predictions with `--generate-new-predictions`):
  - Training dataset (via leave-one-out): `python3 main.py evaluate --scope training --generate-new-predictions`
  - Test dataset: `python3 main.py evaluate --scope test --generate-new-predictions`

- Plot (Random Forest configuration of FuzzyLLI by default):
  ```bash
  python3 main.py plot --adverbial "long time ago"
  ```

- Exemplary Prediction: The most probable interval where the event has taken place (give the adverbial) and all adverbial 
membership values are predicted for each configuration of FuzzyLLI. 
  ```bash
  python3 main.py predict \
    --event-nl "I was at the hospital" \
    --adverbial "some time ago" \
    --minutes-ago 120 \
    --properties '{"Richness":5,"Frequency":1,"Importance":5}'
  ```

## Configuration tips
- Adjust paths, model choices, and GPT model in `src/config.py`.
- Change default events/demo inputs in `main.py`.
- To limit training/eval to specific configurations, comment out the calls inside `main.py`.

## Notes
- Training may call GPT (incurs cost/latency).
