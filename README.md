# FuzzyLLI – A fuzzy probabilistic model of vague temporal adverbials

FuzzyLLI blends ML and cognitive modeling to predict temporal adverbials (e.g., *just*, *recently*, *long time ago*) and event recency. It trains event-specific embeddings, random forests, power/exponential functions, and baseline classifiers/regressors, and it uses GPT to derive event properties (Richness, Frequency, Importance).

## Highlights
- Predict adverbials for events and minutes-ago from adverbials.
- Train/evaluate on seen events and unseen survey-style events; optional property ablations.
- Plot fitted event/adverbial membership functions.
- GPT-based extraction of event properties for downstream models.

## Repository layout
- `main.py` — CLI entrypoint (full pipeline + subcommands).
- `src/train.py` — training routines.
- `src/predictions.py` — inference + GPT property extraction.
- `src/evaluation.py`, `src/evaluation_survey.py` — metrics and evaluation drivers.
- `src/plot.py` — visualisations.
- `src/config.py` — paths, model settings, and function choices.

## Setup
1) Python 3.9+ recommended. Create and activate a virtual environment.
2) Install dependencies: `pip install -r requirements.txt`.
3) Set OpenAI credentials for property extraction (e.g., `export OPENAI_API_KEY=...`). GPT versions are configurable in `src/config.py`.

## Data expectations
- Experimental event data under `data/with_event_properties/<event>/cleanedData_minutes.json`.
- Survey/unseen data under `data/evaluation_survey`.
- Model artifacts and metrics are written to `results/...` (see constants in `src/config.py`).

## Usage (CLI)
- Full pipeline (train → eval seen → eval survey → property ablations → plots → demo prediction):
  ```
  python3 main.py
  ```
- Train only:
  ```
  python3 main.py train
  ```
- Evaluate:
  - Seen events: `python3 main.py evaluate --scope seen --generate-new-predictions`
  - Survey/unseen: `python3 main.py evaluate --scope survey --generate-new-predictions`
  - Property ablations: `python3 main.py evaluate --scope properties`
- Plot fitted functions:
  ```
  python3 main.py plot --adverbial "long time ago"
  ```
- Predict for a custom event (properties can be a JSON object or a list of objects):
  ```
  python3 main.py predict \
    --event-nl "I was at the hospital" \
    --adverbial "just" \
    --minutes-ago 120 \
    --properties '{"Richness":5,"Frequency":1,"Importance":5}'
  ```

## Configuration
- Edit defaults, paths, and function choices in `src/config.py`.
- Default events and the demo prediction live in `main.py`; the CLI uses these when you omit arguments.

## Outputs
- Fits: `results/fits/` (plus `results/fits/simple_models/`).
- Evaluation metrics/predictions: `results/evaluation/seen_events/` and `results/evaluation/unseen_events/`.
- Plots: `results/plots/` (e.g., `highestStd_allAdverbials.png`).
- GPT prompts/properties: `data/with_event_properties/event_properties.json` and `data/evaluation_survey/event_properties.json`.

## Notes
- Training and evaluation will generate/overwrite files under `results/` and may call GPT (cost + latency).
- Some functionality depends on scikit-learn/xgboost; ensure system libraries meet their requirements.
