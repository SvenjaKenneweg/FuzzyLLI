# FuzzyLLI – A fuzzy probabilistic model of vague temporal adverbials

FuzzyLLI blends ML and cognitive modeling to predict temporal adverbials (e.g., *just*, *recently*, *long time ago*) and event recency. It trains event-specific embeddings, random forests, power/exponential functions, and baseline classifiers/regressors, and it uses GPT to derive event properties (Richness, Frequency, Importance).

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
- Training data under `data/with_event_properties`.
- Test data under `data/evaluation_survey`.
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
  - Training dataset (via leave-one-out): `python3 main.py evaluate --scope seen --generate-new-predictions`
  - Test dataset: `python3 main.py evaluate --scope survey --generate-new-predictions`
- Plot fitted functions:
  The used configuration for plotting is Random Forest. 
Change the configuration when calling `plot_results(...)` in the main.py. 
  ```
  python3 main.py plot --adverbial "long time ago"
  ```
- Predict for a custom event:
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
- Fits: `results/fits/`.
- Evaluation metrics/predictions: `results/evaluation/`.
- Plots: `results/plots/`.
- GPT predicted properties: `data/with_event_properties/event_properties.json` and `data/evaluation_survey/event_properties.json`.

## Notes
- Training and evaluation will generate/overwrite files under `results/` and may call GPT (cost + latency).
- Some functionality depends on scikit-learn/xgboost; ensure system libraries meet their requirements.
