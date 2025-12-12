# FuzzyLLI – A fuzzy probabilistic model of human interpretation of vague temporal adverbials


This repository implements **FuzzyLLI**, a hybrid ML + cognitive modeling framework that predicts:

- **Temporal adverbials** (e.g., *just*, *recently*, *long ago*) from events  
- **Event recency** (minutes ago) from adverbials  
- **Event properties** (Richness, Frequency, Importance) using GPT  

FuzzyLLI integrates:

- Event-specific embeddings  
- Random forests  
- Power-law & exponential decay functions  
- Classifier + regression baselines  
- GPT-based feature extraction  
- Rich evaluation + visualization tools

This repository accompanies the paper:

📄 *[A fuzzy probabilistic model of human interpretation of vague temporal adverbials]*  
🔗 *[Under Review]*

The entry point is: `src/main.py`.

---

## What `src/main.py` does

Running `src/main.py` executes a full pipeline:

1. **Fetch event properties via GPT**
   - Writes event properties to JSON files (one for training events, one for survey events).

2. **Train models**
   - Event adverbial models
   - Event-specific embedding models
   - Event-specific random forests
   - Function-based models (power law + exp decay)
   - Baseline classifier and regression

3. **Evaluate**
   - On “seen” experimental events
   - On “unseen” survey-style events
   - Optionally across **all combinations of event properties** (Richness/Frequency/Importance)

4. **Plot results**
   - Generates plots under `results/plots/...`

5. **Run a demo prediction**
   - Predict adverbials and/or timeframe for a test event.

---

## Project structure (high level)

The main script orchestrates functionality from these modules:

- `src/train.py`
  - `fit_event_adverbials`
  - `fit_event_specific_embeddings`
  - `fit_event_specific_random_forest`
  - `fit_event_specific_functions`

- `src/predictions.py`
  - `predict_time_frame_embedding`
  - `predict_adverbial_embedding`
  - `predict_time_frame_random_forest`
  - `predict_adverbial_random_forest`
  - `predict_adverbial_functions`
  - `get_all_event_properties_gpt`

- `src/evaluation.py`
  - `get_predictions_*` (writes predictions to disk)
  - `calculate_metrics` (computes evaluation metrics from saved predictions)

- `src/evaluation_survey.py`
  - `evaluate_survey_*` (unseen-event evaluation)

- `src/plot.py`
  - `plot_all_persons_event_adverbials`
  - `plot_single_events`
  - `plot_events_adverbials`

- `src/config.py`
  - Paths (data, evaluation outputs)
  - Model settings (e.g., which event properties to use)
  - Function choices (`powerlaw`, `exp_decay`)

---

## Requirements

Install dependencies with `pip install -r requirements.txt`.

## Configuration

Configurations are saved in `src/config.py`.

## Running

### Run the full pipeline
From the repository root:

```bash
python -m src.main
