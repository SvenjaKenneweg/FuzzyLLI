import json
import statistics
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.predictions import (
    predict_adverbial_embedding,
    predict_adverbial_gpt_random_forest,
    predict_adverbial_random_forest
)
from src.train import (
    fit_event_adverbials,
    fit_event_specific_embeddings,
    fit_event_specific_random_forest
)
from src.simple_models_training import fit_classifier, fit_regression
from src.simple_models_predictions import (
    predict_adverbial_classifier,
    predict_adverbial_regression
)
from src.config import VAGUE_ADVERBIALS, DURATION_ORDER, FREQUENCY_ORDER, DATA_DIR


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def adjust_duration_votes(votes: List[int]) -> List[int]:
    """Correct for late addition of 'Hours' in survey options."""
    return [
        1 if v == 6 else (v + 1 if v != 0 else v)
        for v in votes
    ]


def get_event_properties(event: str):
    with open(DATA_DIR / event / "event_properties.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_cleaned_data(event: str):
    with open(DATA_DIR / event / "cleanedData_minutes.json", "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_error(predicted: float, true: float) -> float:
    return abs(predicted - true)


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

def evaluate_model(events, event_specific_fn, adverbial_specific_fn,
                   fit_models_fn, predict_fn, metric='mae'):
    errors = {adv: [] for adv in VAGUE_ADVERBIALS}
    all_errors = []

    for i, event in enumerate(events):
        other_events = events[:i] + events[i+1:]
        fit_event_adverbials(other_events, event_specific_fn, adverbial_specific_fn)
        fit_models_fn(other_events)

        cleaned_data = get_cleaned_data(event)

        for adv in VAGUE_ADVERBIALS:
            targets = {k: np.median(v) for k, v in cleaned_data[adv].items()}
            for minutes_ago, true_value in targets.items():
                if predict_fn == predict_adverbial_random_forest:
                    props = get_event_properties(event)
                    freq_votes = [p['Frequency'] for p in props if 'Frequency' in p]
                    dur_votes = [p['Duration'] for p in props if 'Duration' in p]

                    if 6 in dur_votes:
                        dur_votes = adjust_duration_votes(dur_votes)

                    freq = FREQUENCY_ORDER[int(statistics.median(freq_votes))]
                    dur = DURATION_ORDER[int(statistics.median(dur_votes))]
                    predicted = predict_fn(dur, freq, int(minutes_ago))[adv]
                else:
                    predicted = predict_fn(event, int(minutes_ago))[adv]

                error = calculate_error(predicted, true_value)
                errors[adv].append(error)
                all_errors.append(error)

    overall = np.mean(all_errors) if metric == 'mae' else np.sqrt(np.mean(np.square(all_errors)))

    return {
        'per_adverbial': {adv: np.mean(errs) for adv, errs in errors.items()},
        'overall_score': overall
    }


def evaluate_baseline_model(events, fit_models_fn, predict_fn, metric='mae'):
    errors = {}
    all_errors = {}

    for i, event in enumerate(events):
        other_events = events[:i] + events[i+1:]
        fit_models_fn(other_events)

        cleaned_data = get_cleaned_data(event)
        props = get_event_properties(event)

        freq_votes = [p['Frequency'] for p in props if 'Frequency' in p]
        dur_votes = [p['Duration'] for p in props if 'Duration' in p]

        if 6 in dur_votes:
            dur_votes = adjust_duration_votes(dur_votes)

        freq = FREQUENCY_ORDER[int(statistics.median(freq_votes))]
        dur = DURATION_ORDER[int(statistics.median(dur_votes))]

        for adv in VAGUE_ADVERBIALS:
            targets = {k: np.median(v) for k, v in cleaned_data[adv].items()}
            for minutes_ago, true_value in targets.items():
                predictions = predict_fn(dur, freq, int(minutes_ago))[adv]

                if isinstance(predictions, dict):
                    for model, pred in predictions.items():
                        errors.setdefault(model, {a: [] for a in VAGUE_ADVERBIALS})
                        all_errors.setdefault(model, [])
                        err = calculate_error(pred, true_value)
                        errors[model][adv].append(err)
                        all_errors[model].append(err)
                else:
                    model = 'default'
                    errors.setdefault(model, {a: [] for a in VAGUE_ADVERBIALS})
                    all_errors.setdefault(model, [])
                    err = calculate_error(predictions, true_value)
                    errors[model][adv].append(err)
                    all_errors[model].append(err)

    results = {}
    for model, errs in errors.items():
        overall = np.mean(all_errors[model]) if metric == 'mae' else np.sqrt(np.mean(np.square(all_errors[model])))
        results[model] = {
            'per_adverbial': {adv: np.mean(errs[adv]) for adv in VAGUE_ADVERBIALS},
            'overall_score': overall
        }

    return results


# ---------------------------------------------------------------------------
# Model-specific Evaluators
# ---------------------------------------------------------------------------

def evaluate_embedding(events, event_fn, adverbial_fn, metric='mae'):
    return evaluate_model(events, event_fn, adverbial_fn,
                          fit_event_specific_embeddings,
                          predict_adverbial_embedding, metric)


def evaluate_gpt_random_forest(events, event_fn, adverbial_fn, metric='mae'):
    return evaluate_model(events, event_fn, adverbial_fn,
                          fit_event_specific_random_forest,
                          predict_adverbial_gpt_random_forest, metric)


def evaluate_random_forest(events, event_fn, adverbial_fn, metric='mae'):
    return evaluate_model(events, event_fn, adverbial_fn,
                          fit_event_specific_random_forest,
                          predict_adverbial_random_forest, metric)


def evaluate_classifier(events, metric='mae'):
    return evaluate_baseline_model(events, fit_classifier,
                                   predict_adverbial_classifier, metric)


def evaluate_regression(events, metric='mae'):
    return evaluate_baseline_model(events, fit_regression,
                                   predict_adverbial_regression, metric)
