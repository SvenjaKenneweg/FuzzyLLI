import json
from pathlib import Path
import numpy as np
import statistics
from typing import Dict, List

from src.predictions import predict_adverbial_embedding, predict_adverbial_gpt_random_forest, predict_adverbial_random_forest
from src.train import fit_event_adverbials, fit_event_specific_embeddings, fit_event_specific_random_forest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_FILE_PATH = Path("results/fits/")
DATA_DIR = Path("data/with_event_properties")

VAGUE_ADVERBIALS: List[str] = [
    "recently",
    "just",
    "some time ago",
    "long time ago",
]

duration_order = ['Minutes', 'Hours', 'Days', 'Weeks', 'Months', 'Years', 'Decades']
frequency_order = ['Daily', 'Monthly', 'Yearly', 'Decadal', 'Once in Life']

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(events, event_specific_function, adverbial_specific_function,
                   fit_models_fn, predict_fn, metric='mae'):
    errors = {adv: [] for adv in VAGUE_ADVERBIALS}
    all_errors = []

    for i, event in enumerate(events):
        events_without_event = events[:i] + events[i+1:]

        # Fit embeddings or models
        fit_event_adverbials(events_without_event, event_specific_function, adverbial_specific_function)
        fit_models_fn(events_without_event)

        # Load ground truth
        file_path = f"{DATA_DIR}/{event}/cleanedData_minutes.json"
        with open(file_path, "r", encoding="utf-8") as fh:
            values_vagueAdverbial = json.load(fh)

        for adverbial in VAGUE_ADVERBIALS:
            target_values = {k: np.median(v) for k, v in values_vagueAdverbial[adverbial].items()}
            for minutes_ago, true_value in target_values.items():
                if predict_fn == predict_adverbial_random_forest:
                    file_path = f"{DATA_DIR}/{event}/event_properties.json"
                    with open(file_path, "r", encoding="utf-8") as fh:
                        event_properties = json.load(fh)

                    frequency_votes = [d['Frequency'] for d in event_properties if 'Frequency' in d]
                    duration_votes = [d['Duration'] for d in event_properties if 'Duration' in d]

                    # Because participants first could not select "Hours" as Duration I entered hours later and it became entry 6
                    # Real entry 6 ("Decades") is 5
                    if 6 in duration_votes:
                        duration_votes = [
                            1 if x == 6 else (x + 1 if x != 0 else x)
                            for x in duration_votes
                        ]

                    median_frequency = frequency_order[int(statistics.median(frequency_votes))]
                    median_duration = duration_order[int(statistics.median(duration_votes))]
                    predicted = predict_fn(median_duration, median_frequency, int(minutes_ago))[adverbial]
                else:
                    predicted = predict_fn(event, int(minutes_ago))[adverbial]
                error = abs(predicted - true_value)
                errors[adverbial].append(error)
                all_errors.append(error)

    # # Evaluation summary
    # print("\n=== Evaluation Summary ===")
    # for adverbial, adverbial_errors in errors.items():
    #     if metric == 'mae':
    #         score = np.mean(adverbial_errors)
    #         print(f"{adverbial} - MAE: {score:.4f}")
    #     elif metric == 'rmse':
    #         score = np.sqrt(np.mean(np.square(adverbial_errors)))
    #         print(f"{adverbial} - RMSE: {score:.4f}")
    #     else:
    #         raise ValueError(f"Unsupported metric: {metric}. Use 'mae' or 'rmse'.")

    if metric == 'mae':
        overall_score = np.mean(all_errors)
    elif metric == 'rmse':
        overall_score = np.sqrt(np.mean(np.square(all_errors)))

    return {
        'per_adverbial': {adv: np.mean(errs) for adv, errs in errors.items()},
        'overall_score': overall_score
    }


# === Specific wrappers ===

def evaluate_embedding(events, event_specific_function, adverbial_specific_function, metric='mae'):
    return evaluate_model(
        events,
        event_specific_function,
        adverbial_specific_function,
        fit_models_fn=fit_event_specific_embeddings,
        predict_fn=predict_adverbial_embedding,
        metric=metric
    )


def evaluate_gpt_random_forest(events, event_specific_function, adverbial_specific_function, metric='mae'):
    return evaluate_model(
        events,
        event_specific_function,
        adverbial_specific_function,
        fit_models_fn=fit_event_specific_random_forest,
        predict_fn=predict_adverbial_gpt_random_forest,
        metric=metric
    )

def evaluate_random_forest(events, event_specific_function, adverbial_specific_function, metric='mae'):
    return evaluate_model(
        events,
        event_specific_function,
        adverbial_specific_function,
        fit_models_fn=fit_event_specific_random_forest,
        predict_fn=predict_adverbial_random_forest,
        metric=metric
    )

# Also Comparison GPT.