import json
import statistics
from pathlib import Path
from typing import List
import openai
import os
from openai import OpenAI

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score
)

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
from src.config import (VAGUE_ADVERBIALS,
                        RESULTS_FILE_PATH,
                        DURATION_ORDER,
                        FREQUENCY_ORDER,
                        DATA_DIR,
                        event_specific_function,
                        adverbial_specific_function, GPT_VERSION, GPT_PROMPT_FILE)


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

def evaluate_model(events, fit_models_fn, predict_fn, metric='mae'):
    errors = {adv: [] for adv in VAGUE_ADVERBIALS}
    all_errors = []

    for i, event in enumerate(events):
        other_events = events[:i] + events[i+1:]
        fit_event_adverbials(other_events)
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

def evaluate_advanced_model(events, fit_models_fn, predict_fn):
    y_true_list = []  # ground-truth label sets per sample
    y_pred_list = []  # predicted label sets per sample

    for i, event in enumerate(events):
        other_events = events[:i] + events[i+1:]
        fit_event_adverbials(other_events)
        fit_models_fn(other_events)

        cleaned_data = get_cleaned_data(event)

        # --- Step 1: Gather all possible numeric keys ---
        overall_targets = {adv: {k: float(np.median(v)) for k, v in cleaned_data[adv].items()} for adv in
                           VAGUE_ADVERBIALS}
        all_keys = sorted({int(k) for adv in overall_targets.values() for k in adv.keys()})

        # --- Step 2: Function to get value for a given key, using nearest if missing ---
        def get_value_for_key(adv_dict, target):
            numeric_keys = sorted(int(k) for k in adv_dict.keys())
            if str(target) in adv_dict:
                return adv_dict[str(target)]
            nearest = min(numeric_keys, key=lambda x: abs(x - target))
            return adv_dict[str(nearest)]

        # --- Step 3: Build inverted dict ---
        best_adverbials = {}
        for key in all_keys:
            values = {adv: get_value_for_key(adv_dict, key) for adv, adv_dict in overall_targets.items()}
            max_val = max(values.values())
            best_adverbials[key] = [adv for adv, val in values.items() if val == max_val]

        # --- Collect predictions vs truth for metrics ---
        for minutes_ago, adverbials in best_adverbials.items():
            if predict_fn == predict_adverbial_random_forest:
                props = get_event_properties(event)
                freq_votes = [p['Frequency'] for p in props if 'Frequency' in p]
                dur_votes = [p['Duration'] for p in props if 'Duration' in p]

                if 6 in dur_votes:
                    dur_votes = adjust_duration_votes(dur_votes)

                freq = FREQUENCY_ORDER[int(statistics.median(freq_votes))]
                dur = DURATION_ORDER[int(statistics.median(dur_votes))]
                predictions = predict_fn(dur, freq, int(minutes_ago))
            else:
                predictions = predict_fn(event, int(minutes_ago))

            best_predicted_adverbials = [adv for adv, val in predictions.items() if val == max(predictions.values())]
            y_true_list.append(adverbials)
            y_pred_list.append(best_predicted_adverbials)

    # ==== Metrics ====
    # Binarize over the union of all observed labels
    mlb = MultiLabelBinarizer()
    mlb.fit(y_true_list + y_pred_list)
    y_true_bin = mlb.transform(y_true_list)
    y_pred_bin = mlb.transform(y_pred_list)

    results = {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision (sample)": precision_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "recall (sample)": recall_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "f1 (sample)": f1_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true_bin, y_pred_bin),
        "jaccard (sample)": jaccard_score(y_true_bin, y_pred_bin, average="samples"),
    }
    return results



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
            print(str(adv) + " " + str(targets))
            if adv not in errors:
                errors.setdefault(adv, {})

            for minutes_ago, true_value in targets.items():
                predictions = predict_fn(dur, freq, int(minutes_ago))[adv]

                # If prediction is a dict (multiple models)
                if isinstance(predictions, dict):
                    for model_name, pred_value in predictions.items():
                        errors[adv].setdefault(model_name, []).append(calculate_error(pred_value, true_value))
                        all_errors.setdefault(model_name, []).append(calculate_error(pred_value, true_value))
                else:
                    # Single prediction (default model)
                    model_name = 'baseline'
                    errors[adv].setdefault(model_name, []).append(calculate_error(predictions, true_value))
                    all_errors.setdefault(model_name, []).append(calculate_error(predictions, true_value))

    # Compute results
    results = {}
    for model_name, model_errors in all_errors.items():
        results[model_name] = {
            'per_adverbial': {adv: np.mean(errors[adv][model_name]) for adv in VAGUE_ADVERBIALS},
            'overall_score': np.mean(model_errors) if metric == 'mae' else np.sqrt(
                np.mean(np.square(model_errors)))
        }

    return results

def evaluate_advanced_baseline_model(events, fit_models_fn, predict_fn, model_to_predict=None):
    y_true_list = []  # ground-truth label sets per sample
    y_pred_list = []  # predicted label sets per sample

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

        # --- Step 1: Gather all possible numeric keys ---
        overall_targets = {adv: {k: float(np.median(v)) for k, v in cleaned_data[adv].items()} for adv in VAGUE_ADVERBIALS}
        all_keys = sorted({int(k) for adv in overall_targets.values() for k in adv.keys()})

        # --- Step 2: Function to get value for a given key, using nearest if missing ---
        def get_value_for_key(adv_dict, target):
            numeric_keys = sorted(int(k) for k in adv_dict.keys())
            if str(target) in adv_dict:
                return adv_dict[str(target)]
            nearest = min(numeric_keys, key=lambda x: abs(x - target))
            return adv_dict[str(nearest)]

        # --- Step 3: Build inverted dict ---
        best_adverbials = {}
        for key in all_keys:
            values = {adv: get_value_for_key(adv_dict, key) for adv, adv_dict in overall_targets.items()}
            max_val = max(values.values())
            best_adverbials[key] = [adv for adv, val in values.items() if val == max_val]

        # --- Collect predictions vs truth for metrics ---
        for minutes_ago, adverbials in best_adverbials.items():
            if model_to_predict is not None:
                predictions = predict_fn(dur, freq, int(minutes_ago), model_to_predict=model_to_predict)
            else:
                predictions = predict_fn(dur, freq, int(minutes_ago))
            best_predicted_adverbials = [adv for adv, val in predictions.items() if val == max(predictions.values())]
            y_true_list.append(adverbials)
            y_pred_list.append(best_predicted_adverbials)

    # ==== Metrics ====
    # Binarize over the union of all observed labels
    mlb = MultiLabelBinarizer()
    mlb.fit(y_true_list + y_pred_list)
    y_true_bin = mlb.transform(y_true_list)
    y_pred_bin = mlb.transform(y_pred_list)

    results = {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision (sample)": precision_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "recall (sample)": recall_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "f1 (sample)": f1_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true_bin, y_pred_bin),
        "jaccard (sample)": jaccard_score(y_true_bin, y_pred_bin, average="samples"),
    }
    return results





client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def gpt_chat_answer(model_engine, instructions, prompt):
    messages = [
        {"role": "system", "content": str(instructions)},
        {"role": "user", "content": str(prompt)}
    ]
    completion = client.chat.completions.create(
        model=model_engine,
        messages=messages,
        temperature=0 # GPT-5 does not support setting the temperature
    )
    return completion.choices[0].message.content.strip()


def predict_gpt(event_name, minutes_ago, model_engine=GPT_VERSION):
    # System instructions: guidance on GPT's behavior
    instructions = """
    You are to choose the most appropriate adverbial(s) from the following options: 
    just, recently, some time ago, long time ago.
    Base your choice on the given event and the number of minutes that have passed since it happened.
    Respond ONLY with the appropriate adverbial(s) and nothing else.
    """

    # Build user prompt
    prompt = f"""
    Event: {event_name}
    Minutes since event: {minutes_ago}
    """

    # Get GPT's answer
    output = gpt_chat_answer(model_engine, instructions, prompt)
    return output, instructions, prompt
# ---------------------------------------------------------------------------
# Model-specific Evaluators
# ---------------------------------------------------------------------------

def evaluate_embedding(events, metric='mae'):
    return evaluate_model(events, fit_event_specific_embeddings, predict_adverbial_embedding, metric)

def evaluate_advanced_embedding(events):
    return evaluate_advanced_model(events, fit_event_specific_embeddings, predict_adverbial_embedding)

def evaluate_gpt_random_forest(events, metric='mae'):
    return evaluate_model(events, fit_event_specific_random_forest, predict_adverbial_gpt_random_forest, metric)

def evaluate_advanced_gpt_random_forest(events):
    return evaluate_advanced_model(events, fit_event_specific_random_forest, predict_adverbial_gpt_random_forest)

def evaluate_random_forest(events, metric='mae'):
    return evaluate_model(events, fit_event_specific_random_forest, predict_adverbial_random_forest, metric)

def evaluate_advanced_random_forest(events):
    return evaluate_advanced_model(events, fit_event_specific_random_forest, predict_adverbial_random_forest)

def evaluate_classifier(events, metric='mae'):
    return evaluate_baseline_model(events, fit_classifier, predict_adverbial_classifier, metric)

def evaluate_advanced_classifier(events):
    return evaluate_advanced_baseline_model(events, fit_classifier, predict_adverbial_classifier)

def evaluate_regression(events, metric='mae'):
    return evaluate_baseline_model(events, fit_regression, predict_adverbial_regression, metric)

def evaluate_advanced_regression(events, model_to_predict="RandomForestRegressor"):
    print(model_to_predict + ":")
    return evaluate_advanced_baseline_model(events, fit_regression, predict_adverbial_regression,model_to_predict=model_to_predict)




def evaluate_gpt(events):
    y_true_list = []  # ground-truth label sets per sample
    y_pred_list = []  # predicted label sets per sample

    with open(GPT_PROMPT_FILE, "w") as f:
        json.dump([], f)  # Start with an empty list
    predictions_data = []

    for i, event in enumerate(events):
        cleaned_data = get_cleaned_data(event)

        # --- Step 1: Gather all possible numeric keys ---
        overall_targets = {adv: {k: float(np.median(v)) for k, v in cleaned_data[adv].items()} for adv in VAGUE_ADVERBIALS}
        all_keys = sorted({int(k) for adv in overall_targets.values() for k in adv.keys()})

        # --- Step 2: Function to get value for a given key, using nearest if missing ---
        def get_value_for_key(adv_dict, target):
            numeric_keys = sorted(int(k) for k in adv_dict.keys())
            if str(target) in adv_dict:
                return adv_dict[str(target)]
            nearest = min(numeric_keys, key=lambda x: abs(x - target))
            return adv_dict[str(nearest)]

        # --- Step 3: Build inverted dict ---
        best_adverbials = {}
        for key in all_keys:
            values = {adv: get_value_for_key(adv_dict, key) for adv, adv_dict in overall_targets.items()}
            max_val = max(values.values())
            best_adverbials[key] = [adv for adv, val in values.items() if val == max_val]

        # --- Collect predictions vs truth for metrics ---
        for minutes_ago, adverbials in best_adverbials.items():
            prediction, instruction, prompt = predict_gpt(event, minutes_ago)
            y_true_list.append(adverbials)
            y_pred_list.append([prediction])
            # --- Step 5: Save prediction to JSON structure ---
            predictions_data.append({
                "prediction": prediction,
                "true": adverbials,
                "prompt": prompt
            })

    # --- Step 6: Write all predictions to JSON (overwrite at the end) ---
    with open(GPT_PROMPT_FILE, "w") as f:
        json.dump(predictions_data, f, indent=4)
        json.dump(instruction, f, indent=4)
        json.dump(GPT_VERSION, f, indent=4)

    # ==== Metrics ====
    # Binarize over the union of all observed labels
    mlb = MultiLabelBinarizer()
    mlb.fit(y_true_list + y_pred_list)
    y_true_bin = mlb.transform(y_true_list)
    y_pred_bin = mlb.transform(y_pred_list)

    results = {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision (sample)": precision_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "recall (sample)": recall_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "f1 (sample)": f1_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true_bin, y_pred_bin),
        "jaccard (sample)": jaccard_score(y_true_bin, y_pred_bin, average="samples"),
    }
    return results
