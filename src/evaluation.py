import json
import statistics
from pathlib import Path
from typing import List
import openai
import os
from scipy.stats import kendalltau
from openai import OpenAI
from collections import Counter

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    ndcg_score,
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
                        DURATION_ORDER,
                        FREQUENCY_ORDER,
                        DATA_DIR,
                        GPT_VERSION,
                        EVALUATION_FILE_PATH,
                        GPT4_PROMPT_FILE, GPT5_PROMPT_FILE, EVALUATION_EMBEDDING_FILE,
                        EVALUATION_REGRESSION_FILE, EVALUATION_CLASSIFIER_FILE,
                        EVALUATION_RANDOM_FOREST_FILE, EVALUATION_GPT4_RANDOM_FOREST_FILE, EVALUATION_GPT5_RANDOM_FOREST_FILE)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def adjust_duration_votes(votes: List[int]) -> List[int]:
    """Correct for late addition of 'Hours' in unseen_events options."""
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

def run_evaluation_and_save_preds(events, fit_models_fn, predict_fn, events_nl=None):
    raw_results = []

    for i, event in enumerate(events):
        other_events = events[:i] + events[i+1:]
        fit_event_adverbials(other_events)
        if events_nl:
            other_events_nl = events_nl[:i] + events_nl[i + 1:]
            fit_models_fn(other_events, other_events_nl)
        else:
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
        gt_adverbials = {}
        for key in all_keys:
            values = {adv: get_value_for_key(adv_dict, key) for adv, adv_dict in overall_targets.items()}
            gt_adverbials[key] = values

        # --- Collect predictions vs truth for metrics ---
        for minutes_ago, adverbial_values in gt_adverbials.items():
            if predict_fn in (predict_adverbial_random_forest,
                              predict_adverbial_classifier,
                              predict_adverbial_regression):
                props = get_event_properties(event)
                freq_votes = [p['Frequency'] for p in props if 'Frequency' in p]
                dur_votes = [p['Duration'] for p in props if 'Duration' in p]

                if 6 in dur_votes:
                    dur_votes = adjust_duration_votes(dur_votes)

                freq = FREQUENCY_ORDER[int(statistics.median(freq_votes))]
                dur = DURATION_ORDER[int(statistics.median(dur_votes))]
                predictions = predict_fn(dur, freq, int(minutes_ago))
            else:
                if events_nl:
                    predictions = predict_fn(events_nl[i], int(minutes_ago))
                else:
                    predictions = predict_fn(event, int(minutes_ago))

            raw_results.append({
                "Event": event,
                "Minutes ago": minutes_ago,
                "Prediction":  predictions,
                "GT": adverbial_values
            })
    return raw_results

def calculate_metrics_seen_events():
    def _rank_groups(d):
        """Return rank groups like: [[best ties...], [2nd...], ...]"""
        items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
        groups, last = [], None
        for k, v in items:
            if last is None or v != last:
                groups.append([k])
                last = v
            else:
                groups[-1].append(k)
        return groups

    # Loop through all .json files containing the predictions
    for json_file in EVALUATION_FILE_PATH.glob("*.json"):
        print(f"\nFile: {json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        tau_list, ndcg_list, acc1_list, prec1_list, rec1_list = [], [], [], [], []

        for row in data:
            event = row.get("Event", "<unknown>")
            pred = row["Prediction"]
            true = row["GT"]

            labels = sorted(set(pred) | set(true))
            y_score = np.array([pred.get(lbl, 0.0) for lbl in labels], dtype=float)
            y_true = np.array([true.get(lbl, 0.0) for lbl in labels], dtype=float)

            # Ranks (for display)
            pred_groups = _rank_groups({lbl: s for lbl, s in zip(labels, y_score)})
            true_groups = _rank_groups({lbl: r for lbl, r in zip(labels, y_true)})

            # Kendall's tau-b (SciPy)
            tau = kendalltau(y_score, y_true, variant="b").correlation
            if tau is None:  # fallback if undefined (e.g., all ties)
                tau = 0.0

            # NDCG (scikit-learn) — shape (1, n_labels)
            if np.all(y_true == 0.0):
                ndcg = 0.0
            else:
                ndcg = float(ndcg_score(y_true.reshape(1, -1),
                                        y_score.reshape(1, -1)))

            # Top-1 metrics
            top_idx = int(np.argmax(y_score))
            top_label = labels[top_idx]
            max_true = float(np.max(y_true))
            top_true_set = {labels[i] for i, v in enumerate(y_true) if v == max_true}

            acc1 = 1.0 if top_label in top_true_set else 0.0
            prec1 = 1.0 if y_true[top_idx] > positive_threshold else 0.0

            positives = {labels[i] for i, v in enumerate(y_true) if v > positive_threshold}
            rec1 = (1.0 / len(positives)) if positives and (top_label in positives) else 0.0

            # Collect
            tau_list.append(tau);
            ndcg_list.append(ndcg)
            acc1_list.append(acc1);
            prec1_list.append(prec1);
            rec1_list.append(rec1)

            # Print per-event
            def _fmt(groups):
                return " | ".join(f"{i + 1}: {g}" for i, g in enumerate(groups))

            print(f"\nEvent: {event}")
            print(f"  Prediction rank: {_fmt(pred_groups)}")
            print(f"  True rank      : {_fmt(true_groups)}")
            print(f"  Kendall's tau-b: {tau:.6f}")
            print(f"  NDCG           : {ndcg:.6f}")
            print(f"  Top-1 Acc / Prec / Rec: {acc1:.3f} / {prec1:.3f} / {rec1:.3f}")

        # File-level averages
        if tau_list:
            n = len(tau_list)
            print("\nAverages in this file:")
            print(f"  Kendall's tau-b (avg): {np.mean(tau_list):.6f}")
            print(f"  NDCG (avg)           : {np.mean(ndcg_list):.6f}")
            print(f"  Top-1 Accuracy (avg) : {np.mean(acc1_list):.6f}")
            print(f"  Top-1 Precision (avg): {np.mean(prec1_list):.6f}")
            print(f"  Top-1 Recall (avg)   : {np.mean(rec1_list):.6f}")



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def gpt_chat_answer(model_engine, instructions, prompt):
    messages = [
        {"role": "system", "content": str(instructions)},
        {"role": "user", "content": str(prompt)}
    ]
    if "5" in model_engine:
        temperature = 1 # GPT-5 does not support a temperature of 0
    else:
        temperature = 0

    completion = client.chat.completions.create(
        model=model_engine,
        messages=messages,
        temperature=temperature
    )
    return completion.choices[0].message.content.strip()


def predict_gpt(event_nl, minutes_ago, model_engine=GPT_VERSION):
    # System instructions: guidance on GPT's behavior
    instructions = """
    You are to choose the most appropriate adverbial(s) from the following options: 
    just, recently, some time ago, long time ago.
    Base your choice on the given event and the number of minutes that have passed since it happened.
    Respond ONLY with the appropriate adverbial(s) and nothing else.
    """

    # Build user prompt
    prompt = f"""
    Event: {event_nl}
    Minutes since event: {minutes_ago}
    """

    # Get GPT's answer
    if "gpt-5" in model_engine: # As version 5 does not support the temperature 0 setting get the output 5 times and chose the most frequent adverbials
        outputs = []
        for i in range(0, 5):
            outputs.append(gpt_chat_answer(model_engine, instructions, prompt))
        counter = Counter(outputs)
        output, count = counter.most_common(1)[0]
    else:
        output = gpt_chat_answer(model_engine, instructions, prompt)
    return output, instructions, prompt

# ---------------------------------------------------------------------------
# Model-specific Evaluators
# ---------------------------------------------------------------------------
def get_predictions_embedding(events, events_nl):
    raw_results = run_evaluation_and_save_preds(events, fit_event_specific_embeddings, predict_adverbial_embedding, events_nl=events_nl)
    output = {
        "raw": raw_results,
    }
    with open(EVALUATION_EMBEDDING_FILE, "w") as f:
        json.dump(output, f, indent=4)
    return

def get_predictions_gpt_random_forest(events, events_nl):
    raw_results = run_evaluation_and_save_preds(events, fit_event_specific_random_forest, predict_adverbial_gpt_random_forest, events_nl=events_nl)
    output = {
        "raw": raw_results,
        "GPT-Version": GPT_VERSION
    }
    if "gpt-4" in GPT_VERSION:
        save_file = EVALUATION_GPT4_RANDOM_FOREST_FILE
    else:
        save_file = EVALUATION_GPT5_RANDOM_FOREST_FILE
    with open(save_file, "w") as f:
        json.dump(output, f, indent=4)
    return

def get_predictions_random_forest(events):
    raw_results = run_evaluation_and_save_preds(events, fit_event_specific_random_forest, predict_adverbial_random_forest)
    output = {
        "raw": raw_results,
    }
    with open(EVALUATION_RANDOM_FOREST_FILE, "w") as f:
        json.dump(output, f, indent=4)
    return

def get_predictions_classifier(events):
    raw_results = run_evaluation_and_save_preds(events, fit_classifier, predict_adverbial_classifier)
    output = {
        "raw": raw_results,
    }
    with open(EVALUATION_CLASSIFIER_FILE, "w") as f:
        json.dump(output, f, indent=4)
    return

def get_predictions_regression(events):
    raw_results = run_evaluation_and_save_preds(events, fit_regression, predict_adverbial_regression)
    output = {
        "raw": raw_results,
    }
    with open(EVALUATION_REGRESSION_FILE, "w") as f:
        json.dump(output, f, indent=4)
    return


def evaluate_gpt(events, events_nl):
    y_true_list = []  # ground-truth label sets per sample
    y_pred_list = []  # predicted label sets per sample

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
            prediction, instruction, prompt = predict_gpt(events_nl[i], minutes_ago)
            y_true_list.append(adverbials)
            y_pred_list.append([prediction])
            # --- Step 5: Save prediction to JSON structure ---
            predictions_data.append({
                "Prediction": prediction,
                "GT": adverbials,
                "Prompt": prompt
            })

    # --- Step 6: Write all predictions to JSON (overwrite at the end) ---
    json_drop = [{
        "Raw": predictions_data,
        "Instruction": instruction,
        "GPT-Version": GPT_VERSION
    }]
    if "gpt-4" in GPT_VERSION:
        save_file = GPT4_PROMPT_FILE
    else:
        save_file = GPT5_PROMPT_FILE
    with open(save_file, "w") as f:
        json.dump(json_drop, f, indent=4)
    return
