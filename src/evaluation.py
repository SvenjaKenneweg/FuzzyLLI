import json
import statistics
from pathlib import Path
from typing import List
import math
import openai
import os
from scipy.stats import kendalltau
from openai import OpenAI
from collections import Counter

import numpy as np
import pandas as pd
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
    predict_adverbial_gpt_classifier,
    predict_adverbial_gpt_regression
)
from src.config import (VAGUE_ADVERBIALS,
                        DURATION_ORDER,
                        FREQUENCY_ORDER,
                        DATA_DIR,
                        GPT_VERSION,
                        EVALUATION_FILE_PATH,
                        GPT4_PROMPT_FILE, GPT5_PROMPT_FILE, EVALUATION_EMBEDDING_FILE,
                        EVALUATION_GPT_REGRESSION_FILE, EVALUATION_GPT_CLASSIFIER_FILE,
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

def calculate_metrics(file_path):
    def rank_from_scores(scores, descending=True):
        """
        Normalize scores into ranks (1 = best, ties get average rank).
        Supports:
          - dict[label -> score]  -> true ranking (rankable=True)
          - list/tuple/set[str]   -> all items become rank 1 (rankable=False)
          - str                   -> that single label becomes rank 1 (rankable=False)

        Returns:
          ranks: dict[label] -> rank (float)
          rankable: bool     -> True iff input was a dict with actual scores
        """
        # dict case: compute average (mid) ranks
        if isinstance(scores, dict) and len(scores) > 0:
            items = list(scores.items())
            items.sort(key=lambda x: x[1], reverse=descending)
            ranks = {}
            i = 0
            while i < len(items):
                j = i
                while j < len(items) and items[j][1] == items[i][1]:
                    j += 1
                avg_rank = (i + 1 + j) / 2.0  # average of 1-based positions for ties
                for k in range(i, j):
                    ranks[items[k][0]] = avg_rank
                i = j
            return ranks, True

        # list/tuple/set -> all are top-1
        if isinstance(scores, (list, tuple, set)) and len(scores) > 0:
            return {str(lbl): 1.0 for lbl in scores}, False

        # single string -> top-1
        if isinstance(scores, str) and scores:
            return {scores: 1.0}, False

        # empty / unsupported
        return {}, False

    def top1_on_ranks(rank_pred, rank_gt):
        # labels = sorted(set(rank_pred.keys()) | set(rank_gt.keys()))

        min_pred = min(rank_pred[l] for l in rank_pred)
        pred_top = {l for l in rank_pred if rank_pred[l] == min_pred}
        min_gt = min(rank_gt[l] for l in rank_gt)
        gt_best = {l for l in rank_gt if rank_gt[l] == min_gt}
        inter = pred_top & gt_best

        # Accuracy@1: any correct among the predicted top-rank set?
        acc1 = 1.0 if len(inter) > 0 else 0.0
        # Precision@1: fraction of predicted top-rank items that are correct
        p1 = (len(inter) / len(pred_top)) if len(pred_top) > 0 else 0.0
        # Recall@1: fraction of GT-best items recovered at predicted top rank
        r1 = (len(inter) / len(gt_best)) if len(gt_best) > 0 else 0.0
        return acc1, p1, r1

    # Loop through all .json files containing the predictions
    for json_file in file_path.glob("*.json"):
        print(f"\nFile: {json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)["raw"]

        results = []
        for r in data:
            pred = r["Prediction"]  # dict[label] -> score
            gt = r["GT"]  # dict[label] -> relevance

            rank_pred, pred_rankable = rank_from_scores(pred, descending=True)
            rank_gt, gt_rankable = rank_from_scores(gt, descending=True)
            rankable = pred_rankable and gt_rankable

            if rankable:
                # ---- Kendall's Tau-b on ranks ----
                # Align to the same label order (use GT's labels as base)
                labels = list(gt.keys())
                x = [rank_pred[l] for l in labels]
                y = [rank_gt[l] for l in labels]
                tau, _ = kendalltau(x, y, variant="b", nan_policy="raise", alternative="two-sided", method="auto")

                # ---- NDCG computed on ranks ----
                # Transform ranks to "higher is better" scores: rel = max_rank + 1 - rank
                max_r = max(max(rank_gt.values()), max(rank_pred.values()))
                y_true = [[(max_r + 1 - rank_gt[l]) for l in labels]]
                y_score = [[(max_r + 1 - rank_pred[l]) for l in labels]]
                ndcg = ndcg_score(y_true, y_score) if any(v > 0 for v in y_true[0]) else 0.0
            else:
                # not rankable (e.g. GPT4 Prompts having only predicted the first item)
                tau = 0.0
                ndcg = 0.0

            # ---- Top-1 on ranks ----
            acc1, p1, r1 = top1_on_ranks(rank_pred, rank_gt)

            results.append({
                "KendallTauB": float(tau) if not math.isnan(tau) else 0.0,
                "NDCG": float(ndcg),
                "Top1_Accuracy": acc1,
                "Top1_Precision": p1,
                "Top1_Recall": r1,
            })

        df = pd.DataFrame(results)
        overall = df.mean(numeric_only=True).to_frame(name="Overall").T
        print(overall)
    return

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

def get_predictions_classifier(events, events_nl):
    raw_results = run_evaluation_and_save_preds(events, fit_classifier, predict_adverbial_gpt_classifier, events_nl=events_nl)
    output = {
        "raw": raw_results,
        "GPT-Version": GPT_VERSION
    }
    with open(EVALUATION_GPT_CLASSIFIER_FILE, "w") as f:
        json.dump(output, f, indent=4)
    return

def get_predictions_regression(events, events_nl):
    raw_results = run_evaluation_and_save_preds(events, fit_regression, predict_adverbial_gpt_regression, events_nl=events_nl)
    output = {
        "raw": raw_results,
        "GPT-Version": GPT_VERSION
    }
    with open(EVALUATION_GPT_REGRESSION_FILE, "w") as f:
        json.dump(output, f, indent=4)
    return


def evaluate_gpt(events, events_nl):
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
            # --- Step 5: Save prediction to JSON structure ---
            predictions_data.append({
                "Prediction": prediction,
                "GT": adverbials,
                "Prompt": prompt
            })

    # --- Step 6: Write all predictions to JSON (overwrite at the end) ---
    json_drop = {
        "raw": predictions_data,
        "Instruction": instruction,
        "GPT-Version": GPT_VERSION
    }
    if "gpt-4" in GPT_VERSION:
        save_file = GPT4_PROMPT_FILE
    else:
        save_file = GPT5_PROMPT_FILE
    with open(save_file, "w") as f:
        json.dump(json_drop, f, indent=4)
    return
