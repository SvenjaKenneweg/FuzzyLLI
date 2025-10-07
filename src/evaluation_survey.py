import os
import json
import re
import pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata
from collections import defaultdict

from src.config import (
                        DATA_EVALUATION_SURVEY_PATH, GPT_VERSION,
                        EVALUATION_SURVEY_GPT4, EVALUATION_SURVEY_GPT5, EVALUATION_SURVEY_RANDOM_FOREST,
                        GPT4_SURVEY_PROMPT_FILE, GPT5_SURVEY_PROMPT_FILE,
                        EVALUATION_SURVEY_EMBEDDINGS, VAGUE_ADVERBIALS,
                        EVALUATION_SURVEY_GPT_REGRESSION, EVALUATION_SURVEY_GPT_CLASSIFIER
                        )

from src.train import (
    fit_event_adverbials,
    fit_event_specific_embeddings,
    fit_event_specific_random_forest
)
from src.predictions import (
    predict_adverbial_embedding,
    predict_adverbial_random_forest,
    predict_adverbial_gpt_random_forest,
)
from src.evaluation import predict_gpt
from src.simple_models_training import fit_classifier, fit_regression
from src.simple_models_predictions import predict_adverbial_gpt_classifier, predict_adverbial_gpt_regression


# Regex to match questions like "You did X 3 hours ago."
pattern = re.compile(r'^(.*?)(\d+\s+\w+\s+ago)\.?$', re.IGNORECASE)

def time_ago_to_minutes(time_str):
    """
    Converts a time expression like '4 hours ago' to minutes (int).
    Supports minutes, hours, and days.
    """
    time_str = time_str.lower().replace('ago', '').strip()
    # Match expressions like "5 minutes", "2 hours", "1 day"
    match = re.match(r'(\d+)\s+(seconds|minute|minutes|hour|hours|day|days|week|weeks|months|month|year|years)', time_str)
    if not match:
        return None  # Could not parse
    number = int(match.group(1))
    unit = match.group(2)
    if 'seconds' in unit:
        return number / 60
    elif 'minute' in unit:
        return number
    elif 'hour' in unit:
        return number * 60
    elif 'day' in unit:
        return number * 1440  # 24*60
    elif 'week' in unit:
        return number * 10080
    elif 'month' in unit:
        return number * 43800
    elif 'year' in unit:
        return number * 525600
    else:
        return None  # Unexpected unit

def get_percentages():
    event_time_answer_counts = defaultdict(lambda: defaultdict(int))
    attention_check_questions = defaultdict(lambda: defaultdict(int))
    votes_path = os.path.join(DATA_EVALUATION_SURVEY_PATH, "votes")
    for filename in os.listdir(votes_path):
        if filename.endswith('.json'):
            file_path = os.path.join(votes_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                answers = data.get('answers', {})
                for key_str, answer in answers.items():
                    try:
                        key_data = json.loads(key_str)
                        question = key_data.get('question', '').strip()

                        # Handle attention check separately
                        if "attention check" in question.lower():
                            attention_check_questions[question][answer] += 1
                            continue

                        # Try to parse event and time
                        match = pattern.match(question)
                        if match:
                            event = match.group(1).strip()
                            time_ago = time_ago_to_minutes(match.group(2).strip().replace("ago", "").strip())
                            if time_ago is None:
                                print(f"Could not convert time to minutes: '{time_ago}' in question: '{question}'")
                            else:
                                if "some" in answer:
                                    event_time_answer_counts[(event, time_ago)]["some time ago"] += 1
                                elif "long" in answer:
                                    event_time_answer_counts[(event, time_ago)]["long time ago"] += 1
                                else:
                                    event_time_answer_counts[(event, time_ago)][answer] += 1
                        elif "half a year ago" in question:
                            event = "You bought a house"
                            time_ago = 262800
                            if "some" in answer:
                                event_time_answer_counts[(event, time_ago)]["some time ago"] += 1
                            elif "long" in answer:
                                event_time_answer_counts[(event, time_ago)]["long time ago"] += 1
                            else:
                                event_time_answer_counts[(event, time_ago)][answer] += 1
                        else:
                            print(f"Could not parse this question: '{question}'")

                    except json.JSONDecodeError:
                        print(f"Invalid JSON key in file {filename}: {key_str}")
    return event_time_answer_counts


def compare_fuzzy_ranks(fuzzy_prediction: dict, ground_truth: dict):
    # Ensure adverbials are aligned
    all_adverbials = sorted(set(fuzzy_prediction.keys()))

    # Fill missing entries in ground truth with 0.0
    truth_filled = {k: ground_truth.get(k, 0.0) for k in all_adverbials}

    # Extract probability lists
    pred_probs = [fuzzy_prediction.get(k, 0.0) for k in all_adverbials]
    truth_probs = [truth_filled[k] for k in all_adverbials]

    # Convert probabilities to ranks (higher prob = lower rank)
    pred_ranks = rankdata([-p for p in pred_probs], method="min")
    truth_ranks = rankdata([-p for p in truth_probs], method="min")

    # Compute rank correlation coefficients
    spearman_corr, _ = spearmanr(pred_ranks, truth_ranks)
    kendall_corr, _ = kendalltau(pred_ranks, truth_ranks)

    return {
        'spearman': spearman_corr,
        'kendall': kendall_corr
    }

def run_survey_evaluation_and_save_preds(events_to_fit, fit_fn, predict_fn, events_to_fit_nl = None):
    fit_event_adverbials(events_to_fit)
    if fit_fn is not None:
        if events_to_fit_nl is not None:
            fit_fn(events_to_fit, events_to_fit_nl)
        else:
            fit_fn(events_to_fit)
    survey_data = get_percentages()
    raw_results = []

    for (event, minutes_ago), answers in survey_data.items():
        if predict_fn == predict_adverbial_random_forest:
            file_path = f"{DATA_EVALUATION_SURVEY_PATH}/event_properties.json"
            with open(file_path, "r", encoding="utf-8") as fh:
                event_properties = json.load(fh)

            properties = pd.DataFrame([{
                'Frequency': event_properties[event.replace("You", "I")]["Frequency"],
                'Duration': event_properties[event.replace("You", "I")]["Duration"],
                'Importance': event_properties[event.replace("You", "I")]["Importance"]
            }])
            fuzzy_prediction =  predict_fn(properties, minutes_ago)
        else:
            fuzzy_prediction = predict_fn(event, minutes_ago)
        # Normalize ground truth to probabilities
        ground_truth = {k: v / sum(answers.values()) for k, v in answers.items()}
        for adv in VAGUE_ADVERBIALS:
            if adv not in ground_truth:
                ground_truth[adv] = 0.0

        raw_results.append({
            "Event": event,
            "Minutes ago": minutes_ago,
            "Prediction": fuzzy_prediction,
            "GT": ground_truth
        })
    return raw_results

def evaluate_survey_gpt_random_forest(events_to_fit, events_to_fit_nl):
    raw_results = run_survey_evaluation_and_save_preds(events_to_fit, fit_event_specific_random_forest, predict_adverbial_gpt_random_forest, events_to_fit_nl = events_to_fit_nl)
    output = {
        "raw": raw_results,
        "GPT-Version": GPT_VERSION
    }
    if "gpt-4" in GPT_VERSION:
        save_file = EVALUATION_SURVEY_GPT4
    else:
        save_file = EVALUATION_SURVEY_GPT5
    with open(save_file, "w") as f:
        json.dump(output, f, indent=4)
    return

def evaluate_survey_random_forest(events_to_fit, events_to_fit_nl):
    raw_results = run_survey_evaluation_and_save_preds(events_to_fit, fit_event_specific_random_forest, predict_adverbial_random_forest, events_to_fit_nl = events_to_fit_nl)
    output = {
        "raw": raw_results
    }
    with open(EVALUATION_SURVEY_RANDOM_FOREST, "w") as f:
        json.dump(output, f, indent=4)
    return

def evaluate_survey_embedding(events_to_fit, events_to_fit_nl):
    raw_results = run_survey_evaluation_and_save_preds(events_to_fit, fit_event_specific_embeddings, predict_adverbial_embedding, events_to_fit_nl = events_to_fit_nl)
    output = {
        "raw": raw_results,
    }
    with open(EVALUATION_SURVEY_EMBEDDINGS, "w") as f:
        json.dump(output, f, indent=4)
    return

def evaluate_survey_gpt_classifier(events_to_fit, events_to_fit_nl):
    raw_results = run_survey_evaluation_and_save_preds(events_to_fit, fit_classifier, predict_adverbial_gpt_classifier, events_to_fit_nl = events_to_fit_nl)
    output = {
        "raw": raw_results,
        "GPT-Version": GPT_VERSION
    }
    with open(EVALUATION_SURVEY_GPT_CLASSIFIER, "w") as f:
        json.dump(output, f, indent=4)
    return

def evaluate_survey_gpt_regression(events_to_fit, events_to_fit_nl):
    raw_results = run_survey_evaluation_and_save_preds(events_to_fit, fit_regression, predict_adverbial_gpt_regression, events_to_fit_nl = events_to_fit_nl)
    output = {
        "raw": raw_results,
        "GPT-Version": GPT_VERSION
    }
    with open(EVALUATION_SURVEY_GPT_REGRESSION, "w") as f:
        json.dump(output, f, indent=4)
    return


def evaluate_survey_gpt():
    predictions_data = []
    survey_data = get_percentages()

    for (event, minutes_ago), answers in survey_data.items():
        ground_truth = {k: v / sum(answers.values()) for k, v in answers.items()}
        for adv in VAGUE_ADVERBIALS:
            if adv not in ground_truth:
                ground_truth[adv] = 0.0

        prediction, instruction, prompt = predict_gpt(event, minutes_ago)
        # --- Step 5: Save prediction to JSON structure ---
        predictions_data.append({
            "Prediction": prediction,
            "GT": ground_truth,
            "Prompt": prompt
        })

    # --- Step 6: Write all predictions to JSON (overwrite at the end) ---
    json_drop = {
        "raw": predictions_data,
        "Instruction": instruction,
        "GPT-Version": GPT_VERSION
    }
    if "gpt-4" in GPT_VERSION:
        save_file = GPT4_SURVEY_PROMPT_FILE
    else:
        save_file = GPT5_SURVEY_PROMPT_FILE
    with open(save_file, "w") as f:
        json.dump(json_drop, f, indent=4)
    return

