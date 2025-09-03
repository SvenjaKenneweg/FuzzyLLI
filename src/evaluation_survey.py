import os
import json
import re
from scipy.stats import spearmanr, kendalltau, rankdata
from collections import defaultdict

from src.config import (
                        DATA_EVALUATION_SURVEY_PATH, GPT_VERSION,
                        EVALUATION_SURVEY_GPT4, EVALUATION_SURVEY_GPT5,
                        EVALUATION_SURVEY_EMBEDDINGS
                        )

from src.train import (
    fit_event_adverbials,
    fit_event_specific_embeddings,
    fit_event_specific_random_forest
)

from src.predictions import (
    predict_adverbial_embedding,
    predict_adverbial_gpt_random_forest,
    predict_adverbial_random_forest,
    predict_random
)

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
    for filename in os.listdir(DATA_EVALUATION_SURVEY_PATH):
        if filename.endswith('.json'):
            file_path = os.path.join(DATA_EVALUATION_SURVEY_PATH, filename)
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

def evaluate_survey(events_to_fit, fit_fn, predict_fn, events_to_fit_nl = None):
    fit_event_adverbials(events_to_fit)
    if fit_fn is not None:
        if events_to_fit_nl is not None:
            fit_fn(events_to_fit, events_to_fit_nl)
        else:
            fit_fn(events_to_fit)
    survey_data = get_percentages()

    spearman_total = 0.0
    kendall_total = 0.0
    count = 0

    # Use defaultdict to accumulate correlations per event
    per_event_spearman = defaultdict(list)
    per_event_kendall = defaultdict(list)

    raw_results = {}

    for (event, minutes_ago), answers in survey_data.items():
        fuzzy_prediction = predict_fn(event, minutes_ago)

        # Normalize ground truth to probabilities
        ground_truth = {k: v / sum(answers.values()) for k, v in answers.items()}

        corr = compare_fuzzy_ranks(fuzzy_prediction, ground_truth)

        # Store JSON-safe values
        raw_results[f"{event}: {minutes_ago}"] = {
            'ground_truth': ground_truth,
            'fuzzy_prediction': fuzzy_prediction
        }

        if corr['spearman'] is not None and corr['kendall'] is not None:
            spearman_total += corr['spearman']
            kendall_total += corr['kendall']
            count += 1

            per_event_spearman[event].append(corr['spearman'])
            per_event_kendall[event].append(corr['kendall'])

    # Compute average per-event correlations
    per_event_correlations = {
        event: {
            'spearman': sum(s_list) / len(s_list) if s_list else None,
            'kendall': sum(k_list) / len(k_list) if k_list else None
        }
        for event, s_list in per_event_spearman.items()
        for k_list in [per_event_kendall[event]]
    }
    results = {
        'spearman': spearman_total / count if count else None,
        'kendall': kendall_total / count if count else None,
        'per_event': per_event_correlations,
    }
    return results, raw_results

def evaluate_survey_gpt_random_forest(events_to_fit):
    results, raw_results = evaluate_survey(events_to_fit, fit_event_specific_random_forest, predict_adverbial_gpt_random_forest)
    # Combine into one JSON-serializable dictionary
    output = {
        "evaluation": results,
        "raw": raw_results,
        "GPT-Version": GPT_VERSION
    }
    if "gpt-4" in GPT_VERSION:
        save_file = EVALUATION_SURVEY_GPT4
    else:
        save_file = EVALUATION_SURVEY_GPT5
    with open(save_file, "w") as f:
        json.dump(output, f, indent=4)
    return results

def evaluate_survey_embedding(events_to_fit, events_to_fit_nl):
    results, raw_results = evaluate_survey(events_to_fit, fit_event_specific_embeddings, predict_adverbial_embedding, events_to_fit_nl = events_to_fit_nl)
    # Combine into one JSON-serializable dictionary
    output = {
        "evaluation": results,
        "raw": raw_results
    }
    with open(EVALUATION_SURVEY_EMBEDDINGS, "w") as f:
        json.dump(output, f, indent=4)
    return results

def evaluate_survey_baseline():
    results, raw_results = evaluate_survey(None, None, predict_random)
    return results

