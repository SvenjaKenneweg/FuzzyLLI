import os
import json
import re
from scipy.stats import spearmanr, kendalltau
from collections import defaultdict

from src.config import DATA_EVALUATION_SURVEY_PATH

from src.train import (
    fit_event_adverbials,
    fit_event_specific_embeddings,
    fit_event_specific_random_forest
)

from src.predictions import (
    predict_adverbial_embedding,
    predict_adverbial_gpt_random_forest,
    predict_adverbial_random_forest
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

def compare_fuzzy_ranks(fuzzy_prediction: dict, ground_truth: dict, verbose=True):
    all_adverbials = set(fuzzy_prediction.keys())
    truth_filled = {k: ground_truth.get(k, 0.0) for k in all_adverbials}

    def get_ranks(d):
        sorted_items = sorted(d.items(), key=lambda x: -x[1])
        return {k: i + 1 for i, (k, _) in enumerate(sorted_items)}

    pred_ranks = get_ranks(fuzzy_prediction)
    truth_ranks = get_ranks(truth_filled)

    # Convert to lists for correlation computation
    sorted_keys = sorted(all_adverbials)
    pred_rank_list = [pred_ranks[k] for k in sorted_keys]
    truth_rank_list = [truth_ranks[k] for k in sorted_keys]

    # Compute correlations
    spearman_corr, _ = spearmanr(pred_rank_list, truth_rank_list)
    kendall_corr, _ = kendalltau(pred_rank_list, truth_rank_list)

    if verbose:
        print("Predicted Ranks:", pred_ranks)
        print("Predicted Fuzzy:", fuzzy_prediction)
        print("Ground Truth Ranks:", truth_ranks)
        print("GT Prob:", ground_truth)
        print(f"Spearman Rank Correlation: {spearman_corr:.4f}")
        print(f"Kendall Tau: {kendall_corr:.4f}\n")

    return {
        'spearman': spearman_corr,
        'kendall': kendall_corr,
        'predicted_ranks': pred_ranks,
        'ground_truth_ranks': truth_ranks
    }


def evaluate_survey_gpt_random_forest(events_to_fit):
    fit_event_adverbials(events_to_fit)
    fit_event_specific_random_forest(events_to_fit)
    survey_data = get_percentages()

    for (event, minutes_ago), answers in survey_data.items():
        fuzzy_prediction = predict_adverbial_gpt_random_forest(event, minutes_ago)
        # Normalize ground truth to probabilities
        total_answers = sum(answers.values())
        ground_truth = {k: v / total_answers for k, v in answers.items()}
        compare_fuzzy_ranks(fuzzy_prediction, ground_truth)

def evaluate_survey_embedding(events_to_fit):
    fit_event_adverbials(events_to_fit)
    fit_event_specific_embeddings(events_to_fit)
    survey_data = get_percentages()
    for (event, minutes_ago), answers in survey_data.items():
        prediction = predict_adverbial_embedding(event, minutes_ago)
        print(prediction)
        for adverbial, count in answers.items():
            total_answers += count
        for adverbial, count in answers.items():
            print(f"  {adverbial}: {count / total_answers}")
        return

