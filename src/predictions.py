import numpy as np
import math
import json
import re
import os
import random
from openai import OpenAI
from pathlib import Path
from typing import Dict
from joblib import load
from collections import Counter
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (event_specific_function,
                        inverse_event_specific_function,
                        adverbial_specific_function,
                        gauss_inverse,
                        GPT_VERSION,
                        RESULTS_FILE_PATH,
                        VAGUE_ADVERBIALS,
                        EMBEDDING_RIDGE_FILE,
                        RANDOM_FOREST_FILE,
                        EMBEDDING_MODEL,
                        FREQUENCY_ORDER,
                        DURATION_ORDER)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_packed(path: Path = RESULTS_FILE_PATH / "event_adverbials") -> Dict[str, dict]:
    path = path.with_suffix('.json')
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def _safe_round(value):
    return round(value) if math.isfinite(value) else value

# Define the description mappings for each dimension
DURATION_DESCRIPTIONS = {
    1: "Very Short (Seconds to 5 minutes)",
    2: "Short (5 minutes to 1 Hour)",
    3: "Moderate (1-8 hours)",
    4: "Long (8–24 hours)",
    5: "Extended (More than a day)"
}

FREQUENCY_DESCRIPTIONS = {
    1: "One-time (Once or extremely rare per life)",
    2: "Rare (Once or a few times a year)",
    3: "Occasional (Monthly or semi-regular)",
    4: "Frequent (Weekly)",
    5: "Very Frequent (Daily or more)"
}

IMPORTANCE_DESCRIPTIONS = {
    1: "Not Important (Trivial or forgettable)",
    2: "Slightly Important (Minor relevance or enjoyment)",
    3: "Moderately Important (Some meaning)",
    4: "Important (Emotionally or practically significant)",
    5: "Very Important (Life-defining or deeply personal)"
}

def get_all_event_properties_gpt(events_nl, file_path, model_engine=GPT_VERSION):
    def format_scale(scale_dict):
        return "\n".join([f"{k} – {v}" for k, v in scale_dict.items()])

    instruction = f"""You are an assistant that evaluates events based on three dimensions: duration, frequency, and importance.
    - Duration → How long the event typically lasts.  
      Scale: {format_scale(DURATION_DESCRIPTIONS)}

    - Frequency → How often the event typically occurs.  
      Scale: {format_scale(FREQUENCY_DESCRIPTIONS)}

    - Importance → How important or meaningful the event is to the individual.  
      Scale: {format_scale(IMPORTANCE_DESCRIPTIONS)}

    Instructions:
    - Read the event description carefully.
    - Determine whether the event was performed by the user or someone else.
    - Use general human knowledge to estimate duration, frequency, and importance.
    - Always respond in the following exact format:
    Duration: <1–5>
    Frequency: <1–5>
    Importance: <1–5>
    """
    results = {}
    for event_nl in events_nl:
        event_clean = event_nl.replace("You", "I").replace("Tom", "A friend")
        prompt = f"Event: {event_clean}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]

        completion = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=0
        )

        content = completion.choices[0].message.content
        duration_match = re.search(r"Duration:\s*(\d)", content)
        frequency_match = re.search(r"Frequency:\s*(\d)", content)
        importance_match = re.search(r"Importance:\s*(\d)", content)

        if duration_match and frequency_match and importance_match:
            duration = int(duration_match.group(1))
            frequency = int(frequency_match.group(1))
            importance = int(importance_match.group(1))

            results[event_clean] = {
                "Duration": duration,
                "Duration_desc": DURATION_DESCRIPTIONS.get(duration),
                "Frequency": frequency,
                "Frequency_desc": FREQUENCY_DESCRIPTIONS.get(frequency),
                "Importance": importance,
                "Importance_desc": IMPORTANCE_DESCRIPTIONS.get(importance)
            }
        else:
            print(f"Could not parse response for event: {event_nl}")
            results[event_clean] = {
                "Duration": None,
                "Duration_desc": None,
                "Frequency": None,
                "Frequency_desc": None,
                "Importance": None,
                "Importance_desc": None,
                "raw_response": content
            }

    # Save to JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    return


def get_event_properties_gpt(event_nl, model_engine=GPT_VERSION):
    event_nl = event_nl.replace("You", "I")
    instruction = f"""
    You are an assistant that evaluates events based on two dimensions: duration and frequency.
    
    Definitions:
    - Duration → How long the event typically lasts.
      Scale: Minutes, Hours, Days, Weeks, Months, Years, Decades.
    - Frequency → How often the event typically occurs.
      Scale: Daily, Monthly, Yearly, Decadal, Once in Life.
    
    Instructions:
    - Read the event description carefully.
    - Determine whether the event was performed by the user or someone else.
    - Use general human knowledge to estimate duration and frequency.
    - Always respond in the following exact format:
    
    Duration: <one of the duration scale values>
    Frequency: <one of the frequency scale values>
    """

    prompt = f"""
    Event: {event_nl}
    """

    messages = [
        {"role": "system", "content": str(instruction)},
        {"role": "user", "content": str(prompt)}
    ]

    if "gpt-5" in model_engine:
        temperature = 1 # GPT-5 does not support a temperature of 0
        runs = 5
    else:
        temperature = 0
        runs = 1

    data = []
    for _ in range(0, runs):
        completion = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=temperature
        )
        content = completion.choices[0].message.content
        data.append(content)

    durations = [d.split("\n")[0].split(": ")[1] for d in data]
    frequencies = [d.split("\n")[1].split(": ")[1] for d in data]
    result = {
        "Duration": Counter(durations).most_common(1)[0][0],
        "Frequency": Counter(frequencies).most_common(1)[0][0]
    }
    return result
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_time_frame_embedding(event, adverbial, min_prob=0.6, max_prob=1.0):
    params = _load_packed()
    adverbial_mean = params["adverbial_means"][adverbial]
    adverbial_std = params["adverbial_stds"][adverbial]

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    ridge_model = load(RESULTS_FILE_PATH / EMBEDDING_RIDGE_FILE)

    vecc = embedding_model.encode(event)
    log_pred = ridge_model.predict(vecc.reshape(1, -1))[0]
    event_std = int(max(0, np.expm1(log_pred)))

    upper_raw = inverse_event_specific_function(gauss_inverse(max_prob, adverbial_mean, adverbial_std), event_std)
    lower_raw = inverse_event_specific_function(gauss_inverse(min_prob, adverbial_mean, adverbial_std), event_std)

    upper = max(0, _safe_round(upper_raw))
    lower = _safe_round(lower_raw)

    return upper, lower


def predict_adverbial_embedding(event_nl, minutes_ago):
    params = _load_packed()
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    ridge_model = load(RESULTS_FILE_PATH / EMBEDDING_RIDGE_FILE)

    vecc = embedding_model.encode(event_nl)
    log_pred = ridge_model.predict(vecc.reshape(1, -1))[0]
    event_std = int(max(0, np.expm1(log_pred)))

    adverbial_probs = {}
    for adverbial in VAGUE_ADVERBIALS:
        adverbial_mean = params["adverbial_means"][adverbial]
        adverbial_std = params["adverbial_stds"][adverbial]

        prob_adverbial = adverbial_specific_function(event_specific_function(minutes_ago, event_std), adverbial_mean, adverbial_std)
        adverbial_probs[adverbial] = prob_adverbial

    return adverbial_probs


def predict_time_frame_gpt_random_forest(event, adverbial, min_prob=0.6, max_prob=1.0):
    params = _load_packed()
    adverbial_mean = params["adverbial_means"][adverbial]
    adverbial_std = params["adverbial_stds"][adverbial]

    random_forest = load(RESULTS_FILE_PATH / RANDOM_FOREST_FILE)
    gpt_result = get_event_properties_gpt(event)
    required_keys = ["Frequency", "Duration"]
    missing_keys = [key for key in required_keys if key not in gpt_result]
    if missing_keys:
        raise KeyError(f"Missing keys from gpt_result, which is: {gpt_result}")
    event_properties = pd.DataFrame([{
        "Frequency": FREQUENCY_ORDER.index(gpt_result["Frequency"]),
        "Duration": DURATION_ORDER.index(gpt_result["Duration"])
    }])
    event_std = random_forest.predict(event_properties)[0]

    upper_raw = inverse_event_specific_function(gauss_inverse(max_prob, adverbial_mean, adverbial_std), event_std)
    lower_raw = inverse_event_specific_function(gauss_inverse(min_prob, adverbial_mean, adverbial_std), event_std)

    upper = max(0, _safe_round(upper_raw))
    lower = _safe_round(lower_raw)

    return upper, lower


def predict_adverbial_gpt_random_forest(event, minutes_ago,max_retries=10):
    def get_index(value, order_list):
        if value is None:
            return None
        for i, item in enumerate(order_list):
            if item.lower() in value.lower():
                return i
        return -1  # or raise an error or use None if unmatched

    params = _load_packed()
    random_forest = load(RESULTS_FILE_PATH / RANDOM_FOREST_FILE)

    for _ in range(max_retries):
        gpt_result = get_event_properties_gpt(event)

        duration = gpt_result.get("Duration")
        frequency = gpt_result.get("Frequency")

        freq_index = get_index(frequency, FREQUENCY_ORDER)
        dur_index = get_index(duration, DURATION_ORDER)

        if freq_index is not None and dur_index is not None:
            event_properties = pd.DataFrame([{
                "Frequency": freq_index,
                "Duration": dur_index
            }])
            break
    else:
        # After all retries, raise an error with last result for debugging
        raise ValueError(
            f"Failed to get valid Duration/Frequency after {max_retries} retries. "
            f"Last result: {gpt_result}"
        )

    event_std = random_forest.predict(event_properties)[0]

    adverbial_probs = {}
    for adverbial in VAGUE_ADVERBIALS:
        adverbial_mean = params["adverbial_means"][adverbial]
        adverbial_std = params["adverbial_stds"][adverbial]

        prob_adverbial = adverbial_specific_function(event_specific_function(minutes_ago, event_std), adverbial_mean, adverbial_std)
        adverbial_probs[adverbial] = prob_adverbial

    return adverbial_probs



def predict_time_frame_random_forest(duration, frequency, adverbial, min_prob=0.6, max_prob=1.0):
    params = _load_packed()
    adverbial_mean = params["adverbial_means"][adverbial]
    adverbial_std = params["adverbial_stds"][adverbial]

    random_forest = load(RESULTS_FILE_PATH / RANDOM_FOREST_FILE)
    event_properties = pd.DataFrame([{
        "Frequency": FREQUENCY_ORDER.index(frequency),
        "Duration": DURATION_ORDER.index(duration)
    }])
    event_std = random_forest.predict(event_properties)[0]

    upper_raw = inverse_event_specific_function(gauss_inverse(max_prob, adverbial_mean, adverbial_std), event_std)
    lower_raw = inverse_event_specific_function(gauss_inverse(min_prob, adverbial_mean, adverbial_std), event_std)

    upper = max(0, _safe_round(upper_raw))
    lower = _safe_round(lower_raw)

    return upper, lower


def predict_adverbial_random_forest(properties, minutes_ago):
    params = _load_packed()

    random_forest = load(RESULTS_FILE_PATH / RANDOM_FOREST_FILE)
    event_std = random_forest.predict(properties)[0]

    adverbial_probs = {}
    for adverbial in VAGUE_ADVERBIALS:
        adverbial_mean = params["adverbial_means"][adverbial]
        adverbial_std = params["adverbial_stds"][adverbial]

        prob_adverbial = adverbial_specific_function(event_specific_function(minutes_ago, event_std), adverbial_mean, adverbial_std)
        adverbial_probs[adverbial] = prob_adverbial

    return adverbial_probs



