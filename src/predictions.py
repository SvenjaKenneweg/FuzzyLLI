import numpy as np
import math
import json
import re
import os
from openai import OpenAI
from pathlib import Path
from typing import Dict
from joblib import load
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (event_specific_function,
                        inverse_event_specific_function,
                        adverbial_specific_function,
                        gauss_inverse,
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

def _get_event_properties_gpt(event, gpt_model="gpt-4"):
    prompt = f"""
    I will provide you with an event performed by yourself. Your task is to evaluate the event based on three dimensions: duration and frequency.

    Please use the following definitions and rating scales:

    Duration – How long does the event typically last?
    Scale: Minutes, Hours, Days, Weeks, Months, Years, Decades

    Frequency – How often does the event typically occur?
    Scale: Daily, Monthly, Yearly, Decadal, Once in Life

    Event: {event}

    Respond in this exact format:
    Duration: <value>
    Frequency: <value>
    """

    response = client.chat.completions.create(
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    # Basic parsing
    matches = re.findall(r"(Duration|Frequency):\s*(.+)", content)
    response_dict = {k: v.strip() for k, v in matches}
    return response_dict
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


def predict_adverbial_embedding(event, minutes_ago):
    params = _load_packed()
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    ridge_model = load(RESULTS_FILE_PATH / EMBEDDING_RIDGE_FILE)

    vecc = embedding_model.encode(event)
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
    gpt_result = _get_event_properties_gpt(event)
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

    for attempt in range(max_retries):
        gpt_result = _get_event_properties_gpt(event)

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


def predict_adverbial_random_forest(duration, frequency, minutes_ago):
    params = _load_packed()

    random_forest = load(RESULTS_FILE_PATH / RANDOM_FOREST_FILE)
    event_properties = pd.DataFrame([{
        "Frequency": FREQUENCY_ORDER.index(frequency),
        "Duration": DURATION_ORDER.index(duration)
    }])
    event_std = random_forest.predict(event_properties)[0]

    adverbial_probs = {}
    for adverbial in VAGUE_ADVERBIALS:
        adverbial_mean = params["adverbial_means"][adverbial]
        adverbial_std = params["adverbial_stds"][adverbial]

        prob_adverbial = adverbial_specific_function(event_specific_function(minutes_ago, event_std), adverbial_mean, adverbial_std)
        adverbial_probs[adverbial] = prob_adverbial

    return adverbial_probs



