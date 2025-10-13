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
                        powerlaw, exp_decay,
                        GPT_VERSION,
                        RESULTS_FILE_PATH,
                        VAGUE_ADVERBIALS,
                        EMBEDDING_RIDGE_FILE,
                        RANDOM_FOREST_FILE,
                        EMBEDDING_MODEL,
                        RICHNESS_DESCRIPTIONS, FREQUENCY_DESCRIPTIONS, IMPORTANCE_DESCRIPTIONS, powerlaw)

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

def get_all_event_properties_gpt(events_nl, file_path, model_engine=GPT_VERSION):
    def format_scale(scale_dict):
        return "\n".join([f"{k} – {v}" for k, v in scale_dict.items()])

    instruction = f"""You are an assistant that evaluates events based on three dimensions: richness, frequency, and importance.
    - Richness → How vivid, detailed, and contextually rich the event is.  
      Scale: {format_scale(RICHNESS_DESCRIPTIONS)}

    - Frequency → How often the event typically occurs.  
      Scale: {format_scale(FREQUENCY_DESCRIPTIONS)}

    - Importance → How important or meaningful the event is to the individual.  
      Scale: {format_scale(IMPORTANCE_DESCRIPTIONS)}

    Instructions:
    - Read the event description carefully.
    - Determine whether the event was performed by the user or someone else.
    - Use general human knowledge to estimate richness, frequency, and importance.
    - Always respond in the following exact format:
    Richness: <1–5>
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
        richness_match = re.search(r"Richness:\s*(\d)", content)
        frequency_match = re.search(r"Frequency:\s*(\d)", content)
        importance_match = re.search(r"Importance:\s*(\d)", content)

        if richness_match and frequency_match and importance_match:
            richness = int(richness_match.group(1))
            frequency = int(frequency_match.group(1))
            importance = int(importance_match.group(1))

            results[event_clean] = {
                "Richness": richness,
                "Richness_desc": RICHNESS_DESCRIPTIONS.get(richness),
                "Frequency": frequency,
                "Frequency_desc": FREQUENCY_DESCRIPTIONS.get(frequency),
                "Importance": importance,
                "Importance_desc": IMPORTANCE_DESCRIPTIONS.get(importance)
            }
        else:
            print(f"Could not parse response for event: {event_nl}")
            results[event_clean] = {
                "Richness": None,
                "Richness_desc": None,
                "Frequency": None,
                "Frequency_desc": None,
                "Importance": None,
                "Importance_desc": None,
                "raw_response": content
            }

        print(event_nl)
        print(int(richness_match.group(1)))
        print(RICHNESS_DESCRIPTIONS.get(int(richness_match.group(1))))
        print("")
    # # Save to JSON
    # with open(file_path, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)
    return
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


def predict_adverbial_embedding(event_nl, minutes_ago, *args):
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

def predict_time_frame_random_forest(properties, adverbial, min_prob=0.6, max_prob=1.0):
    params = _load_packed()
    adverbial_mean = params["adverbial_means"][adverbial]
    adverbial_std = params["adverbial_stds"][adverbial]

    random_forest = load(RESULTS_FILE_PATH / RANDOM_FOREST_FILE)
    event_std = random_forest.predict(properties)[0]

    upper_raw = inverse_event_specific_function(gauss_inverse(max_prob, adverbial_mean, adverbial_std), event_std)
    lower_raw = inverse_event_specific_function(gauss_inverse(min_prob, adverbial_mean, adverbial_std), event_std)

    upper = max(0, _safe_round(upper_raw))
    lower = _safe_round(lower_raw)

    return upper, lower


def predict_adverbial_random_forest(properties, minutes_ago, *args):
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


def predict_adverbial_functions(properties, minutes_ago, function_to_predict, *args):
    params = _load_packed()

    model = load( f"{RESULTS_FILE_PATH}/{function_to_predict.__name__}.pkl")
    a, b, c, d = model["params"].values()
    event_std = function_to_predict((properties["Richness"].values, properties["Frequency"].values, properties["Importance"].values), a, b, c, d)

    adverbial_probs = {}
    for adverbial in VAGUE_ADVERBIALS:
        adverbial_mean = params["adverbial_means"][adverbial]
        adverbial_std = params["adverbial_stds"][adverbial]

        prob_adverbial = adverbial_specific_function(event_specific_function(minutes_ago, event_std), adverbial_mean, adverbial_std)
        adverbial_probs[adverbial] = prob_adverbial[0]
    return adverbial_probs



