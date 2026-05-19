# src/fuzzylli/config.py
import math
import pandas as pd
from scipy.special import erf, erfinv
import numpy as np

RANDOM_FOREST_FILE = "configuration_random_forest.pkl"
EMBEDDING_RIDGE_FILE = "configuration_word_embeddings.pkl"
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
KGQA_EVENT_TYPES_FILE = "kgqa_event_types.json"
VAGUE_ADVERBIALS = ["just", "recently", "some time ago", "long time ago"]  # set your real list

def properties_dataframe(properties):
    """
    Accepts either dict or list[dict]; returns a pandas DataFrame in the expected column order.
    """
    if isinstance(properties, dict):
        properties = [properties]
    df = pd.DataFrame(properties)
    # If you require a specific order / subset:
    # df = df[["Richness", "Frequency", "Importance"]]
    return df


def safe_round(x: float) -> int:
    # match your previous behavior if different
    return int(round(float(x)))


# --- math helpers ---
def adverbial_specific_function(x, mean, std):
    # Fuzzy Gaussian
    return (np.exp(-0.5 * ((x - mean) / std) ** 2))

def event_specific_function(temporal_distance, std):
    # Cumulative distribution function of a gaussian distribution
    return 1/2 * (erf(temporal_distance / (math.sqrt(2) * std))+1)

def gauss_inverse(y, mean, std):
    val = std * np.sqrt(-2 * np.log(y))
    return mean - val, mean + val

def inverse_event_specific_function(y, std):
    clipped_y = max(-1, min(1, 2 * y - 1))
    return math.sqrt(2) * std * erfinv(clipped_y)
