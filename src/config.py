from pathlib import Path
from typing import List
import numpy as np
import math
from scipy.special import erf, erfinv

# File paths
RESULTS_FILE_PATH = Path("results/fits/")
PLOT_FILE_PATH = Path("results/plots")
EVALUATION_FILE_PATH = Path("results/evaluation/training_dataset/")
EVALUATION_TEST_DATASET_FILE_PATH = Path("results/evaluation/test_dataset/")
DATASET_TEST_PATH =  Path("datasets/test")
DATASET_SPATIAL_PATH = Path("datasets/spatial/free_scenario")
DATA_DIR = Path("datasets/training")
EMBEDDING_RIDGE_FILE = 'configuration_word_embeddings.pkl'
RANDOM_FOREST_FILE = 'configuration_random_forest.pkl'
EMBEDDING_MODEL = 'paraphrase-MiniLM-L6-v2'
RESULTS_SIMPLE_FILE_PATH = Path("results/fits/simple_models/")
GRADIENT_BOOSTING_FILE = 'gradient_boosting_classifier.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'
ONE_HOT_ENCODER_FILE = 'onehotencoder.pkl'
XGB_REGRESSOR_FILE = 'XGBRegressor.pkl'

EVALUATION_EMBEDDING_FILE = EVALUATION_FILE_PATH / 'embeddings_regression.json'
EVALUATION_RANDOM_FOREST_FILE = EVALUATION_FILE_PATH / 'random_forest.json'
EVALUATION_FUNCTIONS_FILE = EVALUATION_FILE_PATH / 'functions_'
EVALUATION_CLASSIFIER_FILE = EVALUATION_FILE_PATH / 'classifier.json'
EVALUATION_REGRESSION_FILE = EVALUATION_FILE_PATH / 'regression.json'
GPT4_PROMPT_FILE = EVALUATION_FILE_PATH / 'GPT4_Prompts.json'
GPT5_PROMPT_FILE = EVALUATION_FILE_PATH / 'GPT5_Prompts.json'

GPT_VERSION = "gpt-4.1-2025-04-14"

EVALUATION_TEST_DATASET_RANDOM_FOREST = EVALUATION_TEST_DATASET_FILE_PATH / 'random_forest.json'
EVALUATION_TEST_DATASET_FUNCTIONS_FILE = EVALUATION_TEST_DATASET_FILE_PATH / 'functions_'
GPT4_TEST_DATASET_PROMPT_FILE = EVALUATION_TEST_DATASET_FILE_PATH / 'GPT4_Prompts.json'
GPT5_TEST_DATASET_PROMPT_FILE = EVALUATION_TEST_DATASET_FILE_PATH / 'GPT5_Prompts.json'
EVALUATION_TEST_DATSET_EMBEDDINGS = EVALUATION_TEST_DATASET_FILE_PATH / 'embeddings_regression.json'
EVALUATION_TEST_DATASET_CLASSIFIER = EVALUATION_TEST_DATASET_FILE_PATH / 'classifier.json'
EVALUATION_TEST_DATASET_REGRESSION = EVALUATION_TEST_DATASET_FILE_PATH / 'regression.json'

# List of vague adverbials
VAGUE_ADVERBIALS: List[str] = ["close to", "moderately far", "far away"] #["recently", "just", "some time ago", "long time ago"]

# Define the description mappings for each dimension
RICHNESS_DESCRIPTIONS = {
    1: "Minimal detail (routine, automatic)",
    2: "Simple, few contextual cues",
    3: "Moderate richness (some distinct aspects)",
    4: "Rich (many sensory/contextual elements)",
    5: "Extremely rich (vivid, complex, multisensory)"
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

properties_to_use = ["Richness", "Frequency", "Importance"]

# ========================
# Fitting Functions
# ========================

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

def powerlaw(vars, *params):
    # First parameter is the scaling factor 'a'
    a = params[0]
    exponents = params[1:]

    # Multiply all (variable ** corresponding exponent)
    result = a
    for var, exp in zip(vars, exponents):
        result *= (var ** exp)
    return result

def exp_decay(vars, a, b, c, d):
    Richness, Frequency, Importance = vars
    return a * np.exp(b * Richness + c * Importance - d * Frequency)