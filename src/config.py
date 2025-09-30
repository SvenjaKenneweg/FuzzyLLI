from pathlib import Path
from typing import List
import numpy as np
import math
from scipy.special import erf, erfinv

# File paths
RESULTS_FILE_PATH = Path("results/fits/")
PLOT_FILE_PATH = Path("results/plots")
EVALUATION_FILE_PATH = Path("results/evaluation/seen_events/")
EVALUATION_SURVEY_FILE_PATH = Path("results/evaluation/unseen_events/")
DATA_EVALUATION_SURVEY_PATH =  Path("data/evaluation_survey")
DATA_DIR = Path("data/with_event_properties")
RESULTS_JSON = RESULTS_FILE_PATH / "event_adverbials.json"
EMBEDDING_RIDGE_FILE = 'event_embeddings_ridge.pkl'
RANDOM_FOREST_FILE = 'event_random_forest.pkl'
EMBEDDING_MODEL = 'paraphrase-MiniLM-L6-v2'
RESULTS_SIMPLE_FILE_PATH = Path("results/fits/simple_models/")
GRADIENT_BOOSTING_FILE = 'gradient_boosting_classifier.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'
ONE_HOT_ENCODER_FILE = 'onehotencoder.pkl'
RANDOM_FOREST_REGRESSOR_FILE = 'RandomForestRegressor.pkl'
XGB_REGRESSOR_FILE = 'XGBRegressor.pkl'

EVALUATION_EMBEDDING_FILE = EVALUATION_FILE_PATH / 'embeddings_regression.json'
EVALUATION_GPT4_RANDOM_FOREST_FILE = EVALUATION_FILE_PATH / 'gpt4_random_forest.json'
EVALUATION_GPT5_RANDOM_FOREST_FILE = EVALUATION_FILE_PATH / 'gpt5_random_forest.json'
EVALUATION_RANDOM_FOREST_FILE = EVALUATION_FILE_PATH / 'random_forest.json'
EVALUATION_GPT_CLASSIFIER_FILE = EVALUATION_FILE_PATH / 'gpt_classifier.json'
EVALUATION_GPT_REGRESSION_FILE = EVALUATION_FILE_PATH / 'gpt_regression.json'
GPT4_PROMPT_FILE = EVALUATION_FILE_PATH / 'GPT4_Prompts.json'
GPT5_PROMPT_FILE = EVALUATION_FILE_PATH / 'GPT5_Prompts.json'

GPT_VERSION = "gpt-4.1-2025-04-14" # "gpt-5-2025-08-07" #

EVALUATION_SURVEY_GPT4 = EVALUATION_SURVEY_FILE_PATH / 'gpt4_random_forest.json'
EVALUATION_SURVEY_GPT5 = EVALUATION_SURVEY_FILE_PATH / 'gpt5_random_forest.json'
GPT4_SURVEY_PROMPT_FILE = EVALUATION_SURVEY_FILE_PATH / 'GPT4_Prompts.json'
GPT5_SURVEY_PROMPT_FILE = EVALUATION_SURVEY_FILE_PATH / 'GPT5_Prompts.json'
EVALUATION_SURVEY_EMBEDDINGS = EVALUATION_SURVEY_FILE_PATH / 'embeddings_regression.json'
EVALUATION_SURVEY_GPT_CLASSIFIER = EVALUATION_SURVEY_FILE_PATH / 'gpt_classifier.json'
EVALUATION_SURVEY_GPT_REGRESSION = EVALUATION_SURVEY_FILE_PATH / 'gpt_regression.json'

# List of vague adverbials
VAGUE_ADVERBIALS: List[str] = ["recently", "just", "some time ago", "long time ago"]

# Duration and frequency order
DURATION_ORDER = ['Minutes', 'Hours', 'Days', 'Weeks', 'Months', 'Years', 'Decades']
FREQUENCY_ORDER = ['Daily', 'Monthly', 'Yearly', 'Decadal', 'Once in Life']


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
    return mean + std * np.sqrt(-2 * np.log(y))

def inverse_event_specific_function(y, std):
    clipped_y = max(-1, min(1, 2 * y - 1))
    return math.sqrt(2) * std * erfinv(clipped_y)