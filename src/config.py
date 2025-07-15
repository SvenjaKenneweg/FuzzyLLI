from pathlib import Path
from typing import List
import numpy as np
import math
from scipy.special import erf

# File paths
RESULTS_FILE_PATH = Path("results/fits/")
PLOT_FILE_PATH = Path("results/plots")
DATA_DIR = Path("data/with_event_properties")
RESULTS_JSON = RESULTS_FILE_PATH / "event_adverbials.json"

# List of vague adverbials
VAGUE_ADVERBIALS: List[str] = ["recently", "just", "some time ago", "long time ago"]

# Duration and frequency order
DURATION_ORDER = ['Minutes', 'Hours', 'Days', 'Weeks', 'Months', 'Years', 'Decades']
FREQUENCY_ORDER = ['Daily', 'Monthly', 'Yearly', 'Decadal', 'Once in Life']


# ========================
# Fitting Functions
# ========================
def adverbial_specific_function(x, mean, std):
    # Normalized gaussian
    return (np.exp(-0.5 * ((x - mean) / std) ** 2))
def event_specific_function(temporal_distance, std):
    # Cumulative distribution function of a gaussian distribution
    return 1/2 * (erf(temporal_distance / (math.sqrt(2) * std))+1)

