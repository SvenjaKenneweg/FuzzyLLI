from pathlib import Path
from typing import List
import numpy as np
import math
from scipy.special import erf, erfinv


# File paths
RESULTS_FILE_PATH = Path("results/fits/")
PLOT_FILE_PATH = Path("results/plots")
DATASET_SPATIAL_PATH = Path("datasets/spatial/free_scenario")

# List of vague spatial adverbials
VAGUE_ADVERBIALS: List[str] = ["close to", "moderately far", "far away"]

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

def normalized_gaussian(temporal_distance, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    return np.exp(-0.5 * ((temporal_distance - mu) / sigma) ** 2)

def inverse_event_specific_function(y, std):
    clipped_y = max(-1, min(1, 2 * y - 1))
    return math.sqrt(2) * std * erfinv(clipped_y)