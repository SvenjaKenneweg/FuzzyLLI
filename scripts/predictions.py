import math
import json
from pathlib import Path
from typing import Dict
from . import config

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_packed(path: Path = config.RESULTS_FILE_PATH / "event_adverbials") -> Dict[str, dict]:
    path = path.with_suffix('.json')
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def _safe_round(value):
    return round(value) if math.isfinite(value) else value

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_distance_fuzzylli(event, adverbial, *args):
    params = _load_packed()
    event_std = params["event_params"][event][0]
    adverbial_mean = params["adverbial_means"][adverbial]
    adverbial_std = params["adverbial_stds"][adverbial]
    lower_adverbial, higher_adverbial = config.gauss_inverse(1, adverbial_mean, adverbial_std)
    upper_raw = config.inverse_event_specific_function(higher_adverbial, event_std)
    lower_raw = config.inverse_event_specific_function(lower_adverbial, event_std)
    upper = _safe_round(upper_raw)
    lower = _safe_round(lower_raw)
    return upper, lower


def predict_adverbial_fuzzylli(event, minutes_ago, *args):
    params = _load_packed()
    event_std = params["event_params"][event][0]
    adverbial_probs = {}
    for adverbial in config.VAGUE_ADVERBIALS:
        adverbial_mean = params["adverbial_means"][adverbial]
        adverbial_std = params["adverbial_stds"][adverbial]

        prob_adverbial = config.adverbial_specific_function(config.event_specific_function(minutes_ago, event_std), adverbial_mean, adverbial_std)
        adverbial_probs[adverbial] = prob_adverbial

    return adverbial_probs