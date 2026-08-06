import json
from functools import lru_cache
from importlib import resources
from joblib import load

from . import config


def _res(*parts: str):
    # points to src/fuzzylli/resources/...
    return resources.files("fuzzylli").joinpath("resources", *parts)


@lru_cache(maxsize=1)
def _load_event_adverbials() -> dict:
    with _res("event_adverbials.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _load_random_forest():
    rf_ref = _res(config.RANDOM_FOREST_FILE)
    # joblib needs a filesystem path -> as_file provides one
    with resources.as_file(rf_ref) as rf_path:
        return load(rf_path)


def predict_time_frame_random_forest(properties, adverbial, min_prob=0.6):
    params = _load_event_adverbials()
    adverbial_mean = params["adverbial_means"][adverbial]
    adverbial_std = params["adverbial_stds"][adverbial]

    rf = _load_random_forest()
    properties_df = config.properties_dataframe(properties)
    event_std = rf.predict(properties_df)[0]

    lower_adv, higher_adv = config.gauss_inverse(min_prob, adverbial_mean, adverbial_std)
    upper_raw = config.inverse_event_specific_function(higher_adv, event_std)
    lower_raw = config.inverse_event_specific_function(lower_adv, event_std)

    upper = max(0, config.safe_round(upper_raw))
    lower = config.safe_round(lower_raw)
    return upper, lower


def predict_adverbial_random_forest(properties, minutes_ago):
    params = _load_event_adverbials()

    rf = _load_random_forest()
    properties_df = config.properties_dataframe(properties)
    event_std = rf.predict(properties_df)[0]

    adverbial_probs = {}
    for adv in config.VAGUE_ADVERBIALS:
        mean = params["adverbial_means"][adv]
        std = params["adverbial_stds"][adv]
        p = config.adverbial_specific_function(
            config.event_specific_function(minutes_ago, event_std),
            mean,
            std,
        )
        adverbial_probs[adv] = p

    return adverbial_probs