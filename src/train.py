from __future__ import annotations

import inspect
import json
import pickle
import joblib
import statistics
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer

from src.config import RESULTS_FILE_PATH, DATA_DIR, event_specific_function, adverbial_specific_function


# ========================
# Parameters for the fitting
# ========================

# Initial guesses for vague adverbials' mean values
adverbials_initial_means: Dict[str, float] = {
    "recently": 0.5,
    "just": 0.5,
    "some time ago": 0.9,
    "long time ago": 0.9,
}

# Standard deviation initial guesses for adverbials
adverbials_initial_stds: Dict[str, float] = {
    "recently": 0.2,
    "just": 0.1,
    "some time ago": 0.2,
    "long time ago": 0.3,
}

# Bounds for the fitting (lower, upper)
fitting_bounds_events = ([1e-6], [np.inf])
fitting_bounds_adverbials_mean = ([0.4], [1.0])
fitting_bounds_adverbials_std = ([1e-6], [1.0])

# ---------------------------------------------------------------------------
# Core optimisation helper
# ---------------------------------------------------------------------------

def optimize_leastSquares(
    values_vagueAdverbial: dict,
    initial_params: List[float],
    events_to_fit_naming: List[str],
    event_specific_function,
    adverbial_specific_function,
) -> np.ndarray:
    """Least‑squares fit for all events & adverbials.

    The *params* vector packs event parameters first, then adverbial means, then adverbial stds.
    """

    def residuals(params: np.ndarray, data: dict) -> np.ndarray:  # noqa: C901  (complexity ok here)
        resid: List[float] = []
        num_event_params = len(inspect.signature(event_specific_function).parameters) - 1

        # Slices into the flat *params* vector
        means_offset = len(events_to_fit_naming) * num_event_params
        stds_offset = means_offset + len(adverbials_initial_means)

        means = params[means_offset:stds_offset]
        stds = params[stds_offset:]

        for event, time_dict in data.items():
            # Which slice belongs to this *event*?
            for i, event_name in enumerate(events_to_fit_naming):
                if event_name in event:
                    event_params = params[i * num_event_params : (i + 1) * num_event_params]
                    break
            else:
                raise ValueError(f"Unrecognised event name in data: {event}")

            for adverbial, sub_dict in time_dict.items():
                # Find mean/std index for this adverbial label
                try:
                    adv_idx = list(adverbials_initial_means.keys()).index(adverbial)
                except ValueError as exc:
                    raise ValueError(f"Unrecognised adverbial label: {adverbial}") from exc

                mean = means[adv_idx]
                std = stds[adv_idx]

                for time_ago_str, votes in sub_dict.items():
                    t = float(time_ago_str)
                    t_arr = np.repeat(t, len(votes))
                    rel_y = event_specific_function(t_arr, *event_params)
                    mu = adverbial_specific_function(rel_y, mean, std)
                    resid.extend(mu - np.asarray(votes, dtype=float))

        return np.asarray(resid, dtype=float)

    num_event_params = len(inspect.signature(event_specific_function).parameters) - 1
    total_event_params = len(events_to_fit_naming) * num_event_params

    bounds = (
        fitting_bounds_events[0] * total_event_params
        + fitting_bounds_adverbials_mean[0] * len(adverbials_initial_means)
        + fitting_bounds_adverbials_std[0] * len(adverbials_initial_stds),
        fitting_bounds_events[1] * total_event_params
        + fitting_bounds_adverbials_mean[1] * len(adverbials_initial_means)
        + fitting_bounds_adverbials_std[1] * len(adverbials_initial_stds),
    )

    res = least_squares(
        residuals,
        initial_params,
        bounds=bounds,
        args=(values_vagueAdverbial,),
        verbose=1,
    )
    return res.x


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_packed(path: Path = RESULTS_FILE_PATH / "event_adverbials") -> Dict[str, dict]:
    path = path.with_suffix('.json')
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _pack_params(
    params: np.ndarray,
    events: List[str],
    event_specific_function,
    adverbials: List[str],
) -> dict:
    """Convert the flat *params* array into a labelled dictionary."""
    num_event_params = len(inspect.signature(event_specific_function).parameters) - 1
    idx = 0

    event_params: Dict[str, List[float]] = {}
    for ev in events:
        event_params[ev] = params[idx : idx + num_event_params].tolist()
        idx += num_event_params

    adverbial_means: Dict[str, float] = {}
    for adv in adverbials:
        adverbial_means[adv] = float(params[idx])
        idx += 1

    adverbial_stds: Dict[str, float] = {}
    for adv in adverbials:
        adverbial_stds[adv] = float(params[idx])
        idx += 1

    return {
        "event_params": event_params,
        "adverbial_means": adverbial_means,
        "adverbial_stds": adverbial_stds,
    }


def save_optimized_params(
    file_base: str,
    optimized_params: np.ndarray,
    events_to_fit_naming: List[str],
    event_specific_function,
    adverbial_names: List[str],
) -> None:
    """Write *.json* (labelled) & *.pkl* (raw) parameter files."""
    packed = _pack_params(
        optimized_params, events_to_fit_naming, event_specific_function, adverbial_names
    )

    path_base = Path(file_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)

    with (path_base.with_suffix(".json")).open("w", encoding="utf-8") as fh:
        json.dump(packed, fh, indent=2)

    with (path_base.with_suffix(".pkl")).open("wb") as fh:
        pickle.dump(optimized_params, fh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Fit the whole FuzzyLLI (event specific and adverbial specific function)
def fit_event_adverbials(events_to_fit_naming: List[str]) -> dict:
    """Fit all specified events & return labelled parameter dictionary."""
    initial_std_relation: List[float] = []
    all_event_values: Dict[str, dict] = {}

    if events_to_fit_naming is None:
        return

    for event_name in events_to_fit_naming:
        file_path = f"{DATA_DIR}/{event_name}/cleanedData_minutes.json"
        with open(file_path, "r", encoding="utf-8") as fh:
            values_vagueAdverbial = json.load(fh)

        max_time_ago = max(map(int, values_vagueAdverbial["long time ago"].keys()))
        initial_std_relation.append(max_time_ago / 5)

        # Re‑order vague adverbials to match *vagueAdverbials_naming_initialguessmeans*
        values_vagueAdverbial = {
            adv: values_vagueAdverbial[adv]
            for adv in adverbials_initial_means.keys()
        }
        all_event_values[event_name] = values_vagueAdverbial

    # Build initial parameter vector
    num_event_params = len(inspect.signature(event_specific_function).parameters) - 1
    initial_std_relation = [val for val in initial_std_relation for _ in range(num_event_params)]
    initial_means = list(adverbials_initial_means.values())
    initial_stds = list(adverbials_initial_stds.values())
    initial_params = initial_std_relation + initial_means + initial_stds

    # Fit
    optimized = optimize_leastSquares(
        all_event_values,
        initial_params,
        events_to_fit_naming,
        event_specific_function,
        adverbial_specific_function,
    )

    # Persist results (both raw & labelled)
    save_optimized_params(
        RESULTS_FILE_PATH/"event_adverbials",
        optimized,
        events_to_fit_naming,
        event_specific_function,
        list(adverbials_initial_means.keys()),
    )

    return _pack_params(
        optimized,
        events_to_fit_naming,
        event_specific_function,
        list(adverbials_initial_means.keys()),
    )


# Fit the event specific function using word embeddings and a regression model
def fit_event_specific_embeddings(events_to_fit, events_to_fit_nl):
    packed = _load_packed()
    values = []

    for event in events_to_fit:
        values.append(packed["event_params"][event])

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    X = model.encode(events_to_fit_nl)
    y_log = np.log1p(np.array(np.concatenate(values)))

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y_log)
    joblib.dump(ridge, RESULTS_FILE_PATH / 'event_embeddings_ridge.pkl')
    return ridge


def fit_event_specific_random_forest(events_to_fit):
    packed = _load_packed()
    values = []
    median_rows = []  # To store median rows per event

    for event in events_to_fit:
        values.append(packed["event_params"][event])

    y = np.array(np.concatenate(values))

    for event in events_to_fit:
        file_path = f"{DATA_DIR}/{event}/event_properties.json"
        with open(file_path, "r", encoding="utf-8") as fh:
            event_properties = json.load(fh)

        frequency_votes = [d['Frequency'] for d in event_properties if 'Frequency' in d]
        duration_votes = [d['Duration'] for d in event_properties if 'Duration' in d]

        # Because participants first could not select "Hours" as Duration I entered hours later and it became entry 6
        # Real entry 6 ("Decades") is 5
        if 6 in duration_votes:
            duration_votes = [
                1 if x == 6 else (x + 1 if x != 0 else x)
                for x in duration_votes
            ]

        median_frequency = statistics.median(frequency_votes)
        median_duration = statistics.median(duration_votes)

        median_rows.append({
            'Frequency': median_frequency,
            'Duration': median_duration
        })


    X = pd.DataFrame(median_rows)

    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    joblib.dump(model, RESULTS_FILE_PATH / 'event_random_forest.pkl')
