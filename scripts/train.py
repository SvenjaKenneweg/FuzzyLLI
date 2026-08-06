from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import least_squares

from .config import event_specific_function, adverbial_specific_function
from . import config

# ========================
# Parameters for the fitting
# ========================

# Initial guesses for vague adverbials' mean values
adverbials_initial_means: Dict[str, float] = {
    "close to": 0.5,
    "moderately far": 0.7,
    "far away": 0.9,
}

# Standard deviation initial guesses for adverbials
adverbials_initial_stds: Dict[str, float] = {
    "close to": 0.1,
    "moderately far": 0.1,
    "far away": 0.1,
}

# Bounds for the fitting (lower, upper)
fitting_bounds_events = ([10], [500])
fitting_bounds_adverbials_mean = ([0.45], [1.0])
fitting_bounds_adverbials_std = ([1e-6], [0.5])

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

def _load_packed(path: Path = config.RESULTS_FILE_PATH / "event_adverbials") -> Dict[str, dict]:
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
def fit_object_adverbials(events_to_fit_naming: List[str]) -> dict:
    """Fit all specified events & return labelled parameter dictionary."""
    initial_std_relation: List[float] = []
    all_event_values: Dict[str, dict] = {}

    if events_to_fit_naming is None:
        return

    for survey_name in events_to_fit_naming:
        file_path = f"{config.DATASET_SPATIAL_PATH}/{survey_name}.json"
        with open(file_path, "r", encoding="utf-8") as fh:
            values_vagueAdverbial = json.load(fh)

        max_time_ago = max(map(float, values_vagueAdverbial["far away"].keys()))
        initial_std_relation.append(max_time_ago)

        # Re‑order vague adverbials to match *vagueAdverbials_naming_initialguessmeans*
        values_vagueAdverbial = {
            adv: values_vagueAdverbial[adv]
            for adv in adverbials_initial_means.keys()
        }
        all_event_values[survey_name] = values_vagueAdverbial

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
        config.RESULTS_FILE_PATH/"event_adverbials",
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
