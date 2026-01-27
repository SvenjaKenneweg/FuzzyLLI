from __future__ import annotations

import inspect
import json
import shap
import pickle
import joblib
import statistics
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sentence_transformers import SentenceTransformer

from src.config import event_specific_function, adverbial_specific_function
import src.config as config

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
    "close to": 0.05,
    "moderately far": 0.1,
    "far away": 0.02,
}

# Bounds for the fitting (lower, upper)
fitting_bounds_events = ([1e-6], [np.inf])
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
def fit_event_adverbials(events_to_fit_naming: List[str]) -> dict:
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
        initial_std_relation.append(max_time_ago / 5)

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


# Fit the event specific function using word embeddings and a regression model
def fit_event_specific_embeddings(events_to_fit, events_to_fit_nl, *args):
    packed = _load_packed()
    values = []

    for event in events_to_fit:
        values.append(packed["event_params"][event])

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    X = model.encode(events_to_fit_nl)
    y_log = np.log1p(np.array(np.concatenate(values)))

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y_log)
    joblib.dump(ridge, config.RESULTS_FILE_PATH / 'configuration_word_embeddings.pkl')
    return ridge


def fit_event_specific_random_forest(events_to_fit, events_to_fit_nl, inspect_properties=False, *args):
    packed = _load_packed()
    values = []
    properties = []  # To store properties per event

    for event in events_to_fit:
        values.append(packed["event_params"][event])
    y = np.array(np.concatenate(values))

    with open(f"{config.DATA_DIR}/event_properties.json", "r", encoding="utf-8") as fh:
        event_properties = json.load(fh)
    for event_nl in events_to_fit_nl:
        prop_dict = event_properties[event_nl.replace("Tom", "A friend")]
        properties.append({prop: prop_dict[prop] for prop in config.properties_to_use})
    X = pd.DataFrame(properties)

    model = RandomForestRegressor(n_estimators=8, max_depth=5, random_state=42)
    model.fit(X, y)

    if inspect_properties:
        importances = model.feature_importances_
        print("Feature Importances:")
        for name, value in zip(X.columns, importances):
            print(f"{name}: {value:.4f}")
        result = permutation_importance(model, X, y, n_repeats=30, random_state=42)
        print("\nPermutation Feature Importance:")
        for name, value in zip(X.columns, result.importances_mean):
            print(f"{name}: {value:.4f}")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_size=(12, 6))
        shap.dependence_plot("Richness", shap_values, X)
        shap.dependence_plot("Richness", shap_values, X, interaction_index="Frequency")

        PartialDependenceDisplay.from_estimator(model, X, ['Frequency', 'Richness', 'Importance'])
        plt.tight_layout()
        plt.show()

    joblib.dump(model, config.RESULTS_FILE_PATH / 'configuration_random_forest.pkl')


def fit_event_specific_functions(events_to_fit, events_to_fit_nl, function_to_fit, *args):
    packed = _load_packed()
    values = []
    properties = []  # To store properties per event

    for event in events_to_fit:
        values.append(packed["event_params"][event])
    y = np.array(np.concatenate(values))

    with open(f"{config.DATA_DIR}/event_properties.json", "r", encoding="utf-8") as fh:
        event_properties = json.load(fh)
    for event in events_to_fit_nl:
        prop_dict = event_properties[event.replace("Tom", "A friend")]
        properties.append({prop: prop_dict[prop] for prop in config.properties_to_use})

    X = pd.DataFrame(properties)
    X_tuple = tuple(X[prop].values for prop in config.properties_to_use)

    # Fit curve with initial guess depending on number of parameters
    p0 = [1.0] * (len(config.properties_to_use) + 1)  # one extra param for intercept
    p0[0] = 1000
    p, _ = curve_fit(function_to_fit, X_tuple, y, p0=p0, maxfev=20000)

    # print(config.properties_to_use)
    # print(p)
    # Save parameters with generic labels
    param_labels = "abcdefghijklmnopqrstuvwxyz"[:len(p)]
    joblib.dump({"params": dict(zip(param_labels, p))},
                f"{config.RESULTS_FILE_PATH}/{function_to_fit.__name__}.pkl")
