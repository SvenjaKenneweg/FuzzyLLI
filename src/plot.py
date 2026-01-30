# plot_event_adverbial_fits.py
"""Visualise the fitted event/adverbial membership functions.

This version plots **one figure**: it chooses the event *with the largest
fitted σ* **within the user‑supplied `events_to_plot` list only**, then
overlays all supplied events for comparison.  The script consumes the
labelled *JSON* parameters produced by `event_adverbial_fitting.py`.
"""
from __future__ import annotations

import inspect
import statistics
import json
from pathlib import Path
from typing import Callable, Dict, List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


import src.config as config
from src.predictions import predict_adverbial_random_forest, predict_adverbial_functions, predict_adverbial_embedding, predict_adverbial_fuzzylli


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

requires_properties = {
    predict_adverbial_random_forest,
    predict_adverbial_functions
}
config.PLOT_FILE_PATH.mkdir(parents=True, exist_ok=True)
Y_LIMS = (0.3, 1.02)  # <- fixed y‑axis for both sub‑plots

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_packed(path: Path = config.RESULTS_FILE_PATH/"event_adverbials.json") -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def _initial_x_axis(values_vague_adverbial: dict) -> np.ndarray:
    # max_minutes = max(map(float, values_vague_adverbial["far away"].keys()))
    return np.linspace(-2, 10, 400_000)

def _select_main_event(packed: Dict[str, dict], candidates: List[str]) -> str:
    """Return the candidate event with the largest σ (first/maximum param).

    Raises
    ------
    ValueError
        If none of *candidates* has fitted parameters.
    """
    best_event: str | None = None
    best_std: float = -np.inf

    for ev in candidates:
        params = packed["event_params"].get(ev)
        if params is None:
            continue  # skip events lacking a fit
        std_val = max(map(float, params))  # use max in case of multi‑σ model
        if std_val > best_std:
            best_std = std_val
            best_event = ev

    if best_event is None:
        raise ValueError("None of the provided events have fitted parameters.")
    return best_event


def normalize_event_label(ev: str) -> str:
    noun_to_action = {
        "shower": "Taking a Shower",
        "rent_payment": "Paying Rent",
        "birthday": "Birthday",
        "vacation": "Going on Vacation",
        "year_abroad": "Year Abroad",
        "wedding_celebration": "Celebrating Marriage",
        "winebottle_storage": "Storing Wine Bottle",
        "chatting_friend": "Chatting with Friend",
        "reading_book": "Reading Book",
    }

    # Determine person
    person = "$1^{st}$ p" if ev.startswith("own_") else "$3^{rd}$ p"

    # Remove any prefix like 'own_' or name prefix
    parts = ev.split("_")
    core = "_".join(parts[1:]) if parts[0] == "own" else "_".join(parts[1:])

    # Check if mapped to action
    core_lower = core.lower()
    label = noun_to_action.get(core_lower, " ".join(core.split("_")).title())

    return f"{label} ({person})"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_all_persons_event_adverbials(
    events_to_plot: List[str]
) -> None:
    """Plot one figure: main event = highest σ among *events_to_plot*.

    Parameters
    ----------
    events_to_plot
        List of event names to consider & overlay.  The main plot is chosen
        **only** from this list.
    """

    packed = _load_packed()
    main_event = _select_main_event(packed, events_to_plot)

    # Use x‑axis sized to the main event's datasets
    json_path = f"{config.DATASET_SPATIAL_PATH}/{main_event}.json"
    with open(json_path, "r", encoding="utf-8") as fh:
        values_vague_adverbial = json.load(fh)
    x1 = _initial_x_axis(values_vague_adverbial)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colour_cycle = iter(colours)

    sorted_events = sorted(
        [ev for ev in events_to_plot if packed["event_params"].get(ev) is not None],
        key=lambda ev: packed["event_params"][ev][0]  # assuming σₑ = first param
    )
    # Overlay each user‑requested event in sorted order
    for ev in sorted_events:
        event_label = ev #normalize_event_label(ev)
        params = packed["event_params"][ev]
        y_ev = config.event_specific_function(x1, *params)
        ax_left.plot(
            x1,
            y_ev,
            label=event_label,
            color=next(colour_cycle, None),
        )

    # Left‑axis styling
    ax_left.set(
        ylim = Y_LIMS,
        xlabel="Time units ago in minutes",
        ylabel="Beforeness of event",
        title="Event specific functions $\Phi_e$",
    )
    ax_left.grid(True)
    ax_left.legend(loc="lower right")

    # Right plot: adverbial curves for *main_event* only
    main_params = packed["event_params"][main_event]
    y_relation = config.event_specific_function(x1, *main_params)

    for i, adv in enumerate(config.VAGUE_ADVERBIALS):
        mu = packed["adverbial_means"][adv]
        sigma = packed["adverbial_stds"][adv]
        x_norm = np.linspace(y_relation.min(), y_relation.max(), 1_000)
        y_norm = config.adverbial_specific_function(x_norm, mu, sigma)
        ax_right.plot(
            y_norm,
            x_norm,
            # label=f"{adv.title()}\n$\\mu_e={mu:.2f}$, $\\sigma_a={sigma:.2f}$",
            label=f"{adv.title()}",
            color=colours[i % len(colours)],
        )

    ax_right.set(
        ylim=Y_LIMS,
        xlabel="Fuzzy Membership Value of Adverbial",
        ylabel="Beforeness of Event",
        title="Adverbial specific functions $\mu_a$",
    )
    ax_right.grid(True)
    ax_right.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    outfile = config.PLOT_FILE_PATH / "fuzzylli.png"
    fig.savefig(outfile, dpi=300)
    plt.close()


def plot_events_adverbials_fitted(event_name, adverbial):
    plt.figure(figsize=(10, 6))

    # Iterate over both `events` and `event_name_nl` simultaneously
    with open(f"{config.DATASET_SPATIAL_PATH}/{event_name}.json", "r", encoding="utf-8") as f:
        cleaned_data = json.load(f)[adverbial]

    # Calculate medians for this event
    real_data_values = {key: float(np.median(values)) for key, values in cleaned_data.items()}
    # Convert keys to int and sort by minutes
    minutes = sorted(float(k) for k in real_data_values.keys())
    time_unit = minutes #sorted(int(k)/43800 for k in real_data_values.keys())
    values = [real_data_values[str(m)] for m in minutes]

    event_label = event_name
    plt.plot(time_unit, values, marker='o', label=f'Survey: {event_label}; Adverbial: {adverbial}')

    # Generate prediction for a denser range of time_unit
    max_time = max(time_unit)
    dense_minutes = np.arange(0, max_time + 10, max_time / 50)

    predicted_values = []
    for minute in dense_minutes:
        prob_adverbial = predict_adverbial_fuzzylli(event_name, int(minute))
        predicted_values.append(prob_adverbial[adverbial])

    # Plot the predicted data
    plt.plot(dense_minutes, predicted_values, label=f'Fit')

    plt.ylim(0, 1.1)
    plt.xlabel('Distance in Pixel')
    plt.ylabel(f'Membership Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return
