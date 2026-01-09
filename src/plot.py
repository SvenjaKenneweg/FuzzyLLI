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
from src.predictions import predict_adverbial_random_forest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
config.PLOT_FILE_PATH.mkdir(parents=True, exist_ok=True)
Y_LIMS = (0.3, 1.02)  # <- fixed y‑axis for both sub‑plots

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_packed(path: Path = config.RESULTS_JSON) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _initial_x_axis(values_vague_adverbial: dict) -> np.ndarray:
    max_minutes = max(map(int, values_vague_adverbial["long time ago"].keys()))
    return np.linspace(-(max_minutes / 2), max_minutes * 1.6, 400_000)


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
    json_path = config.DATA_DIR / main_event / "cleanedData_minutes.json"
    with json_path.open("r", encoding="utf-8") as fh:
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
        params = packed["event_params"][ev]
        y_ev = config.event_specific_function(x1, *params)
        ax_left.plot(
            x1,
            y_ev,
            # label=f"{ev.title()}",
            label=f"{ev.replace('tom', 'Friend').replace('_', ' ').title()}",
            # label=f"{ev.replace('_', ' ').title()}\n$\\sigma_e={', '.join(f'{p:.0f}' for p in params)}$",
            color=next(colour_cycle, None),
        )

    # Left‑axis styling
    ax_left.set(
        ylim = Y_LIMS,
        xlabel="Time units ago in minutes",
        ylabel="Beforeness of event",
        title="Event specific functions $P_{e}$",
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

# def plot_events_adverbials(events, adverbial):
#     medians_per_event = []
#     for event_name in events:
#         with open(config.DATA_DIR / event_name / "cleanedData_minutes.json", "r", encoding="utf-8") as f:
#             cleaned_data = json.load(f)[adverbial]
#         medians = {key: float(np.median(values)) for key, values in cleaned_data.items()}
#         medians_per_event.append({
#             "event": event_name,
#             "medians": medians
#         })
#
#     plt.figure(figsize=(10, 6))
#
#     for event in medians_per_event:
#         event_name = event['event']
#         medians_dict = event['medians']
#
#         # Convert keys to int and sort by minutes
#         minutes = sorted(int(k) for k in medians_dict.keys())
#         values = [medians_dict[str(m)] for m in minutes]
#
#         plt.plot(minutes, values, marker='o', label=event_name)
#
#     plt.xlabel('Minutes ago')
#     plt.ylabel('Membership value (median) for the adverbial ' + str(adverbial))
#     plt.legend()
#     plt.grid(False)
#     plt.tight_layout()
#     plt.show()


def plot_events_adverbials_fitted(events, events_nl, adverbial, predict_fn=None):
    if len(events) != len(events_nl):
        raise ValueError("The lengths of events and event_name_nl must be the same.")

    plt.figure(figsize=(10, 6))

    # Iterate over both `events` and `event_name_nl` simultaneously
    for event_name, event_name_nl in zip(events, events_nl):
        with open(config.DATA_DIR / event_name / "cleanedData_minutes.json", "r", encoding="utf-8") as f:
            cleaned_data = json.load(f)[adverbial]

        # Calculate medians for this event
        real_data_values = {key: float(np.median(values)) for key, values in cleaned_data.items()}
        # Convert keys to int and sort by minutes
        minutes = sorted(int(k) for k in real_data_values.keys())
        values = [real_data_values[str(m)] for m in minutes]

        if predict_fn == predict_adverbial_random_forest:
            file_path = f"{config.DATA_DIR}/event_properties.json"
            with open(file_path, "r", encoding="utf-8") as fh:
                event_properties = json.load(fh)

            properties = pd.DataFrame([{
                'Frequency': event_properties[event_name_nl.replace("Tom", "A friend").replace("You", "I")][
                    "Frequency"],
                'Richness': event_properties[event_name_nl.replace("Tom", "A friend").replace("You", "I")][
                    "Richness"],
                'Importance': event_properties[event_name_nl.replace("Tom", "A friend").replace("You", "I")][
                    "Importance"]
            }])

            # Generate prediction for a denser range of minutes
            max_time = max(minutes)
            dense_minutes = np.arange(0, max_time + 10, max_time/100)  # More granular minute values

            # Predict for the denser time range
            predicted_values = []
            for minute in dense_minutes:
                prob_adverbial = predict_fn(properties, int(minute))
                predicted_values.append(prob_adverbial[adverbial])

            # Plot the real and predicted data
            real_line = plt.plot(minutes, values, marker='o', label=f'{event_name} (real)', linestyle='-')
            real_color = real_line[0].get_color()  # Get the color of the real line
            plt.plot(dense_minutes, predicted_values, label=f'{event_name} (predicted)', linestyle='--', color=real_color)
        if not predict_fn:
            plt.plot(minutes, values, marker='o', label=f'{event_name}', linestyle='-')

    plt.ylim(0, 1)
    plt.xlabel('Minutes ago')
    plt.ylabel(f'Membership value (median) for the adverbial {adverbial}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return
