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
import numpy as np

from src.config import (VAGUE_ADVERBIALS,
                        PLOT_FILE_PATH,
                        DURATION_ORDER,
                        FREQUENCY_ORDER,
                        RESULTS_JSON,
                        DATA_DIR,
                        event_specific_function,
                        adverbial_specific_function)

from src.predictions import (
    predict_adverbial_embedding,
    predict_adverbial_gpt_random_forest,
    predict_adverbial_random_forest
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PLOT_FILE_PATH.mkdir(parents=True, exist_ok=True)
Y_LIMS = (0.3, 1.02)  # <- fixed y‑axis for both sub‑plots

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_packed(path: Path = RESULTS_JSON) -> Dict[str, dict]:
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

    # Use x‑axis sized to the main event's data
    json_path = DATA_DIR / main_event / "cleanedData_minutes.json"
    with json_path.open("r", encoding="utf-8") as fh:
        values_vague_adverbial = json.load(fh)
    x1 = _initial_x_axis(values_vague_adverbial)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colour_cycle = iter(colours)

    # Overlay each user‑requested event
    for ev in events_to_plot:
        params = packed["event_params"].get(ev)
        if params is None:
            continue  # skip events without fits
        y_ev = event_specific_function(x1, *params)
        ax_left.plot(
            x1,
            y_ev,
            label=f"{ev.replace('_', ' ').title()}\n$\\sigma_e={', '.join(f'{p:.0f}' for p in params)}$",
            color=next(colour_cycle, None),
        )

    # Left‑axis styling
    ax_left.set(
        ylim = Y_LIMS,
        xlabel="Time units ago in minutes",
        ylabel="Beforeness of event",
        title="Event specific functions $P_{Ev}$",
    )
    ax_left.grid(True)
    ax_left.legend(loc="lower right")

    # Right plot: adverbial curves for *main_event* only
    main_params = packed["event_params"][main_event]
    y_relation = event_specific_function(x1, *main_params)

    for i, adv in enumerate(VAGUE_ADVERBIALS):
        mu = packed["adverbial_means"][adv]
        sigma = packed["adverbial_stds"][adv]
        x_norm = np.linspace(y_relation.min(), y_relation.max(), 1_000)
        y_norm = adverbial_specific_function(x_norm, mu, sigma)
        ax_right.plot(
            y_norm,
            x_norm,
            label=f"{adv.title()}\n$\\mu_e={mu:.2f}$, $\\sigma_a={sigma:.2f}$",
            color=colours[i % len(colours)],
        )

    ax_right.set(
        ylim=Y_LIMS,
        xlabel="Probability of adverbial",
        ylabel="Beforeness of event",
        title="Adverbial specific functions $P_{Adv}$",
    )
    ax_right.grid(True)
    ax_right.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    outfile = PLOT_FILE_PATH / "highestStd_allAdverbials.png"
    fig.savefig(outfile, dpi=300)


def get_event_properties(event: str):
    with open(DATA_DIR / event / "event_properties.json", "r", encoding="utf-8") as f:
        return json.load(f)
def adjust_duration_votes(votes: List[int]) -> List[int]:
    """Correct for late addition of 'Hours' in unseen_events options."""
    return [
        1 if v == 6 else (v + 1 if v != 0 else v)
        for v in votes
    ]
def plot_single_events(event_name, predict_fn):
    with open(DATA_DIR / event_name / "cleanedData_minutes.json", "r", encoding="utf-8") as f:
        cleaned_data = json.load(f)

    x_limit_max = None
    x_limit_min = None
    results = {}

    # --- Collect everything once ---
    for adverbial, time_series in cleaned_data.items():
        times, medians = zip(*[
            (int(t), np.median(v)) for t, v in time_series.items()
        ])
        sorted_indices = np.argsort(times)
        times = np.array(times)[sorted_indices]
        medians = np.array(medians)[sorted_indices]

        if adverbial == "recently":
            x_limit_max = times[-1]
        elif "some" in adverbial:
            x_limit_min = times[0]

        predictions = []
        for minutes_ago in times:
            if predict_fn == predict_adverbial_random_forest:
                props = get_event_properties(event_name)
                freq_votes = [p['Frequency'] for p in props if 'Frequency' in p]
                dur_votes = [p['Duration'] for p in props if 'Duration' in p]

                if 6 in dur_votes:
                    dur_votes = adjust_duration_votes(dur_votes)

                freq = FREQUENCY_ORDER[int(statistics.median(freq_votes))]
                dur = DURATION_ORDER[int(statistics.median(dur_votes))]
                prob_adverbial = predict_fn(dur, freq, int(minutes_ago))
                predictions.append(prob_adverbial[adverbial])
        results[adverbial] = (times, medians, predictions)

    if x_limit_max is None or x_limit_min is None:
        raise ValueError("x_limit was not set – check that 'recently' is in cleaned_data")

    # --- Plot 1: Until x_limit ---
    plt.figure(figsize=(8, 5))
    for adverbial, (times, medians, predictions) in results.items():
        line, = plt.plot(times, medians, marker='o', label=adverbial)
        plt.plot(times, predictions, linestyle='--', color=line.get_color())

    plt.xlim(left=0, right=x_limit_max)
    plt.xlabel('Time ago in minutes')
    plt.ylabel('Median Value')
    plt.title(f'{event_name} with {predict_fn.__name__}')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    outfile = PLOT_FILE_PATH / event_name / "small_time_ago.png"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)

    # --- Plot 2: After x_limit ---
    plt.figure(figsize=(8, 5))
    for adverbial, (times, medians, predictions) in results.items():
        line, = plt.plot(times, medians, marker='o', label=adverbial)
        plt.plot(times, predictions, linestyle='--', color=line.get_color())

    plt.xlim(left=x_limit_min, right=max(max(t) for t, _, _ in results.values()))
    plt.xlabel('Time ago in minutes')
    plt.ylabel('Median Value')
    plt.title(f'{event_name} with {predict_fn.__name__}')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    outfile = PLOT_FILE_PATH / event_name / "big_time_ago.png"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    return

