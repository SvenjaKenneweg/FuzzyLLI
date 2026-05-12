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


from .predictions import predict_adverbial_random_forest, predict_adverbial_functions, predict_adverbial_embedding
from .evaluation_test_dataset import get_percentages
from . import config


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
        event_label = normalize_event_label(ev)
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

def plot_events_adverbials_fitted(events, events_nl, adverbials, predict_functions=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.figure(figsize=(10, 6))

    if isinstance(adverbials, str):
        adverbials = [adverbials]

    predict_functions = predict_functions or []
    line_styles = ['--', ':', '-.']

    # One stable color per adverbial
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    if not color_cycle:
        color_cycle = ["C0", "C1", "C2", "C3", "C4"]

    adverbial_colors = {
        adv: color_cycle[i % len(color_cycle)]
        for i, adv in enumerate(adverbials)
    }

    def predict_label(predict_fn):
        name = predict_fn.__name__.lower()
        if "random" in name:
            return "Random Forest"
        elif "embedding" in name:
            return "Text Embeddings"
        elif "moc" in name:
            return "MOC"
        elif "dnn" in name:
            return "DNN"
        else:
            return "Power Law"

    def normalize_property_key(event_name_nl):
        return event_name_nl.replace("Tom", "A friend").replace("You", "I")

    def load_event_properties_for_train():
        with open(config.DATA_DIR / "event_properties.json", "r", encoding="utf-8") as fh:
            return json.load(fh)

    def load_event_properties_for_test():
        event_properties = {}

        with open(f"{config.DATASET_TEST_PATH}/event_properties_1.json", "r", encoding="utf-8") as fh:
            event_properties.update(json.load(fh))

        with open(f"{config.DATASET_TEST_PATH}/event_properties_2.json", "r", encoding="utf-8") as fh:
            event_properties.update(json.load(fh))

        return event_properties

    def get_properties(event_name_nl, event_properties):
        prop_key = normalize_property_key(event_name_nl)

        if prop_key not in event_properties:
            raise KeyError(
                f"No properties found for event '{event_name_nl}' "
                f"using normalized key '{prop_key}'."
            )

        prop_dict = event_properties[prop_key]
        return [{prop: prop_dict[prop] for prop in config.properties_to_use}]

    def add_structured_legend():
        ax = plt.gca()

        adverbial_handles = [
            Patch(
                facecolor=adverbial_colors[adv],
                alpha=0.55,
                label=adv
            )
            for adv in adverbials
        ]

        element_handles = [
            Patch(
                facecolor="gray",
                alpha=0.35,
                label="GT Top-1 Adverbial"
            )
        ]

        for i, predict_fn in enumerate(predict_functions):
            element_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=line_styles[i % len(line_styles)],
                    label=f"Fit: {predict_label(predict_fn)}"
                )
            )

        legend1 = ax.legend(
            handles=adverbial_handles,
            title="Adverbial color",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0
        )

        ax.add_artist(legend1)

        ax.legend(
            handles=element_handles,
            title="Plot element",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.72),
            borderaxespad=0
        )

    def plot_one_event(event_name_nl, adverbial_to_minutes_values, event_properties=None):
        def plot_gt_winner_ribbon(highest_adverbial_by_minute):
            winner_minutes = sorted(highest_adverbial_by_minute.keys())

            if not winner_minutes:
                return

            if len(winner_minutes) == 1:
                minute = winner_minutes[0]
                spans = [
                    (
                        max(0, minute - 1),
                        minute + 1,
                        highest_adverbial_by_minute[minute]
                    )
                ]
            else:
                boundaries = []

                first_width = winner_minutes[1] - winner_minutes[0]
                boundaries.append(max(0, winner_minutes[0] - first_width / 2))

                for left_minute, right_minute in zip(winner_minutes[:-1], winner_minutes[1:]):
                    boundaries.append((left_minute + right_minute) / 2)

                last_width = winner_minutes[-1] - winner_minutes[-2]
                boundaries.append(winner_minutes[-1] + last_width / 2)

                spans = [
                    (
                        boundaries[i],
                        boundaries[i + 1],
                        highest_adverbial_by_minute[winner_minutes[i]]
                    )
                    for i in range(len(winner_minutes))
                ]

            # Merge neighboring spans with the same winning adverbial
            merged_spans = []

            for left, right, adverbial in spans:
                if merged_spans and merged_spans[-1][2] == adverbial:
                    old_left, _, _ = merged_spans[-1]
                    merged_spans[-1] = (old_left, right, adverbial)
                else:
                    merged_spans.append((left, right, adverbial))

            # Draw a small GT winner ribbon at the bottom of the plot
            for left, right, adverbial in merged_spans:
                plt.axvspan(
                    left,
                    right,
                    ymin=0.0,
                    ymax=0.055,
                    color=adverbial_colors[adverbial],
                    alpha=0.35,
                    linewidth=0,
                    zorder=0,
                    label="_nolegend_"
                )

        # Collect all minutes that appear for any adverbial
        all_minutes = sorted({
            int(minute)
            for minutes_to_values in adverbial_to_minutes_values.values()
            for minute in minutes_to_values.keys()
        })

        # Find which adverbial has the highest GT value at each minute
        highest_adverbial_by_minute = {}

        for minute in all_minutes:
            candidates = {}

            for adv, minutes_to_values in adverbial_to_minutes_values.items():
                if minute in minutes_to_values:
                    candidates[adv] = minutes_to_values[minute]

            if candidates:
                highest_adverbial_by_minute[minute] = max(
                    candidates,
                    key=candidates.get
                )

        # Highlight GT winner as a bottom ribbon, not as points or a line
        plot_gt_winner_ribbon(highest_adverbial_by_minute)

        # Plot fitted prediction curves only
        for adverbial, minutes_to_values in adverbial_to_minutes_values.items():
            if not minutes_to_values:
                continue

            minutes = sorted(int(m) for m in minutes_to_values.keys())
            color = adverbial_colors[adverbial]

            if not predict_functions:
                continue

            max_time = max(minutes)

            if max_time == 0:
                dense_minutes = np.array([0])
            else:
                dense_minutes = np.linspace(0, max_time, 51)

            for i, predict_fn in enumerate(predict_functions):
                predicted_values = []

                if predict_fn in requires_properties:
                    if event_properties is None:
                        raise ValueError(
                            f"{predict_fn.__name__} requires event properties, "
                            "but no event_properties were provided."
                        )

                    properties = get_properties(event_name_nl, event_properties)

                    for minute in dense_minutes:
                        prob_adverbial = predict_fn(properties, int(minute))
                        predicted_values.append(prob_adverbial.get(adverbial, 0.0))

                else:
                    for minute in dense_minutes:
                        prob_adverbial = predict_fn(event_name_nl, int(minute))
                        predicted_values.append(prob_adverbial.get(adverbial, 0.0))

                plt.plot(
                    dense_minutes,
                    predicted_values,
                    label="_nolegend_",
                    linestyle=line_styles[i % len(line_styles)],
                    color=color,
                    zorder=2
                )

    if len(events_nl) > len(events):
        all_data = get_percentages()
        event_properties = None

        if any(fn in requires_properties for fn in predict_functions):
            event_properties = load_event_properties_for_test()

        for event_name_nl in events_nl:
            adverbial_to_minutes_values = {
                adv: {}
                for adv in adverbials
            }

            for (event_name, time), counts in all_data.items():
                if event_name != event_name_nl:
                    continue

                total = sum(counts.values())

                for adv in adverbials:
                    if total == 0:
                        value = 0.0
                    else:
                        value = counts.get(adv, 0) / total

                    adverbial_to_minutes_values[adv][int(time)] = value

            if not any(adverbial_to_minutes_values.values()):
                print(f"No data found for event: {event_name_nl}")
                continue

            plot_one_event(
                event_name_nl=event_name_nl,
                adverbial_to_minutes_values=adverbial_to_minutes_values,
                event_properties=event_properties
            )

    else:
        event_properties = None

        if any(fn in requires_properties for fn in predict_functions):
            event_properties = load_event_properties_for_train()

        for event_name, event_name_nl in zip(events, events_nl):
            with open(
                config.DATA_DIR / event_name / "cleanedData_minutes.json",
                "r",
                encoding="utf-8"
            ) as f:
                cleaned_data_all = json.load(f)

            adverbial_to_minutes_values = {}

            for adv in adverbials:
                if adv not in cleaned_data_all:
                    print(f"No cleaned data for adverbial '{adv}' in event '{event_name}'")
                    continue

                cleaned_data = cleaned_data_all[adv]

                adverbial_to_minutes_values[adv] = {
                    int(key): float(np.mean(values))
                    for key, values in cleaned_data.items()
                }

            if not adverbial_to_minutes_values:
                continue

            plot_one_event(
                event_name_nl=event_name_nl,
                adverbial_to_minutes_values=adverbial_to_minutes_values,
                event_properties=event_properties
            )

    plt.ylim(0, 1)

    # Log-like scale that still supports x = 0
    plt.xscale("symlog", linthresh=60)
    plt.xticks(
        [0, 60, 1440, 10080, 20160, 43200],
        ["0", "1h", "1d", "1w", "2w", "1mo"]
    )
    plt.xlim(left=0)

    plt.xlabel("Time ago")
    plt.ylabel("Membership Value")

    add_structured_legend()

    plt.grid(True, which="both")
    plt.tight_layout()

    outfile = config.PLOT_FILE_PATH / "fitted_curve.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# def plot_events_adverbials_fitted(events, events_nl, adverbials, predict_functions=None):
#     plt.figure(figsize=(10, 6))
#
#     if isinstance(adverbials, str):
#         adverbials = [adverbials]
#
#     predict_functions = predict_functions or []
#     line_styles = ['--', ':', '-.']
#
#     # One stable color per adverbial
#     color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
#     adverbial_colors = {
#         adv: color_cycle[i % len(color_cycle)]
#         for i, adv in enumerate(adverbials)
#     }
#
#     def predict_label(predict_fn):
#         name = predict_fn.__name__.lower()
#         if "random" in name:
#             return "Random Forest"
#         elif "embedding" in name:
#             return "Word Embeddings"
#         elif "moc" in name:
#             return "MOC"
#         elif "dnn" in name:
#             return "DNN"
#         else:
#             return "Power Law"
#
#     def normalize_property_key(event_name_nl):
#         return event_name_nl.replace("Tom", "A friend").replace("You", "I")
#
#     def load_event_properties_for_train():
#         with open(config.DATA_DIR / "event_properties.json", "r", encoding="utf-8") as fh:
#             return json.load(fh)
#
#     def load_event_properties_for_test():
#         event_properties = {}
#
#         with open(f"{config.DATASET_TEST_PATH}/event_properties_1.json", "r", encoding="utf-8") as fh:
#             event_properties.update(json.load(fh))
#
#         with open(f"{config.DATASET_TEST_PATH}/event_properties_2.json", "r", encoding="utf-8") as fh:
#             event_properties.update(json.load(fh))
#
#         return event_properties
#
#     def get_properties(event_name_nl, event_properties):
#         prop_key = normalize_property_key(event_name_nl)
#
#         if prop_key not in event_properties:
#             raise KeyError(
#                 f"No properties found for event '{event_name_nl}' "
#                 f"using normalized key '{prop_key}'."
#             )
#
#         prop_dict = event_properties[prop_key]
#         return [{prop: prop_dict[prop] for prop in config.properties_to_use}]
#
#     def plot_one_event(event_name_nl, adverbial_to_minutes_values, event_properties=None):
#         # Collect all minutes that appear for any adverbial
#         all_minutes = sorted({
#             int(minute)
#             for minutes_to_values in adverbial_to_minutes_values.values()
#             for minute in minutes_to_values.keys()
#         })
#
#         # Find which adverbial has the highest GT value at each minute
#         highest_adverbial_by_minute = {}
#
#         for minute in all_minutes:
#             candidates = {}
#
#             for adv, minutes_to_values in adverbial_to_minutes_values.items():
#                 if minute in minutes_to_values:
#                     candidates[adv] = minutes_to_values[minute]
#
#             if candidates:
#                 highest_adverbial_by_minute[minute] = max(
#                     candidates,
#                     key=candidates.get
#                 )
#
#         for adverbial, minutes_to_values in adverbial_to_minutes_values.items():
#             if not minutes_to_values:
#                 continue
#
#             minutes = sorted(int(m) for m in minutes_to_values.keys())
#             values = [minutes_to_values[m] for m in minutes]
#
#             color = adverbial_colors[adverbial]
#
#             # Only highlight GT points where this adverbial is the highest
#             highlighted_minutes = [
#                 minute
#                 for minute in minutes
#                 if highest_adverbial_by_minute.get(minute) == adverbial
#             ]
#
#             highlighted_values = [
#                 minutes_to_values[minute]
#                 for minute in highlighted_minutes
#             ]
#
#             if highlighted_minutes:
#                 plt.scatter(
#                     highlighted_minutes,
#                     highlighted_values,
#                     label=f'GT highest: {adverbial}',
#                     color=color,
#                     s=80,
#                     marker='o',
#                     edgecolors='black',
#                     zorder=5
#                 )
#
#             if not predict_functions:
#                 continue
#
#             max_time = max(minutes)
#
#             if max_time == 0:
#                 dense_minutes = np.array([0])
#             else:
#                 dense_minutes = np.linspace(0, max_time, 51)
#
#             for i, predict_fn in enumerate(predict_functions):
#                 predicted_values = []
#
#                 if predict_fn in requires_properties:
#                     if event_properties is None:
#                         raise ValueError(
#                             f"{predict_fn.__name__} requires event properties, "
#                             "but no event_properties were provided."
#                         )
#
#                     properties = get_properties(event_name_nl, event_properties)
#
#                     for minute in dense_minutes:
#                         prob_adverbial = predict_fn(properties, int(minute))
#                         predicted_values.append(prob_adverbial.get(adverbial, 0.0))
#
#                 else:
#                     for minute in dense_minutes:
#                         prob_adverbial = predict_fn(event_name_nl, int(minute))
#                         predicted_values.append(prob_adverbial.get(adverbial, 0.0))
#
#                 plt.plot(
#                     dense_minutes,
#                     predicted_values,
#                     label=f'Fit: {predict_label(predict_fn)}; Adverbial: {adverbial}',
#                     linestyle=line_styles[i % len(line_styles)],
#                     color=color
#                 )
#
#     if len(events_nl) > len(events):
#         all_data = get_percentages()
#         event_properties = None
#
#         if any(fn in requires_properties for fn in predict_functions):
#             event_properties = load_event_properties_for_test()
#
#         for event_name_nl in events_nl:
#             adverbial_to_minutes_values = {
#                 adv: {}
#                 for adv in adverbials
#             }
#
#             for (event_name, time), counts in all_data.items():
#                 if event_name != event_name_nl:
#                     continue
#
#                 total = sum(counts.values())
#
#                 for adv in adverbials:
#                     if total == 0:
#                         value = 0.0
#                     else:
#                         value = counts.get(adv, 0) / total
#
#                     adverbial_to_minutes_values[adv][int(time)] = value
#
#             if not any(adverbial_to_minutes_values.values()):
#                 print(f"No data found for event: {event_name_nl}")
#                 continue
#
#             plot_one_event(
#                 event_name_nl=event_name_nl,
#                 adverbial_to_minutes_values=adverbial_to_minutes_values,
#                 event_properties=event_properties
#             )
#
#     else:
#         event_properties = None
#
#         if any(fn in requires_properties for fn in predict_functions):
#             event_properties = load_event_properties_for_train()
#
#         for event_name, event_name_nl in zip(events, events_nl):
#             with open(
#                 config.DATA_DIR / event_name / "cleanedData_minutes.json",
#                 "r",
#                 encoding="utf-8"
#             ) as f:
#                 cleaned_data_all = json.load(f)
#
#             adverbial_to_minutes_values = {}
#
#             for adv in adverbials:
#                 if adv not in cleaned_data_all:
#                     print(f"No cleaned data for adverbial '{adv}' in event '{event_name}'")
#                     continue
#
#                 cleaned_data = cleaned_data_all[adv]
#
#                 adverbial_to_minutes_values[adv] = {
#                     int(key): float(np.mean(values))
#                     for key, values in cleaned_data.items()
#                 }
#
#             if not adverbial_to_minutes_values:
#                 continue
#
#             plot_one_event(
#                 event_name_nl=event_name_nl,
#                 adverbial_to_minutes_values=adverbial_to_minutes_values,
#                 event_properties=event_properties
#             )
#
#     plt.ylim(0, 1)
#     plt.xscale("symlog", linthresh=60)
#     plt.xlim(left=0)
#
#     plt.xticks(
#         [0, 60, 1440, 10080, 20160, 43200],
#         ["0", "1h", "1d", "1w", "2w", "1mo"]
#     )
#
#     plt.xlabel('Time ago')
#     plt.ylabel('Membership Value')
#     plt.legend()
#     plt.grid(True, which="both")
#     plt.tight_layout()
#
#     outfile = config.PLOT_FILE_PATH / "fitted_curve.png"
#     plt.savefig(outfile, dpi=300)
#     plt.close()


# def plot_events_adverbials_fitted(events, events_nl, adverbials, predict_functions=None):
#     plt.figure(figsize=(10, 6))
#
#     if isinstance(adverbials, str):
#         adverbials = [adverbials]
#
#     predict_functions = predict_functions or []
#     line_styles = ['--', ':', '-.']
#
#     # One stable color per adverbial
#     color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
#     adverbial_colors = {
#         adv: color_cycle[i % len(color_cycle)]
#         for i, adv in enumerate(adverbials)
#     }
#
#     def predict_label(predict_fn):
#         name = predict_fn.__name__.lower()
#         if "random" in name:
#             return "Random Forest"
#         elif "embedding" in name:
#             return "Word Embeddings"
#         elif "moc" in name:
#             return "MOC"
#         elif "dnn" in name:
#             return "DNN"
#         else:
#             return "Power Law"
#
#     def normalize_property_key(event_name_nl):
#         return event_name_nl.replace("Tom", "A friend").replace("You", "I")
#
#     def load_event_properties_for_train():
#         with open(config.DATA_DIR / "event_properties.json", "r", encoding="utf-8") as fh:
#             return json.load(fh)
#
#     def load_event_properties_for_test():
#         event_properties = {}
#
#         with open(f"{config.DATASET_TEST_PATH}/event_properties_1.json", "r", encoding="utf-8") as fh:
#             event_properties.update(json.load(fh))
#
#         with open(f"{config.DATASET_TEST_PATH}/event_properties_2.json", "r", encoding="utf-8") as fh:
#             event_properties.update(json.load(fh))
#
#         return event_properties
#
#     def get_properties(event_name_nl, event_properties):
#         prop_key = normalize_property_key(event_name_nl)
#
#         if prop_key not in event_properties:
#             raise KeyError(
#                 f"No properties found for event '{event_name_nl}' "
#                 f"using normalized key '{prop_key}'."
#             )
#
#         prop_dict = event_properties[prop_key]
#         return [{prop: prop_dict[prop] for prop in config.properties_to_use}]
#
#     def plot_one_event(event_name_nl, adverbial_to_minutes_values, event_properties=None):
#         for adverbial, minutes_to_values in adverbial_to_minutes_values.items():
#             if not minutes_to_values:
#                 continue
#
#             minutes = sorted(int(m) for m in minutes_to_values.keys())
#             values = [minutes_to_values[m] for m in minutes]
#
#             color = adverbial_colors[adverbial]
#
#             plt.plot(
#                 minutes,
#                 values,
#                 marker='o',
#                 label=f'Event: {event_name_nl}; Adverbial: {adverbial}',
#                 linestyle='-',
#                 color=color
#             )
#
#             if not predict_functions:
#                 continue
#
#             max_time = max(minutes)
#
#             if max_time == 0:
#                 dense_minutes = np.array([0])
#             else:
#                 dense_minutes = np.linspace(0, max_time, 51)
#
#             for i, predict_fn in enumerate(predict_functions):
#                 predicted_values = []
#
#                 if predict_fn in requires_properties:
#                     if event_properties is None:
#                         raise ValueError(
#                             f"{predict_fn.__name__} requires event properties, "
#                             "but no event_properties were provided."
#                         )
#
#                     properties = get_properties(event_name_nl, event_properties)
#
#                     for minute in dense_minutes:
#                         prob_adverbial = predict_fn(properties, int(minute))
#                         predicted_values.append(prob_adverbial.get(adverbial, 0.0))
#
#                 else:
#                     for minute in dense_minutes:
#                         prob_adverbial = predict_fn(event_name_nl, int(minute))
#                         predicted_values.append(prob_adverbial.get(adverbial, 0.0))
#
#                 plt.plot(
#                     dense_minutes,
#                     predicted_values,
#                     label=f'Fit: {predict_label(predict_fn)}; Adverbial: {adverbial}',
#                     linestyle=line_styles[i % len(line_styles)],
#                     color=color
#                 )
#
#     if len(events_nl) > len(events):
#         all_data = get_percentages()
#         event_properties = None
#
#         if any(fn in requires_properties for fn in predict_functions):
#             event_properties = load_event_properties_for_test()
#
#         for event_name_nl in events_nl:
#             adverbial_to_minutes_values = {
#                 adv: {}
#                 for adv in adverbials
#             }
#
#             for (event_name, time), counts in all_data.items():
#                 if event_name != event_name_nl:
#                     continue
#
#                 total = sum(counts.values())
#
#                 for adv in adverbials:
#                     if total == 0:
#                         value = 0.0
#                     else:
#                         value = counts.get(adv, 0) / total
#
#                     adverbial_to_minutes_values[adv][int(time)] = value
#
#             if not any(adverbial_to_minutes_values.values()):
#                 print(f"No data found for event: {event_name_nl}")
#                 continue
#
#             plot_one_event(
#                 event_name_nl=event_name_nl,
#                 adverbial_to_minutes_values=adverbial_to_minutes_values,
#                 event_properties=event_properties
#             )
#
#     else:
#         event_properties = None
#
#         if any(fn in requires_properties for fn in predict_functions):
#             event_properties = load_event_properties_for_train()
#
#         for event_name, event_name_nl in zip(events, events_nl):
#             with open(
#                 config.DATA_DIR / event_name / "cleanedData_minutes.json",
#                 "r",
#                 encoding="utf-8"
#             ) as f:
#                 cleaned_data_all = json.load(f)
#
#             adverbial_to_minutes_values = {}
#
#             for adv in adverbials:
#                 if adv not in cleaned_data_all:
#                     print(f"No cleaned data for adverbial '{adv}' in event '{event_name}'")
#                     continue
#
#                 cleaned_data = cleaned_data_all[adv]
#
#                 adverbial_to_minutes_values[adv] = {
#                     int(key): float(np.mean(values))
#                     for key, values in cleaned_data.items()
#                 }
#
#             if not adverbial_to_minutes_values:
#                 continue
#
#             plot_one_event(
#                 event_name_nl=event_name_nl,
#                 adverbial_to_minutes_values=adverbial_to_minutes_values,
#                 event_properties=event_properties
#             )
#
#     plt.ylim(0, 1)
#     plt.xlabel('Time Units ago in Minutes')
#     plt.ylabel('Membership Value')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#
#     outfile = config.PLOT_FILE_PATH / "fitted_curve.png"
#     plt.savefig(outfile, dpi=300)
#     plt.close()
