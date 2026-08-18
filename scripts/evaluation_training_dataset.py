import json
import statistics
from pathlib import Path
from typing import List
import math
import openai
import itertools
import os
from scipy.stats import kendalltau, wilcoxon, binomtest
from openai import OpenAI
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    ndcg_score,
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score
)

from .predictions import (
    predict_adverbial_embedding,
    predict_adverbial_random_forest,
    predict_adverbial_functions
)
from .train import (
    fit_event_adverbials,
    fit_event_specific_embeddings,
    fit_event_specific_random_forest,
    fit_event_specific_functions
)
from .simple_models_training import fit_classifier, fit_regression
from .simple_models_predictions import (
    predict_adverbial_classifier,
    predict_adverbial_regression
)

from . import config


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def adjust_richness_votes(votes: List[int]) -> List[int]:
    """Correct for late addition of 'Hours' in test_dataset options."""
    return [
        1 if v == 6 else (v + 1 if v != 0 else v)
        for v in votes
    ]

def get_cleaned_data(event: str):
    with open(config.DATA_DIR / event / "cleanedData_minutes.json", "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_error(predicted: float, true: float) -> float:
    return abs(predicted - true)

requires_properties = {
    predict_adverbial_random_forest,
    predict_adverbial_functions
}


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

# Calculation of the Mean Absolute Error for the given predict_fn against the median values for a specific time ago, adverbial, event
def run_MAE_MdSE_evaluation(events, fit_models_fn, predict_fn, events_nl=None, function_to_use=None):
    fit_event_adverbials(events)
    fit_models_fn(events, events_nl, function_to_use)

    all_abs_errors = []
    all_sq_errors = []

    # Error tracking per granularity
    errors_per_event_abs = {i: [] for i in range(len(events))}
    errors_per_event_sq = {i: [] for i in range(len(events))}

    errors_per_adverbial_abs = {adv: [] for adv in config.VAGUE_ADVERBIALS}
    errors_per_adverbial_sq = {adv: [] for adv in config.VAGUE_ADVERBIALS}

    for i, event in enumerate(events):
        cleaned_data = get_cleaned_data(event)

        # --- Step 1: Gather targets ---
        human_medians = {
            adv: {k: float(np.median(v)) for k, v in cleaned_data[adv].items()}
            for adv in config.VAGUE_ADVERBIALS if adv in cleaned_data
        }

        for adverbial, times in human_medians.items():
            for minutes_ago, target_value in times.items():
                # --- Step 2: Generate Prediction ---
                if predict_fn in requires_properties:
                    with open(f"{config.DATA_DIR}/event_properties.json", "r", encoding="utf-8") as fh:
                        event_properties = json.load(fh)

                    prop_dict = event_properties[events_nl[i].replace("Tom", "A friend")]
                    properties = [{prop: prop_dict[prop] for prop in config.properties_to_use}]
                    prediction = predict_fn(properties, int(minutes_ago), function_to_use)
                else:
                    input_data = events_nl[i] if events_nl else event
                    prediction = predict_fn(input_data, int(minutes_ago))

                # --- Step 3: Calculate Errors ---
                diff = prediction[adverbial] - target_value
                abs_error = abs(diff)
                sq_error = diff ** 2

                # Store errors
                all_abs_errors.append(abs_error)
                all_sq_errors.append(sq_error)

                errors_per_event_abs[i].append(abs_error)
                errors_per_event_sq[i].append(sq_error)

                errors_per_adverbial_abs[adverbial].append(abs_error)
                errors_per_adverbial_sq[adverbial].append(sq_error)

        # --- Step 4: Final Metrics ---

        # Global
    mae = np.mean(all_abs_errors) if all_abs_errors else 0.0
    medse = np.median(all_sq_errors) if all_sq_errors else 0.0

    # Per Event
    mae_per_event = {
        i: np.mean(errs) if errs else 0.0
        for i, errs in errors_per_event_abs.items()
    }
    medse_per_event = {
        i: np.median(errs) if errs else 0.0
        for i, errs in errors_per_event_sq.items()
    }

    # Per Adverbial
    mae_per_adverbial = {
        adv: np.mean(errs) if errs else 0.0
        for adv, errs in errors_per_adverbial_abs.items()
    }
    medse_per_adverbial = {
        adv: np.median(errs) if errs else 0.0
        for adv, errs in errors_per_adverbial_sq.items()
    }

    print(
        f"\nFinal MAE: {mae:.4f}, "
        f"Final MedSE: {medse:.4f}, "
        f"used configuration: {predict_fn.__name__}"
    )

    return {
        "overall_mae": mae,
        "overall_medse": medse,
        # "mae_per_event": mae_per_event,
        # "medse_per_event": medse_per_event,
        # "mae_per_adverbial": mae_per_adverbial,
        # "medse_per_adverbial": medse_per_adverbial,
        "all_abs_errors": all_abs_errors,
        "all_sq_errors": all_sq_errors
    }


def calculate_model_significance(results_m1, results_m2, model_1_name="Modell 1", model_2_name="Modell 2"):
    """
    Takes the output dictionaries from two consecutive `run_MAE_MdSE_evaluation` runs
    and calculates the Wilcoxon significance.
    """
    errors_m1 = np.array(results_m1["all_abs_errors"])
    errors_m2 = np.array(results_m2["all_abs_errors"])

    sq_errors_m1 = np.array(results_m1["all_sq_errors"])
    sq_errors_m2 = np.array(results_m2["all_sq_errors"])

    assert len(errors_m1) == len(errors_m2), (f"Error: Length of the items is different")

    # Wilcoxon-Test
    p_val_mae = 1.0
    p_val_medse = 1.0

    if not np.array_equal(errors_m1, errors_m2):
        _, p_val_mae = wilcoxon(errors_m1, errors_m2)

    if not np.array_equal(sq_errors_m1, sq_errors_m2):
        _, p_val_medse = wilcoxon(sq_errors_m1, sq_errors_m2)

    print(f"\n==================================================")
    print(f" Significance test: {model_1_name} vs. {model_2_name}")
    print(f"==================================================")
    print(f"Number of compared items: {len(errors_m1)}")
    print(f"--------------------------------------------------")
    print(f"MAE-Significance (Absolute Error):")
    print(f"  p-Value: {p_val_mae:.6f}")
    print(f"  Status: {'SIGNIFICANT (p < 0.05)' if p_val_mae < 0.05 else 'Not significant'}")
    print(f"--------------------------------------------------")
    print(f"MedSE-Significance (Squared Error):")
    print(f"  p-Value: {p_val_medse:.6f}")
    print(f"  Status: {'SIGNIFICANT (p < 0.05)' if p_val_medse < 0.05 else 'Not significant'}")
    print(f"==================================================\n")

    return {
        "p_value_mae": p_val_mae,
        "p_value_medse": p_val_medse,
        "is_mae_significant": p_val_mae < 0.05,
        "is_medse_significant": p_val_medse < 0.05
    }


def run_leave_one_out_evaluation_and_save_pred(events, fit_models_fn, predict_fn, events_nl=None, function_to_use=None):
    raw_results = []

    for i, event in enumerate(events):
        other_events = events[:i] + events[i+1:]
        other_events_nl = events_nl[:i] + events_nl[i + 1:]
        fit_event_adverbials(other_events)
        fit_models_fn(other_events, other_events_nl, function_to_use)

        cleaned_data = get_cleaned_data(event)

        # --- Step 1: Gather all possible numeric keys ---
        overall_targets = {adv: {k: float(np.median(v)) for k, v in cleaned_data[adv].items()} for adv in
                           config.VAGUE_ADVERBIALS}
        all_keys = sorted({int(k) for adv in overall_targets.values() for k in adv.keys()})

        # --- Step 2: Function to get value for a given key, using nearest if missing ---
        def get_value_for_key(adv_dict, target):
            numeric_keys = sorted(int(k) for k in adv_dict.keys())
            if str(target) in adv_dict:
                return adv_dict[str(target)]
            nearest = min(numeric_keys, key=lambda x: abs(x - target))
            return adv_dict[str(nearest)]

        # --- Step 3: Build inverted dict ---
        gt_adverbials = {}
        for key in all_keys:
            values = {adv: get_value_for_key(adv_dict, key) for adv, adv_dict in overall_targets.items()}
            gt_adverbials[key] = values

        # --- Collect predictions vs truth for metrics ---
        for minutes_ago, adverbial_values in gt_adverbials.items():
            if predict_fn in requires_properties:
                with open(f"{config.DATA_DIR}/event_properties.json", "r", encoding="utf-8") as fh:
                    event_properties = json.load(fh)

                prop_dict = event_properties[events_nl[i].replace("Tom", "A friend")]
                properties = [{prop: prop_dict[prop] for prop in config.properties_to_use}]
                predictions = predict_fn(properties, int(minutes_ago), function_to_use)
            else:
                input_data = events_nl[i] if events_nl else event
                predictions = predict_fn(input_data, int(minutes_ago))

            raw_results.append({
                "Event": event,
                "Minutes ago": float(minutes_ago),
                "Prediction": predictions,
                "GT": adverbial_values
            })
    return raw_results


def calculate_metrics(file_path):
    """
    Calculate per-evaluation-point metrics for every saved model prediction file.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping:
            model/file stem -> dataframe containing one row per
            (Event, Minutes ago) evaluation point.
    """

    def rank_from_scores(scores, descending=True):
        """
        Normalize scores into ranks (1 = best, ties get average rank).

        Supports:
          - dict[label -> score]  -> true ranking (rankable=True)
          - list/tuple/set[str]   -> all items become rank 1 (rankable=False)
          - str                   -> that single label becomes rank 1 (rankable=False)

        Returns:
          ranks: dict[label] -> rank (float)
          rankable: bool
        """

        # Dict case: compute average/mid ranks
        if isinstance(scores, dict) and len(scores) > 0:
            items = list(scores.items())
            items.sort(key=lambda x: x[1], reverse=descending)

            ranks = {}
            i = 0

            while i < len(items):
                j = i

                while j < len(items) and items[j][1] == items[i][1]:
                    j += 1

                # Average of the occupied 1-based ranks
                avg_rank = (i + 1 + j) / 2.0

                for k in range(i, j):
                    ranks[items[k][0]] = avg_rank

                i = j

            return ranks, True

        # list / tuple / set -> all items treated as top-ranked
        if isinstance(scores, (list, tuple, set)) and len(scores) > 0:
            return {str(lbl): 1.0 for lbl in scores}, False

        # Single string -> top-ranked item
        if isinstance(scores, str) and scores:
            return {scores: 1.0}, False

        return {}, False


    def topk_on_ranks(rank_pred, rank_gt, k=2):

        def get_topk(rank_dict, k):
            if not rank_dict:
                return set()

            uniq_ranks = sorted(set(rank_dict.values()))
            kth_rank = uniq_ranks[min(k - 1, len(uniq_ranks) - 1)]

            return {
                label
                for label, rank in rank_dict.items()
                if rank <= kth_rank
            }

        # ----------------
        # Top-1
        # ----------------
        pred_top1 = get_topk(rank_pred, 1)
        gt_top1 = get_topk(rank_gt, 1)

        inter1 = pred_top1 & gt_top1

        acc1 = 1.0 if inter1 else 0.0
        p1 = len(inter1) / len(pred_top1) if pred_top1 else 0.0
        r1 = len(inter1) / len(gt_top1) if gt_top1 else 0.0

        # ----------------
        # Top-2
        # ----------------
        pred_top2 = get_topk(rank_pred, 2)
        gt_top2 = get_topk(rank_gt, 2)

        inter2 = pred_top2 & gt_top2

        acc2 = 1.0 if inter2 else 0.0
        p2 = len(inter2) / len(pred_top2) if pred_top2 else 0.0
        r2 = len(inter2) / len(gt_top2) if gt_top2 else 0.0

        return {
            "top1": (acc1, p1, r1),
            "top2": (acc2, p2, r2),
        }


    # Stores the per-point results for each model
    all_model_results = {}

    # ----------------------------------------------------------
    # Loop through prediction files
    # ----------------------------------------------------------
    for json_file in sorted(file_path.glob("*.json")):

        print(f"\nFile: {json_file.name}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)["raw"]

        results = []

        for r in data:

            pred = r["Prediction"]
            gt = r["GT"]

            rank_pred, pred_rankable = rank_from_scores(
                pred,
                descending=True
            )

            rank_gt, gt_rankable = rank_from_scores(
                gt,
                descending=True
            )

            rankable = pred_rankable and gt_rankable

            # ==================================================
            # Kendall Tau-b + NDCG
            # ==================================================
            if rankable:

                # Keep identical adverbial order
                labels = list(gt.keys())

                x = [rank_pred[label] for label in labels]
                y = [rank_gt[label] for label in labels]

                # ----------------
                # Kendall Tau-b
                # ----------------
                tau, _ = kendalltau(
                    x,
                    y,
                    variant="b",
                    nan_policy="raise",
                    alternative="two-sided",
                    method="auto"
                )

                # ----------------
                # NDCG
                # ----------------
                # Transform ranks so larger = better
                max_r = max(
                    max(rank_gt.values()),
                    max(rank_pred.values())
                )

                y_true = [[
                    max_r + 1 - rank_gt[label]
                    for label in labels
                ]]

                y_score = [[
                    max_r + 1 - rank_pred[label]
                    for label in labels
                ]]

                if any(v > 0 for v in y_true[0]):
                    ndcg = ndcg_score(y_true, y_score)
                else:
                    ndcg = np.nan

            else:
                # Do NOT interpret "not computable" as 0.
                tau = np.nan
                ndcg = np.nan

            # ==================================================
            # Top-k
            # ==================================================
            topk = topk_on_ranks(
                rank_pred,
                rank_gt
            )

            results.append({
                # Identifiers are important for paired tests
                "Event": r["Event"],
                "Minutes ago": float(r["Minutes ago"]),

                # Ranking metrics
                "KendallTauB": (
                    float(tau)
                    if not pd.isna(tau)
                    else np.nan
                ),
                "NDCG": (
                    float(ndcg)
                    if not pd.isna(ndcg)
                    else np.nan
                ),

                # Top-1
                "Acc_1": topk["top1"][0],
                "Prec_1": topk["top1"][1],
                "Rec_1": topk["top1"][2],

                # Top-2
                "Acc_2": topk["top2"][0],
                "Prec_2": topk["top2"][1],
                "Rec_2": topk["top2"][2],
            })

        df = pd.DataFrame(results)

        # File stem is used as model name:
        # random_forest.json -> "random_forest"
        model_name = json_file.stem

        all_model_results[model_name] = df

        # ------------------------------------------------------
        # Print values used for Table 2
        # ------------------------------------------------------
        overall = (
            df.mean(numeric_only=True)
            .to_frame(name="Overall")
            .T
        )

        print(overall.round(3))

    return all_model_results


# ======================================================================
# Helper: align the exact same evaluation points between two models
# ======================================================================

def align_model_results(df_a, df_b, metric):
    """
    Align two models by Event + Minutes ago.

    This guarantees that the significance tests are paired on exactly
    the same evaluation cases.
    """

    keys = ["Event", "Minutes ago"]

    a = (
        df_a[keys + [metric]]
        .rename(columns={metric: "A"})
    )

    b = (
        df_b[keys + [metric]]
        .rename(columns={metric: "B"})
    )

    paired = a.merge(
        b,
        on=keys,
        how="inner",
        validate="one_to_one"
    )

    return paired


# ======================================================================
# Wilcoxon for Kendall Tau-b and NDCG
# ======================================================================

def wilcoxon_model_comparison(df_a, df_b, metric):
    """
    Two-sided paired Wilcoxon signed-rank test.

    Intended for:
        - KendallTauB
        - NDCG
    """

    paired = align_model_results(
        df_a,
        df_b,
        metric
    )

    # Remove observations where either model has an undefined value
    paired = paired.dropna(
        subset=["A", "B"]
    )

    x = paired["A"].to_numpy(dtype=float)
    y = paired["B"].to_numpy(dtype=float)

    if len(x) == 0:
        return {
            "N": 0,
            "Mean A": np.nan,
            "Mean B": np.nan,
            "Mean Difference": np.nan,
            "Median Difference": np.nan,
            "Statistic": np.nan,
            "p-value": np.nan,
        }

    differences = x - y

    # If both models are identical for every point,
    # Wilcoxon cannot meaningfully calculate a statistic.
    # In that case p = 1.
    if np.all(np.isclose(differences, 0.0)):

        statistic = 0.0
        p_value = 1.0

    else:
        statistic, p_value = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto"
        )

    return {
        "N": len(paired),
        "Mean A": np.mean(x),
        "Mean B": np.mean(y),
        "Mean Difference": np.mean(differences),
        "Median Difference": np.median(differences),
        "Statistic": statistic,
        "p-value": p_value,
    }


# ======================================================================
# Exact McNemar for Top-1 Accuracy
# ======================================================================

def mcnemar_model_comparison(df_a, df_b):
    """
    Exact two-sided McNemar test for paired Top-1 correctness.

    Acc_1 must contain:
        1 = correct
        0 = incorrect
    """

    paired = align_model_results(
        df_a,
        df_b,
        "Acc_1"
    )

    paired = paired.dropna(
        subset=["A", "B"]
    )

    a = paired["A"].astype(int).to_numpy()
    b = paired["B"].astype(int).to_numpy()

    # Contingency table:
    #
    #                   B correct    B incorrect
    # A correct         both         A only
    # A incorrect       B only       neither

    both_correct = int(
        np.sum((a == 1) & (b == 1))
    )

    a_correct_b_wrong = int(
        np.sum((a == 1) & (b == 0))
    )

    a_wrong_b_correct = int(
        np.sum((a == 0) & (b == 1))
    )

    both_wrong = int(
        np.sum((a == 0) & (b == 0))
    )

    discordant = (
        a_correct_b_wrong
        + a_wrong_b_correct
    )

    # Exact McNemar test is equivalent to a binomial test
    # on the discordant pairs.
    if discordant == 0:

        p_value = 1.0

    else:

        p_value = binomtest(
            k=min(
                a_correct_b_wrong,
                a_wrong_b_correct
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    return {
        "N": len(paired),

        "Accuracy A": np.mean(a),
        "Accuracy B": np.mean(b),
        "Accuracy Difference": (
            np.mean(a) - np.mean(b)
        ),

        "Both correct": both_correct,
        "A correct / B wrong": a_correct_b_wrong,
        "A wrong / B correct": a_wrong_b_correct,
        "Both wrong": both_wrong,

        "Discordant pairs": discordant,
        "p-value": p_value,
    }


# ======================================================================
# Run all pairwise significance tests
# ======================================================================

def perform_significance_tests(
        all_model_results,
        model_names=None
):
    """
    Perform pairwise significance tests between model configurations.

    KendallTauB -> Wilcoxon signed-rank
    NDCG        -> Wilcoxon signed-rank
    Acc_1       -> exact McNemar

    No multiple-comparison correction is applied.

    Parameters
    ----------
    all_model_results : dict[str, pd.DataFrame]
        Output of calculate_metrics().

    model_names : list[str] or None
        Optional list specifying which models should be compared.
        If None, all loaded JSON prediction files are compared.

    Returns
    -------
    dict[str, pd.DataFrame]
        Results for KendallTauB, NDCG, and Top1.
    """

    if model_names is None:
        model_names = list(
            all_model_results.keys()
        )

    print("\nModels included in significance testing:")
    for model in model_names:
        print(f"  - {model}")

    # Check that requested models actually exist
    missing = [
        model
        for model in model_names
        if model not in all_model_results
    ]

    if missing:
        raise ValueError(
            "The following model result files were not found: "
            + ", ".join(missing)
        )

    comparisons = list(
        itertools.combinations(
            model_names,
            2
        )
    )

    # ==============================================================
    # Kendall Tau-b
    # ==============================================================
    kendall_results = []
    for model_a, model_b in comparisons:
        result = wilcoxon_model_comparison(
            all_model_results[model_a],
            all_model_results[model_b],
            metric="KendallTauB"
        )
        result = {
            "Model A": model_a,
            "Model B": model_b,
            **result
        }
        kendall_results.append(result)
    kendall_df = pd.DataFrame(
        kendall_results
    )

    # ==============================================================
    # NDCG
    # ==============================================================
    ndcg_results = []
    for model_a, model_b in comparisons:
        result = wilcoxon_model_comparison(
            all_model_results[model_a],
            all_model_results[model_b],
            metric="NDCG"
        )
        result = {
            "Model A": model_a,
            "Model B": model_b,
            **result
        }
        ndcg_results.append(result)
    ndcg_df = pd.DataFrame(ndcg_results)

    # ==============================================================
    # Top-1 Accuracy
    # ==============================================================
    top1_results = []
    for model_a, model_b in comparisons:
        result = mcnemar_model_comparison(
            all_model_results[model_a],
            all_model_results[model_b]
        )
        result = {
            "Model A": model_a,
            "Model B": model_b,
            **result
        }
        top1_results.append(result)
    top1_df = pd.DataFrame(top1_results)

    # ==============================================================
    # Print
    # ==============================================================
    print("\n"+ "=" * 70)
    print("SIGNIFICANCE TESTS: Kendall Tau-b")
    print("Two-sided paired Wilcoxon signed-rank test")
    print("=" * 70)

    print(kendall_df.round(6).to_string(index=False))

    print("\n"+ "=" * 70)
    print("SIGNIFICANCE TESTS: NDCG")
    print("Two-sided paired Wilcoxon signed-rank test")
    print("=" * 70)

    print(ndcg_df.round(6).to_string(index=False))

    print("\n"+ "=" * 70)
    print("SIGNIFICANCE TESTS: Top-1 Accuracy")
    print("Exact two-sided McNemar test")
    print("=" * 70)

    print(top1_df.round(6).to_string(index=False))

    return {
        "KendallTauB": kendall_df,
        "NDCG": ndcg_df,
        "Top1": top1_df,
    }


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def gpt_chat_answer(model_engine, instructions, prompt):
    messages = [
        {"role": "system", "content": str(instructions)},
        {"role": "user", "content": str(prompt)}
    ]
    if "5" in model_engine:
        temperature = 1 # GPT-5 does not support a temperature of 0
    else:
        temperature = 0

    completion = client.chat.completions.create(
        model=model_engine,
        messages=messages,
        temperature=temperature
    )
    return completion.choices[0].message.content.strip()


def predict_gpt(event_nl, minutes_ago, model_engine=config.GPT_VERSION):
    instructions = """
    You are to choose the most appropriate adverbial(s) from the following options: 
    just, recently, some time ago, long time ago.
    Base your choice on the given event and the number of minutes that have passed since it happened.
    Respond ONLY with the appropriate adverbial(s) and nothing else.
    """

    # Build user prompt
    prompt = f"""
    Event: {event_nl}
    Minutes since event: {minutes_ago}
    """

    # Get GPT's answer
    if "gpt-5" in model_engine: # As version 5 does not support the temperature 0 setting get the output 5 times and chose the most frequent adverbials
        outputs = []
        for _ in range(0, 5):
            outputs.append(gpt_chat_answer(model_engine, instructions, prompt))
        counter = Counter(outputs)
        output, count = counter.most_common(1)[0]
    else:
        output = gpt_chat_answer(model_engine, instructions, prompt)
    return output, instructions, prompt

# ---------------------------------------------------------------------------
# Model-specific Evaluators
# ---------------------------------------------------------------------------
def get_predictions_embedding(events, events_nl):
    raw_results = run_leave_one_out_evaluation_and_save_pred(events, fit_event_specific_embeddings, predict_adverbial_embedding, events_nl=events_nl)
    output = {
        "raw": raw_results,
    }
    with open(config.EVALUATION_EMBEDDING_FILE, "w") as f:
        json.dump(output, f, indent=4)
    return

def get_predictions_random_forest(events, events_nl):
    raw_results = run_leave_one_out_evaluation_and_save_pred(events, fit_event_specific_random_forest, predict_adverbial_random_forest, events_nl=events_nl)
    output = {
        "raw": raw_results,
    }
    with open(config.EVALUATION_RANDOM_FOREST_FILE, "w") as f:
        json.dump(output, f, indent=4)
    return

def get_predictions_functions(events, events_nl, function_to_use):
    raw_results = run_leave_one_out_evaluation_and_save_pred(events, fit_event_specific_functions, predict_adverbial_functions, events_nl=events_nl, function_to_use=function_to_use)
    output = {
        "raw": raw_results,
    }
    with open(f"{config.EVALUATION_FUNCTIONS_FILE}{function_to_use.__name__}.json", "w") as f:
        json.dump(output, f, indent=4)
    return

def get_predictions_classifier(events, events_nl):
    raw_results = run_leave_one_out_evaluation_and_save_pred(events, fit_classifier, predict_adverbial_classifier, events_nl=events_nl)
    output = {
        "raw": raw_results
    }
    with open(config.EVALUATION_CLASSIFIER_FILE, "w") as f:
        json.dump(output, f, indent=4)
    return

def get_predictions_regression(events, events_nl):
    raw_results = run_leave_one_out_evaluation_and_save_pred(events, fit_regression, predict_adverbial_regression, events_nl=events_nl)
    output = {
        "raw": raw_results
    }
    with open(config.EVALUATION_REGRESSION_FILE, "w") as f:
        json.dump(output, f, indent=4)
    return


def evaluate_gpt(events, events_nl):
    predictions_data = []

    for i, event in enumerate(events):
        cleaned_data = get_cleaned_data(event)

        # --- Step 1: Gather all possible numeric keys ---
        overall_targets = {adv: {k: float(np.median(v)) for k, v in cleaned_data[adv].items()} for adv in config.VAGUE_ADVERBIALS}
        all_keys = sorted({int(k) for adv in overall_targets.values() for k in adv.keys()})

        # --- Step 2: Function to get value for a given key, using nearest if missing ---
        def get_value_for_key(adv_dict, target):
            numeric_keys = sorted(int(k) for k in adv_dict.keys())
            if str(target) in adv_dict:
                return adv_dict[str(target)]
            nearest = min(numeric_keys, key=lambda x: abs(x - target))
            return adv_dict[str(nearest)]

        # --- Step 3: Build inverted dict ---
        best_adverbials = {}
        for key in all_keys:
            values = {adv: get_value_for_key(adv_dict, key) for adv, adv_dict in overall_targets.items()}
            max_val = max(values.values())
            best_adverbials[key] = [adv for adv, val in values.items() if val == max_val]

        # --- Collect predictions vs truth for metrics ---
        for minutes_ago, adverbials in best_adverbials.items():
            prediction, instruction, prompt = predict_gpt(events_nl[i], minutes_ago)
            # --- Step 5: Save prediction to JSON structure ---
            predictions_data.append({
                "Prediction": prediction,
                "GT": adverbials,
                "Prompt": prompt
            })

    # --- Step 6: Write all predictions to JSON (overwrite at the end) ---
    json_drop = {
        "raw": predictions_data,
        "Instruction": instruction,
        "GPT-Version": config.GPT_VERSION
    }
    if "gpt-4" in config.GPT_VERSION:
        save_file = config.GPT4_PROMPT_FILE
    else:
        save_file = config.GPT5_PROMPT_FILE
    with open(save_file, "w") as f:
        json.dump(json_drop, f, indent=4)
    return
