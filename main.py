# src/main.py
import argparse
import json
from itertools import combinations

from src.train import (fit_event_adverbials, fit_event_specific_embeddings, fit_event_specific_random_forest, fit_event_specific_functions)
from src.plot import plot_all_persons_event_adverbials, plot_events_adverbials_fitted
from src.predictions import (predict_time_frame_embedding, predict_adverbial_embedding, predict_adverbial_functions, predict_adverbial_fuzzylli,
                             predict_distance_fuzzylli, predict_time_frame_random_forest, predict_adverbial_random_forest, get_all_event_properties_gpt)
from src.evaluation_training_dataset import (get_predictions_classifier, get_predictions_regression,
                                             get_predictions_embedding, get_predictions_random_forest,
                                             get_predictions_functions, run_MAE_MdSE_evaluation,
                                             evaluate_gpt, calculate_metrics)
from src.evaluation_test_dataset import (evaluate_test_data_random_forest, evaluate_test_data_functions,
                                         evaluate_test_data_embedding, evaluate_test_data_gpt, evaluate_test_data_regression, evaluate_test_data_classifier)
from src.simple_models_training import fit_classifier, fit_regression
from src.simple_models_predictions import predict_adverbial_classifier, predict_adverbial_regression
import src.config as config


DEFAULT_SPATIAL_SURVEY = [
    "Big", "Small"
]

def train_models(spatial_surveys):
    """
    Train various models for event adverbials, embeddings, and random forests.
    """
    print("\nTraining FuzzyLLI for the spatial surveys")
    fit_event_adverbials(spatial_surveys)
    print(predict_adverbial_fuzzylli("Small", 20))
    print(predict_distance_fuzzylli("Small", "moderately far"))


def evaluate_models_training_dataset(spatial_surveys):
    """
    Evaluate the models by calculating the MAE and MdSE
    """
    print("\nCalculating MAE and MdSE:")
    print(run_MAE_MdSE_evaluation(spatial_surveys, None, predict_adverbial_fuzzylli))


def plot_results(events, adverbial):
    """
    Plot the results for event adverbials.
    """
    print("Plotting results are saved under results/plots/... . "
          "The used FuzzyLLI configuration for the plotting is Random Forest")
    print("\nPlotting the FuzzyLLI Overview Diagram (Left each event, right the adverbials")
    # plot_all_persons_event_adverbials(events)
    plot_events_adverbials_fitted(["Big", "Small"], "far away")


def run_full_pipeline():
    """
    Preserve the previous default: train, evaluate, plot, and run a demo prediction.
    """
    # train_models(DEFAULT_SPATIAL_SURVEY)  # Trains FuzzyLLI
    # evaluate_models_training_dataset(DEFAULT_SPATIAL_SURVEY) # Calculates MAE and MdSE
    plot_results(DEFAULT_SPATIAL_SURVEY, "close to")  # Plots FuzzyLLI


if __name__ == '__main__':
    run_full_pipeline()
