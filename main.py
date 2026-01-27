# src/main.py
import argparse
import json
from itertools import combinations

from src.train import (fit_event_adverbials, fit_event_specific_embeddings, fit_event_specific_random_forest, fit_event_specific_functions)
from src.plot import plot_all_persons_event_adverbials, plot_events_adverbials_fitted
from src.predictions import (predict_time_frame_embedding, predict_adverbial_embedding, predict_adverbial_functions,
                             predict_time_frame_random_forest, predict_adverbial_random_forest, get_all_event_properties_gpt)
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
    "0", "1", "2"
]

def train_models(spatial_surveys):
    """
    Train various models for event adverbials, embeddings, and random forests.
    """
    print("\nTraining FuzzyLLI for the spatial surveys")
    fit_event_adverbials(spatial_surveys)


def evaluate_models_training_dataset(events, events_nl, generate_new_predictions=False, calculate_MAE = False):
    """
    Evaluate the models by comparing the predicted labels on training dataset (leave-one-out)
    """
    if calculate_MAE:
        print("\nCalculating MAE and MdSE:")
        print(run_MAE_MdSE_evaluation(events, fit_event_specific_embeddings, predict_adverbial_embedding,
                                      events_nl=events_nl))
        print(run_MAE_MdSE_evaluation(events, fit_event_specific_random_forest, predict_adverbial_random_forest,
                           events_nl=events_nl))
        print(run_MAE_MdSE_evaluation(events, fit_event_specific_functions, predict_adverbial_functions, events_nl=events_nl,
                           function_to_use=config.powerlaw))

    if generate_new_predictions:
        print("\nEvaluation Embeddings + Regressor:")
        get_predictions_embedding(events, events_nl)

        print("\nEvaluation Random Forest:")
        get_predictions_random_forest(events, events_nl)

        print("\nEvaluation Power Law:")
        get_predictions_functions(events, events_nl, function_to_use=config.powerlaw)

    # Calculate only the metrics
    print("\nCalculating the Evaluation metrics from the saved prediction files for the training datasets...")
    calculate_metrics(config.EVALUATION_FILE_PATH)


def plot_results(events, adverbial, predict_functions=None):
    """
    Plot the results for event adverbials.
    """
    print("Plotting results are saved under results/plots/... . "
          "The used FuzzyLLI configuration for the plotting is Random Forest")
    print("\nPlotting the FuzzyLLI Overview Diagram (Left each event, right the adverbials")
    plot_all_persons_event_adverbials(events)
    # plot_events_adverbials_fitted(events, events_nl, adverbial, predict_functions)


def run_full_pipeline():
    """
    Preserve the previous default: train, evaluate, plot, and run a demo prediction.
    """
    train_models(DEFAULT_SPATIAL_SURVEY)  # Trains FuzzyLLI
    # evaluate_models_training_dataset(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, generate_new_predictions=False)
    plot_results(DEFAULT_SPATIAL_SURVEY, "long time ago", predict_functions=None)  # Plots FuzzyLLI


if __name__ == '__main__':
    run_full_pipeline()
