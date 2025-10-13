# src/main.py
from src.train import (fit_event_adverbials, fit_event_specific_embeddings, fit_event_specific_random_forest, fit_event_specific_functions)
from src.plot import plot_all_persons_event_adverbials, plot_single_events
from src.predictions import (predict_time_frame_embedding, predict_adverbial_embedding, predict_adverbial_functions,
                             predict_time_frame_random_forest, predict_adverbial_random_forest, get_all_event_properties_gpt)
from src.evaluation import (get_predictions_classifier, get_predictions_regression,
                            get_predictions_embedding, get_predictions_random_forest,
                            get_predictions_functions,
                            evaluate_gpt, calculate_metrics)
from src.evaluation_survey import (evaluate_survey_random_forest, evaluate_survey_functions,
                                   evaluate_survey_embedding, evaluate_survey_gpt, evaluate_survey_regression, evaluate_survey_classifier)
from src.simple_models_training import fit_classifier, fit_regression
from src.simple_models_predictions import predict_adverbial_classifier, predict_adverbial_regression
from src.config import EVALUATION_SURVEY_FILE_PATH, EVALUATION_FILE_PATH, DATA_DIR, DATA_EVALUATION_SURVEY_PATH, \
    powerlaw, exp_decay

import sys
import pandas as pd

# Open a file in write mode
log_file = open('output.log', 'w')
# Redirect print statements to the file
# sys.stdout = log_file


def train_models(events, events_nl):
    """
    Train various models for event adverbials, embeddings, and random forests.
    """
    print("\nGet the event properties using GPT")
    get_all_event_properties_gpt(events_nl, DATA_DIR / "event_properties.json")
    events_nl_survey = ["You attended a meeting", "You bought a house", "You went camping", "You went to a concert", "You ate breakfast"]
    get_all_event_properties_gpt(events_nl_survey, DATA_EVALUATION_SURVEY_PATH / "event_properties.json")

    # print("\nTraining Event Adverbials, Embeddings, and Random Forest...")
    # fit_event_adverbials(events)
    # fit_event_specific_embeddings(events, events_nl)
    # fit_event_specific_random_forest(events, events_nl)
    # fit_event_specific_functions(events, events_nl,powerlaw)
    # fit_event_specific_functions(events, events_nl, exp_decay)
    #
    # print("\nTraining Simple Models (Classifier, Regression)...")
    # fit_classifier(events, events_nl)
    # fit_regression(events, events_nl)


def make_predictions(event_nl, event_properties, adverbial, minutes_ago):
    """
    Make predictions using different models.
    """
    print("\nPredictions using Classifiers and Regression:")
    print(predict_adverbial_classifier(event_nl, minutes_ago, event_properties))
    print(predict_adverbial_regression(event_nl, minutes_ago, event_properties))

    print("\nPredictions using Embeddings and Random Forest:")
    print(predict_time_frame_embedding(event_nl, adverbial))
    print(predict_adverbial_embedding(event_nl, minutes_ago))
    print(predict_time_frame_random_forest(event_properties, adverbial))
    print(predict_adverbial_random_forest(event_properties, minutes_ago))
    print(predict_adverbial_functions(event_properties, minutes_ago, powerlaw))
    print(predict_adverbial_functions(event_properties, minutes_ago, exp_decay))


def evaluate_models_seen_events(events, events_nl, generate_new_predictions=False):
    """
    Evaluate the models by comparing the predicted labels on seen events
    """
    if generate_new_predictions:
        print("\nEvaluation Embeddings + Regressor:")
        get_predictions_embedding(events, events_nl)

        print("\nEvaluation Random Forest:")
        get_predictions_random_forest(events, events_nl)

        print("\nEvaluation Power Law and Exponential Decay:")
        get_predictions_functions(events, events_nl, function_to_use=powerlaw)
        get_predictions_functions(events, events_nl, function_to_use=exp_decay)

        print("\nEvaluation Baseline Models (Classifier, Regression):")
        get_predictions_classifier(events, events_nl)
        get_predictions_regression(events, events_nl)

        # print("\n Evaluating GPT:")
        # evaluate_gpt(events, events_nl)

    # Calculate only the metrics
    print("Calculating the Evaluation metrics from the saved prediction files for the experimental data...")
    calculate_metrics(EVALUATION_FILE_PATH)



def evaluate_survey(events_to_fit, events_to_fit_nl, generate_new_predictions=False):
    """
    Evaluate the models by comparing the predicted adverbials for new events with the unseen_events results
    """
    if generate_new_predictions:
        print("\nEvaluating Random Forest on the unseen_events data:")
        evaluate_survey_random_forest(events_to_fit, events_to_fit_nl)

        print("\nEvaluating Embeddings + Regressor on the unseen_events data:")
        evaluate_survey_embedding(events_to_fit, events_to_fit_nl)

        print("\nEvaluation Power Law and Exponential Decay on the unseen survey data:")
        evaluate_survey_functions(events_to_fit, events_to_fit_nl, powerlaw)
        evaluate_survey_functions(events_to_fit, events_to_fit_nl, exp_decay)

        print("\nEvaluating Classifier and Regression Model on the unseen_events data:")
        evaluate_survey_classifier(events_to_fit, events_to_fit_nl)
        evaluate_survey_regression(events_to_fit, events_to_fit_nl)

        # print("\nEvaluating GPT on the unseen events data:")
        # evaluate_survey_gpt()

    # Calculate only the metrics
    print("Calculating the Evaluation metrics from the saved prediction files for the survey data...")
    calculate_metrics(EVALUATION_SURVEY_FILE_PATH)


def plot_results(events, events_nl):
    """
    Plot the results for event adverbials.
    """
    # print("Plotting Event Adverbials...")
    # plot_all_persons_event_adverbials(events)

    for event_name, event_name_nl in zip(events, events_nl):
        print("Plotting single Event")
        plot_single_events(event_name, event_name_nl, predict_adverbial_random_forest)


def main():
    events = [
        "tom_wedding_celebration", "own_year_abroad", "own_birthday", "own_vacation",
        "own_rent_payment", "own_shower", "tom_watching_film", "tom_eating_risotto",
        "tom_reading_book", "tom_dancing_salsa", "tom_storing_wineBottle", "tom_drinking_juice",
        "tom_chatting_friend", "own_wedding_celebration", "own_wallet_theft"
    ]

    # Events in Natural Language Format used for GPT to predict the event properties.
    events_nl = [
        "Tom had his wedding celebration", "I spent a year abroad", "I had my birthday", "I went on vacation",
        "I paid rent", "I took a shower", "Tom watched a film", "Tom ate risotto",
        "Tom read a book", "Tom danced salsa", "Tom stored a wine bottle", "Tom drank juice",
        "Tom chatted with a friend", "I had my wedding celebration", "I had my wallet stolen"
    ]


    # Test event for making the predictions
    event_nl = "I was at the hospital"
    event_properties = pd.DataFrame([{
        'Frequency': 2,
        'Richness': 5,
        'Importance': 4
    }])
    adverbial = "just"
    minutes_ago = 120

    # Run the steps sequentially
    # train_models(events, events_nl) # Trains FuzzyLLI in all variants and the baseline models
    # make_predictions(event_nl, event_properties, adverbial, minutes_ago) #Predicts minutes ago or the event and the best fitting adverbials
    evaluate_models_seen_events(events, events_nl, generate_new_predictions=True)
    evaluate_survey(events, events_nl, generate_new_predictions=True)
    # plot_results(events, events_nl) # Plots FuzzyLLI

if __name__ == '__main__':
    main()

# Close the log file at the end
log_file.close()