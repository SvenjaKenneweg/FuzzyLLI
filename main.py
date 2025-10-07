# src/main.py
from src.train import (fit_event_adverbials, fit_event_specific_embeddings, fit_event_specific_random_forest)
from src.plot import plot_all_persons_event_adverbials, plot_single_events
from src.predictions import (predict_time_frame_embedding, predict_adverbial_embedding,
                             predict_time_frame_gpt_random_forest, predict_adverbial_gpt_random_forest,
                             predict_time_frame_random_forest, predict_adverbial_random_forest, get_all_event_properties_gpt)
from src.evaluation import (get_predictions_classifier, get_predictions_regression,
                            get_predictions_embedding, get_predictions_gpt_random_forest, get_predictions_random_forest,
                            evaluate_gpt, calculate_metrics)
from src.evaluation_survey import (evaluate_survey_gpt_random_forest, evaluate_survey_embedding, evaluate_survey_gpt, evaluate_survey_gpt_regression, evaluate_survey_gpt_classifier)
from src.simple_models_training import fit_classifier, fit_regression
from src.simple_models_predictions import predict_adverbial_gpt_classifier, predict_adverbial_gpt_regression
from src.config import EVALUATION_SURVEY_FILE_PATH, EVALUATION_FILE_PATH, DATA_DIR

import sys

# Open a file in write mode
log_file = open('output.log', 'w')
# Redirect print statements to the file
# sys.stdout = log_file


def train_models(events, events_nl):
    """
    Train various models for event adverbials, embeddings, and random forests.
    """
    # print("\nTraining Event Adverbials, Embeddings, and Random Forest...")
    # fit_event_adverbials(events)
    # fit_event_specific_embeddings(events, events_nl)
    # fit_event_specific_random_forest(events)
    #
    # print("\nTraining Simple Models (Classifier, Regression)...")
    # fit_classifier(events)
    # fit_regression(events)

    print("\nGet the event properties using GPT")
    get_all_event_properties_gpt(events_nl, DATA_DIR / "event_properties.json")


def make_predictions(event_details):
    """
    Make predictions using different models.
    """
    event, event_nl, duration, frequency, adverbial, minutes_ago = event_details

    print("\nPredictions using Classifiers and Regression:")
    print(predict_adverbial_gpt_classifier(duration, frequency, minutes_ago))
    print(predict_adverbial_gpt_regression(duration, frequency, minutes_ago))

    print("\nPredictions using Embeddings and Random Forest:")
    print(predict_time_frame_embedding(event, adverbial))
    print(predict_adverbial_embedding(event, minutes_ago))
    print(predict_time_frame_gpt_random_forest(event, adverbial))
    print(predict_adverbial_gpt_random_forest(event_nl, minutes_ago))
    print(predict_time_frame_random_forest(duration, frequency, adverbial))
    print(predict_adverbial_random_forest(duration, frequency, minutes_ago))


def evaluate_models_seen_events(events, events_nl, generate_new_predictions=False):
    """
    Evaluate the models by comparing the predicted labels on seen events
    """
    if generate_new_predictions:
        print("\nEvaluation Embeddings + Regressor:")
        get_predictions_embedding(events, events_nl)

        print("\nEvaluation GPT + Random Forest:")
        get_predictions_gpt_random_forest(events, events_nl)

        print("\nEvaluation Random Forest:")
        get_predictions_random_forest(events)

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
        print("\nEvaluating GPT + Random Forest on the unseen_events data:")
        evaluate_survey_gpt_random_forest(events_to_fit, events_to_fit_nl)

        print("\nEvaluating Embeddings + Regressor on the unseen_events data:")
        evaluate_survey_embedding(events_to_fit, events_to_fit_nl)

        print("\nEvaluating Classifier and Regression Model on the unseen_events data:")
        evaluate_survey_gpt_classifier(events_to_fit, events_to_fit_nl)
        evaluate_survey_gpt_regression(events_to_fit, events_to_fit_nl)

        print("\nEvaluating GPT on the unseen events data:")
        evaluate_survey_gpt()

    # Calculate only the metrics
    print("Calculating the Evaluation metrics from the saved prediction files for the survey data...")
    calculate_metrics(EVALUATION_SURVEY_FILE_PATH)


def plot_results(events):
    """
    Plot the results for event adverbials.
    """
    # print("Plotting Event Adverbials...")
    # plot_all_persons_event_adverbials(events)

    for event_name in events:
        print("Plotting single Event")
        plot_single_events(event_name)


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

    # minutes_ago = [5, 30, 60, 480, 1440, 4320, 10080, 20160, 30240, 43800, 131400, 262800, 525600, 1577000, 2628000, 5256000]
    # for minute in minutes_ago:
    #     event_details = ("own_camping", "I went camping", "Minutes", "Monthly", "just", minute)
    #     make_predictions(event_details)
    #     print(minute)

    # Run the steps sequentially
    train_models(events, events_nl) # Trains FuzzyLLI in all variants and the baseline models
    # make_predictions(event_details) #Predicts minutes ago or the event and the best fitting adverbials
    # evaluate_models_seen_events(events, events_nl, generate_new_predictions=False)
    # evaluate_survey(events, events_nl, generate_new_predictions=False)
    # plot_results(events) # Plots FuzzyLLI

if __name__ == '__main__':
    main()

# Close the log file at the end
log_file.close()