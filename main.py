# src/main.py
from src.train import (fit_event_adverbials, fit_event_specific_embeddings, fit_event_specific_random_forest)
from src.plot import plot_all_persons_event_adverbials
from src.predictions import (predict_time_frame_embedding, predict_adverbial_embedding,
                             predict_time_frame_gpt_random_forest, predict_adverbial_gpt_random_forest,
                             predict_time_frame_random_forest, predict_adverbial_random_forest)
from src.evaluation import (evaluate_embedding, evaluate_gpt_random_forest, evaluate_random_forest,
                            evaluate_classifier, evaluate_regression,
                            evaluate_advanced_classifier, evaluate_advanced_regression,
                            evaluate_advanced_embedding, evaluate_advanced_gpt_random_forest, evaluate_advanced_random_forest,
                            evaluate_gpt)
from src.evaluation_survey import (evaluate_survey_gpt_random_forest, evaluate_survey_embedding, evaluate_survey_baseline)
from src.simple_models_training import fit_classifier, fit_regression
from src.simple_models_predictions import predict_adverbial_classifier, predict_adverbial_regression

import sys

# Open a file in write mode
log_file = open('output.log', 'w')
# Redirect print statements to the file
# sys.stdout = log_file


def train_models(events):
    """
    Train various models for event adverbials, embeddings, and random forests.
    """
    print("\nTraining Event Adverbials, Embeddings, and Random Forest...")
    fit_event_adverbials(events)
    fit_event_specific_embeddings(events)
    fit_event_specific_random_forest(events)

    # print("\nTraining Simple Models (Classifier, Regression)...")
    # fit_classifier(events)
    # fit_regression(events)


def make_predictions(event_details):
    """
    Make predictions using different models.
    """
    event, event_nl, duration, frequency, adverbial, minutes_ago = event_details

    # print("\nPredictions using Classifiers and Regression:")
    # print(predict_adverbial_classifier(duration, frequency, minutes_ago))
    # print(predict_adverbial_regression(duration, frequency, minutes_ago))

    print("\nPredictions using Embeddings and Random Forest:")
    # print(predict_time_frame_embedding(event, adverbial))
    print(predict_adverbial_embedding(event, minutes_ago))
    # print(predict_time_frame_gpt_random_forest(event, adverbial))
    print(predict_adverbial_gpt_random_forest(event_nl, minutes_ago))
    # print(predict_time_frame_random_forest(duration, frequency, adverbial))
    # print(predict_adverbial_random_forest(duration, frequency, minutes_ago))


def evaluate_models(events, events_nl):
    """
    Evaluate the models using MAE and RMSE.
    """
    # print("\Advanced evaluation Embeddings + Regressor:")
    # print(evaluate_embedding(events, events_nl))

    print("\nEvaluating GPT + Random Forest:")
    print(evaluate_gpt_random_forest(events, events_nl))

    # print("\Evaluating Random Forest:")
    # print(evaluate_random_forest(events))
    #
    # print("\Evaluating Baseline Models (Classifier, Regression):")
    # print(evaluate_classifier(events))
    # print(evaluate_regression(events))

def evaluate_advanced_models(events, events_nl):
    """
    Evaluate the models by comparing the predicted labels
    """
    # print("\nAdvanced evaluation Embeddings + Regressor:")
    # print(evaluate_advanced_embedding(events, events_nl))

    # print("\Advanced evaluation GPT + Random Forest:")
    # print(evaluate_advanced_gpt_random_forest(events, events_nl))
    #
    # print("\Advanced evaluation Random Forest:")
    # print(evaluate_advanced_random_forest(events))
    #
    # print("\Advanced evaluation Baseline Models (Classifier, Regression):")
    # print(evaluate_advanced_classifier(events))
    # print(evaluate_advanced_regression(events))

    print("\n Evaluating GPT:")
    print(evaluate_gpt(events, events_nl))

def evaluate_survey(events_to_fit, events_to_fit_nl = None):
    """
    Evaluate the models by comparing the predicted adverbials for new events with the survey results
    """
    print("\nEvaluating GPT + Random Forest on the survey data:")
    print(evaluate_survey_gpt_random_forest(events_to_fit))

    # print("\nEvaluating Embeddings + Regressor on the survey data:")
    # print(evaluate_survey_embedding(events_to_fit, events_to_fit_nl))

    # print("\n Evaluating the baseline (random prediction) on the survey data:")
    # print(evaluate_survey_baseline())


def plot_results(events):
    """
    Plot the results for event adverbials.
    """
    print("Plotting Event Adverbials...")
    plot_all_persons_event_adverbials(events)


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
        "Tom read a book", "Tom danced salsa", "Tom stored a wine bottle", "Tom dran juice",
        "Tom chatted with a friend", "I had my wedding celebration", "I had my wallet stolen"
    ]

    # minutes_ago = [5, 30, 60, 480, 1440, 4320, 10080, 20160, 30240, 43800, 131400, 262800, 525600, 1577000, 2628000, 5256000]
    # for minute in minutes_ago:
    #     event_details = ("own_camping", "I went camping", "Minutes", "Monthly", "just", minute)
    #     make_predictions(event_details)
    #     print(minute)

    # Run the steps sequentially
    # train_models(events) # Trains FuzzyLLI in all variants and the baseline models
    # make_predictions(event_details) #Predicts minutes ago or the event and the best fitting adverbials
    # evaluate_models(events, events_nl) # Evaluated FuzzyLLI and baseline models via leaving one out. metrics: mae and rmse
    # evaluate_advanced_models(events, events_nl)
    # evaluate_survey(events, events_nl)
    # plot_results(events) # Plots FuzzyLLI

if __name__ == '__main__':
    main()

# Close the log file at the end
log_file.close()