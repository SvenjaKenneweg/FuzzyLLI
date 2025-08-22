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

    print("\nTraining Simple Models (Classifier, Regression)...")
    fit_classifier(events)
    fit_regression(events)


def make_predictions(event_details):
    """
    Make predictions using different models.
    """
    event, duration, frequency, adverbial, minutes_ago = event_details

    print("\nPredictions using Classifiers and Regression:")
    print(predict_adverbial_classifier(duration, frequency, minutes_ago))
    print(predict_adverbial_regression(duration, frequency, minutes_ago))

    print("\nPredictions using Embeddings and Random Forest:")
    print(predict_time_frame_embedding(event, adverbial))
    print(predict_adverbial_embedding(event, minutes_ago))
    print(predict_time_frame_gpt_random_forest(event, adverbial))
    print(predict_adverbial_gpt_random_forest(event, minutes_ago))
    print(predict_time_frame_random_forest(duration, frequency, adverbial))
    print(predict_adverbial_random_forest(duration, frequency, minutes_ago))


def evaluate_models(events, metric='rmse'):
    """
    Evaluate the models using different metrics.
    """
    print("\nEvaluating Embeddings + Regressor:")
    print(evaluate_embedding(events, metric=metric))

    print("\nEvaluating GPT + Random Forest:")
    print(evaluate_gpt_random_forest(events, metric=metric))

    print("\nEvaluating Random Forest:")
    print(evaluate_random_forest(events, metric=metric))

    print("\nEvaluating Baseline Models (Classifier, Regression):")
    print(evaluate_classifier(events, metric=metric))
    print(evaluate_regression(events, metric=metric))

def evaluate_advanced_models(events):
    """
    Evaluate the models by comparing the predicted labels
    """
    print("\nEvaluating Embeddings + Regressor:")
    print(evaluate_advanced_embedding(events))
    #
    # print("\nEvaluating GPT + Random Forest:")
    # print(evaluate_advanced_gpt_random_forest(events))
    #
    # print("\nEvaluating Random Forest:")
    # print(evaluate_advanced_random_forest(events))

    # print("\nEvaluating Baseline Models (Classifier, Regression):")
    # print(evaluate_advanced_classifier(events))
    # print(evaluate_advanced_regression(events))

    # print("\n Evaluating GPT:")
    # print(evaluate_gpt(events))


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

    events = [
        "tom_wedding_celebration", "own_year_abroad"
    ]

    event_details = ("Tom had a meeting", "Minutes", "Monthly", "just", 144000)

    # Run the steps sequentially
    # train_models(events) # Trains FuzzyLLI in all variants and the baseline models
    # make_predictions(event_details) #Predicts minutes ago or the event and the best fitting adverbials
    # evaluate_models(events, metric='rmse') # Evaluated FuzzyLLI and baseline models via leaving one out. Possible metrics: mae and rmse
    evaluate_advanced_models(events)
    # plot_results(events) # Plots FuzzyLLI

if __name__ == '__main__':
    main()

# Close the log file at the end
log_file.close()