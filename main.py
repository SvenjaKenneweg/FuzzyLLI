# src/main.py
import argparse
import json
from itertools import combinations

from src.train import (fit_event_adverbials, fit_event_specific_embeddings, fit_event_specific_random_forest, fit_event_specific_functions)
from src.plot import plot_all_persons_event_adverbials, plot_single_events, plot_events_adverbials
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
import src.config as config


DEFAULT_EVENTS = [
    "tom_wedding_celebration", "own_year_abroad", "own_birthday", "own_vacation",
    "own_rent_payment", "own_shower", "tom_watching_film", "tom_eating_risotto",
    "tom_reading_book", "tom_dancing_salsa", "tom_storing_wineBottle", "tom_drinking_juice",
    "tom_chatting_friend", "own_wedding_celebration", "own_wallet_theft"
]

# Events in Natural Language Format
DEFAULT_EVENTS_NL = [
    "Tom had his wedding celebration", "I spent a year abroad", "I had my birthday", "I went on vacation",
    "I paid rent", "I took a shower", "Tom watched a film", "Tom ate risotto",
    "Tom read a book", "Tom danced salsa", "Tom stored a wine bottle", "Tom drank juice",
    "Tom chatted with a friend", "I had my wedding celebration", "I had my wallet stolen"
]

DEFAULT_EVENT_PROPERTIES = [{'Richness': 5, 'Frequency': 1, 'Importance': 5}]
DEFAULT_EVENT_NL = "I was at the hospital"
DEFAULT_ADVERBIAL = "just"
DEFAULT_MINUTES_AGO = 120


def train_models(events, events_nl):
    """
    Train various models for event adverbials, embeddings, and random forests.
    """
    print("\nGet the event properties using GPT")
    get_all_event_properties_gpt(events_nl, config.DATA_DIR / "event_properties.json")
    events_nl_survey = ["You attended a meeting", "You bought a house", "You went camping", "You went to a concert", "You ate breakfast"]
    get_all_event_properties_gpt(events_nl_survey, config.DATA_EVALUATION_SURVEY_PATH / "event_properties.json")

    # print("\nTraining Event Adverbials, Embeddings, and Random Forest...")
    fit_event_adverbials(events)
    fit_event_specific_embeddings(events, events_nl)
    fit_event_specific_random_forest(events, events_nl)
    fit_event_specific_functions(events, events_nl,config.powerlaw)
    fit_event_specific_functions(events, events_nl, config.exp_decay)

    print("\nTraining Simple Models (Classifier, Regression)...")
    fit_classifier(events, events_nl)
    fit_regression(events, events_nl)


def make_predictions(event_nl, event_properties, adverbial, minutes_ago):
    """
    Make predictions using different models.
    """
    print("\nPredictions using Classifiers and Regression:")
    print(predict_adverbial_classifier(event_nl, minutes_ago, event_properties))
    print(predict_adverbial_regression(event_nl, minutes_ago, event_properties))

    print("\nPredictions using FuzzyLLI Configurations:")
    print(predict_time_frame_embedding(event_nl, adverbial))
    print(predict_adverbial_embedding(event_nl, minutes_ago))
    print(predict_time_frame_random_forest(event_properties, adverbial))
    print(predict_adverbial_random_forest(event_properties, minutes_ago))
    print(predict_adverbial_functions(event_properties, minutes_ago, config.powerlaw))
    # print(predict_adverbial_functions(event_properties, minutes_ago, config.exp_decay))


def evaluate_models_seen_events(events, events_nl, generate_new_predictions=False):
    """
    Evaluate the models by comparing the predicted labels on seen events
    """
    if generate_new_predictions:
        print("\nEvaluation Embeddings + Regressor:")
        get_predictions_embedding(events, events_nl)

        print("\nEvaluation Random Forest:")
        get_predictions_random_forest(events, events_nl)

        print("\nEvaluation Power Law:")
        get_predictions_functions(events, events_nl, function_to_use=config.powerlaw)
        # get_predictions_functions(events, events_nl, function_to_use=config.exp_decay)

        print("\nEvaluation Baseline Models (Classifier, Regression):")
        get_predictions_classifier(events, events_nl)
        get_predictions_regression(events, events_nl)

        # print("\n Evaluating GPT:")
        # evaluate_gpt(events, events_nl)

    # Calculate only the metrics
    print("\nCalculating the Evaluation metrics from the saved prediction files for the experimental data...")
    calculate_metrics(config.EVALUATION_FILE_PATH)



def evaluate_survey(events_to_fit, events_to_fit_nl, generate_new_predictions=False):
    """
    Evaluate the models by comparing the predicted adverbials for new events with the unseen_events results
    """
    if generate_new_predictions:
        print("\nEvaluating Random Forest on the unseen_events data:")
        evaluate_survey_random_forest(events_to_fit, events_to_fit_nl)

        print("\nEvaluating Embeddings + Regressor on the unseen_events data:")
        evaluate_survey_embedding(events_to_fit, events_to_fit_nl)

        print("\nEvaluation Power Law on the unseen survey data:")
        evaluate_survey_functions(events_to_fit, events_to_fit_nl, config.powerlaw)
        # evaluate_survey_functions(events_to_fit, events_to_fit_nl, config.exp_decay)

        print("\nEvaluating Classifier and Regression Model on the unseen_events data:")
        evaluate_survey_classifier(events_to_fit, events_to_fit_nl)
        evaluate_survey_regression(events_to_fit, events_to_fit_nl)

        # print("\nEvaluating GPT on the unseen events data:")
        # evaluate_survey_gpt()

    # Calculate only the metrics
    print("\nCalculating the Evaluation metrics from the saved prediction files for the survey data...")
    calculate_metrics(config.EVALUATION_SURVEY_FILE_PATH)

def evaluate_event_properties(events, events_nl): #
    """
    Evaluate all possible combinations of event properties by running the other two evaluations on each combination-
    At this timepoint this is done for every configuration. Comment parts in the other function out to
    evaluate only a specific configuration
    """
    # Evaluate all combinations of event properties
    all_props = ["Richness", "Frequency", "Importance"]
    train_models(events, events_nl)
    for r in range(1, len(all_props) + 1):
        for combo in combinations(all_props, r):
            config.properties_to_use = list(combo)
            evaluate_models_seen_events(events, events_nl, generate_new_predictions=True)
            evaluate_survey(events, events_nl, generate_new_predictions=True)


def plot_results(events, events_nl, adverbial, predict_function):
    """
    Plot the results for event adverbials.
    """
    print("Plotting results are saved under results/plots/...")
    print("Plotting Event Adverbials...")
    plot_all_persons_event_adverbials(events)
    for event_name, event_name_nl in zip(events, events_nl):
        print("Plotting single Event: ", event_name)
        plot_single_events(event_name, event_name_nl, predict_function)
    plot_events_adverbials(events, adverbial)


def _parse_event_properties(raw: str):
    """
    Parse event properties passed on the CLI.

    Accepts either a JSON object (single set of properties) or a JSON array of objects.
    """
    try:
        properties = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON for event properties: {exc}") from exc

    if isinstance(properties, dict):
        return [properties]
    if isinstance(properties, list) and all(isinstance(item, dict) for item in properties):
        return properties
    raise argparse.ArgumentTypeError("Event properties must be a JSON object or a list of JSON objects.")


def run_full_pipeline():
    """
    Preserve the previous default: train, evaluate, plot, and run a demo prediction.
    """
    train_models(DEFAULT_EVENTS, DEFAULT_EVENTS_NL)  # Trains FuzzyLLI in all variants and the baseline models
    evaluate_models_seen_events(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, generate_new_predictions=True)
    evaluate_survey(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, generate_new_predictions=True)
    evaluate_event_properties(DEFAULT_EVENTS, DEFAULT_EVENTS_NL)
    plot_results(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, "long time ago", predict_adverbial_random_forest)  # Plots FuzzyLLI
    make_predictions(DEFAULT_EVENT_NL, DEFAULT_EVENT_PROPERTIES, DEFAULT_ADVERBIAL, DEFAULT_MINUTES_AGO)


def build_parser():
    parser = argparse.ArgumentParser(description="FuzzyLLI command line interface.")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Train all models on the default events.")
    train_parser.set_defaults(func=lambda args: train_models(DEFAULT_EVENTS, DEFAULT_EVENTS_NL))

    # Evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate models on seen or unseen events.")
    evaluate_parser.add_argument(
        "--scope",
        choices=["seen", "survey", "properties"],
        default="seen",
        help="Which evaluation to run.",
    )
    evaluate_parser.add_argument(
        "--generate-new-predictions",
        action="store_true",
        help="Regenerate predictions before computing metrics.",
    )

    def _evaluate_cmd(args):
        if args.scope == "seen":
            evaluate_models_seen_events(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, generate_new_predictions=args.generate_new_predictions)
        elif args.scope == "survey":
            evaluate_survey(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, generate_new_predictions=args.generate_new_predictions)
        elif args.scope == "properties":
            evaluate_event_properties(DEFAULT_EVENTS, DEFAULT_EVENTS_NL)

    evaluate_parser.set_defaults(func=_evaluate_cmd)

    # Plot
    plot_parser = subparsers.add_parser("plot", help="Plot fitted functions and adverbials.")
    plot_parser.add_argument(
        "--adverbial",
        default="long time ago",
        help="Adverbial to highlight when plotting event adverbials.",
    )
    plot_parser.set_defaults(func=lambda args: plot_results(
        DEFAULT_EVENTS, DEFAULT_EVENTS_NL, args.adverbial, predict_adverbial_random_forest)
    )

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Make predictions for a given event.")
    predict_parser.add_argument("--event-nl", default=DEFAULT_EVENT_NL, help="Natural language description of the event.")
    predict_parser.add_argument("--adverbial", default=DEFAULT_ADVERBIAL, help="Adverbial used for time frame prediction.")
    predict_parser.add_argument("--minutes-ago", type=int, default=DEFAULT_MINUTES_AGO, help="Minutes ago to evaluate.")
    predict_parser.add_argument(
        "--properties",
        type=_parse_event_properties,
        default=None,
        help="JSON object or list of objects with Richness/Frequency/Importance.",
    )
    predict_parser.set_defaults(func=lambda args: make_predictions(
        args.event_nl,
        args.properties if args.properties is not None else DEFAULT_EVENT_PROPERTIES,
        args.adverbial,
        args.minutes_ago)
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # No sub-command provided -> run the full pipeline (previous default behaviour)
    if not getattr(args, "command", None):
        run_full_pipeline()
        return

    args.func(args)

if __name__ == '__main__':
    main()
