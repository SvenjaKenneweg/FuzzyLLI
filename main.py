# src/main.py
import argparse
import json
from itertools import combinations

from src.train import (fit_event_adverbials, fit_event_specific_embeddings, fit_event_specific_random_forest, fit_event_specific_functions)
from src.plot import plot_all_persons_event_adverbials, plot_single_events, plot_events_adverbials
from src.predictions import (predict_time_frame_embedding, predict_adverbial_embedding, predict_adverbial_functions,
                             predict_time_frame_random_forest, predict_adverbial_random_forest, get_all_event_properties_gpt)
from src.evaluation_training_dataset import (get_predictions_classifier, get_predictions_regression,
                                             get_predictions_embedding, get_predictions_random_forest,
                                             get_predictions_functions,
                                             evaluate_gpt, calculate_metrics)
from src.evaluation_test_dataset import (evaluate_test_data_random_forest, evaluate_test_data_functions,
                                         evaluate_test_data_embedding, evaluate_test_data_gpt, evaluate_test_data_regression, evaluate_test_data_classifier)
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
    print("\nGet the event properties using GPT for the training dataset (saved under datasets/training/event_properties.json)")
    # get_all_event_properties_gpt(events_nl, config.DATA_DIR / "event_properties.json")
    print("\nGet the event properties using GPT for the training dataset (saved under datasets/test/event_properties_1/2.json)")
    events_nl_test_1 = ["You attended a meeting", "You bought a house", "You went camping", "You went to a concert", "You ate breakfast"]
    events_nl_test_2 = ["You had a job interview", "You had a dental checkup", "You took a long-haul flight", "You prepared dinner", "You visited a museum", "You did laundry"]
    # get_all_event_properties_gpt(events_nl_test_1, config.DATASET_TEST_PATH / "event_properties_1.json")
    get_all_event_properties_gpt(events_nl_test_2, config.DATASET_TEST_PATH / "event_properties_2.json")

    # print("\nTraining FuzzyLLI in all three configurations (Power Law, Random Forest, Word Embeddings)...")
    # fit_event_adverbials(events)
    # fit_event_specific_embeddings(events, events_nl)
    # fit_event_specific_random_forest(events, events_nl)
    # fit_event_specific_functions(events, events_nl,config.powerlaw)


def make_predictions(event_nl, event_properties, adverbial, minutes_ago):
    """
    Make predictions using different models.
    """
    print("\nPredict the interval where the event (threshold = 0.6) has taken place and the adverbials"
          " memberships using Word Embeddings:")
    tf_emb = predict_time_frame_embedding(event_nl, adverbial)
    print(f"Event: '{event_nl}' | Adverbial: '{adverbial}' -> Predicted interval (minutes ago): {tf_emb}")
    adv_emb = predict_adverbial_embedding(event_nl, minutes_ago)
    print(f"Event: '{event_nl}' | Minutes ago: {minutes_ago} -> Adverbial memberships: {adv_emb}")

    print("\nPredict the interval where the event (threshold = 0.6) has taken place and the adverbials"
          " memberships using Random Forest:")
    tf_rf = predict_time_frame_random_forest(event_properties, adverbial)
    print(f"Properties: {event_properties} | Adverbial: '{adverbial}' -> Predicted interval (minutes ago): {tf_rf}")
    adv_rf = predict_adverbial_random_forest(event_properties, minutes_ago)
    print(f"Properties: {event_properties} | Minutes ago: {minutes_ago} -> Adverbial memberships: {adv_rf}")

    print("\nPredict the adverbials memberships using PowerLaw:")
    adv_func = predict_adverbial_functions(event_properties, minutes_ago, config.powerlaw)
    print(f"Properties: {event_properties} | Minutes ago: {minutes_ago} -> Adverbial memberships: {adv_func}")


def evaluate_models_training_dataset(events, events_nl, generate_new_predictions=False):
    """
    Evaluate the models by comparing the predicted labels on training dataset (leave-one-out)
    """
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



def evaluate_models_test_dataset(events_to_fit, events_to_fit_nl, generate_new_predictions=False):
    """
    Evaluate the models by comparing the predicted adverbials for new events with the results of the test dataset
    """
    if generate_new_predictions:
        print("\nEvaluating Random Forest on the test dataset:")
        evaluate_test_data_random_forest(events_to_fit, events_to_fit_nl)

        print("\nEvaluating Embeddings + Regressor on the test dataset:")
        evaluate_test_data_embedding(events_to_fit, events_to_fit_nl)

        print("\nEvaluation Power Law on the test dataset:")
        evaluate_test_data_functions(events_to_fit, events_to_fit_nl, config.powerlaw)

    # Calculate only the metrics
    print("\nCalculating the Evaluation metrics from the saved prediction files for the test dataset...")
    calculate_metrics(config.EVALUATION_TEST_DATASET_FILE_PATH)


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
            evaluate_models_training_dataset(events, events_nl, generate_new_predictions=True)
            evaluate_models_test_dataset(events, events_nl, generate_new_predictions=True)


def plot_results(events, events_nl, adverbial, predict_function):
    """
    Plot the results for event adverbials.
    """
    print("Plotting results are saved under results/plots/... . "
          "The used FuzzyLLI configuration for the plotting is Random Forest")
    print("\nPlotting the FuzzyLLI Overview Diagram (Left each event, right the adverbials")
    plot_all_persons_event_adverbials(events)
    print("\nPlotting the course of single events after fitting")
    for event_name, event_name_nl in zip(events, events_nl):
        plot_single_events(event_name, event_name_nl, predict_function)
    print("\nPlot only the median membership values of the given adverbial for the given events")
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
    # evaluate_models_training_dataset(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, generate_new_predictions=True)
    # evaluate_models_test_dataset(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, generate_new_predictions=True)
    # # evaluate_event_properties(DEFAULT_EVENTS, DEFAULT_EVENTS_NL)
    # plot_results(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, "long time ago", predict_adverbial_random_forest)  # Plots FuzzyLLI
    # make_predictions(DEFAULT_EVENT_NL, DEFAULT_EVENT_PROPERTIES, DEFAULT_ADVERBIAL, DEFAULT_MINUTES_AGO)


def build_parser():
    parser = argparse.ArgumentParser(description="FuzzyLLI command line interface.")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Train all models on the default events.")
    train_parser.set_defaults(func=lambda args: train_models(DEFAULT_EVENTS, DEFAULT_EVENTS_NL))

    # Evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate models on training (leave-one-out) or test dataset.")
    evaluate_parser.add_argument(
        "--scope",
        choices=["training", "test"],
        default="test",
        help="Which evaluation to run.",
    )
    evaluate_parser.add_argument(
        "--generate-new-predictions",
        action="store_true",
        help="Regenerate predictions before computing metrics.",
    )

    def _evaluate_cmd(args):
        if args.scope == "training":
            evaluate_models_training_dataset(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, generate_new_predictions=args.generate_new_predictions)
        elif args.scope == "test":
            evaluate_models_test_dataset(DEFAULT_EVENTS, DEFAULT_EVENTS_NL, generate_new_predictions=args.generate_new_predictions)

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
