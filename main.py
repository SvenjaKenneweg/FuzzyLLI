# main.py
import argparse
import json
from itertools import combinations


from scripts.train import fit_object_adverbials
from scripts.plot import plot_spatial_fuzzylli, plot_objects_adverbials_fitted
from scripts.predictions import predict_adverbial_fuzzylli, predict_distance_fuzzylli
from scripts.evaluation_training_dataset import run_MAE_MdSE_evaluation
from scripts import config


DEFAULT_OBJECT_TYPE = [
    "Big", "Small"
]

def train_models(spatial_surveys):
    """
    Train the base FuzzyLLI model.
    """
    print("\nTraining FuzzyLLI for the spatial surveys")
    fit_object_adverbials(spatial_surveys)


def predict(object_type, adverbial=None, distance=None):
    """
    Make predictions with the base FuzzyLLI (without variants) for an adverbial or distance
    """
    if adverbial is None and distance is None:
        raise ValueError("Provide either 'adverbial' or 'distance'.")
    if distance is not None:
        print(f"Adverbials membership values for {object_type} and {distance} are: ",predict_adverbial_fuzzylli(object_type, distance))
    if adverbial is not None:
        print(f"Distance for {object_type} and {adverbial} is:", predict_distance_fuzzylli(object_type, adverbial))


def evaluate_models_training_dataset(spatial_surveys):
    """
    Evaluate the models by calculating the MAE and MdSE
    """
    print("\nCalculating MAE and MdSE:")
    print(run_MAE_MdSE_evaluation(spatial_surveys, None, predict_adverbial_fuzzylli))


def plot_results(objects, adverbial):
    """
    Plot the results for object types and adverbials.
    """
    print("Plotting results are saved under results/plots/... . ")
    print("Plotting the FuzzyLLI Overview Diagram (Left each object type, right the adverbials)")
    plot_spatial_fuzzylli(objects)
    plot_objects_adverbials_fitted(objects, adverbial)


def run_full_pipeline():
    """
    Preserve the previous default: train, evaluate, plot, and run a demo prediction.
    """
    train_models(DEFAULT_OBJECT_TYPE)  # Trains FuzzyLLI
    evaluate_models_training_dataset(DEFAULT_OBJECT_TYPE) # Calculates MAE and MdSE
    plot_results(DEFAULT_OBJECT_TYPE, "close to")  # Plots FuzzyLLI
    predict("Small", adverbial="close to", distance=20)


def build_parser():
    parser = argparse.ArgumentParser(description="FuzzyLLI command line interface.")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Train all models on the two default object types.")
    train_parser.set_defaults(func=lambda args: train_models(DEFAULT_OBJECT_TYPE))

    # Evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Calculate the MAE and MdSE")

    # Plot
    plot_parser = subparsers.add_parser("plot", help="Plot fitted functions and adverbials.")
    plot_parser.add_argument(
        "--adverbial",
        default="close to",
        help="Adverbial to highlight when plotting the course of object types x adverbial.",
    )
    plot_parser.set_defaults(func=lambda args: plot_results(
        DEFAULT_OBJECT_TYPE, args.adverbial)
    )

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Make predictions for a given object type.")
    predict_parser.add_argument("--object", default="Small", help="Object Type.")
    predict_parser.add_argument("--adverbial", default="close to", help="Adverbial used for spatial distance prediction.")
    predict_parser.add_argument("--distance", type=int, default=20, help="Distance ago to evaluate.")
    predict_parser.set_defaults(func=lambda args: predict(args.object, args.adverbial, args.distance))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # No sub-command provided -> run the full pipeline (previous default behaviour)
    if not getattr(args, "command", None):
        run_full_pipeline()
        return


if __name__ == '__main__':
    run_full_pipeline()
