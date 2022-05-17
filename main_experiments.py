"""
Authors: Bartosz Cywiński, Łukasz Staniszewski
"""
from experiments.run_experiments import run_parameters, run_models
import argparse


def main():
    parser = argparse.ArgumentParser(
        description=("Experiments performer for SVM with Decision Tree hybrid.")
    )
    parser.add_argument(
        "-WHAT",
        choices=["parameters", "models"],
        help="Specifies which experiment will take place",
        required=True,
    )
    args = parser.parse_args()
    if args.WHAT == "parameters":
        run_parameters()
    elif args.WHAT == "models":
        run_models()
    else:
        raise Exception("Wrong value for parameter WHAT.")


if __name__ == "__main__":
    main()
