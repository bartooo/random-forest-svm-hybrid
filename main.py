import argparse
from utils import cross_validate

# Examplary running command:
# python .\main.py --dataset breast_cancer --num_classifiers 2 --n_folds 5

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="breast_cancer",
    help="Dataset to train on",
)
parser.add_argument(
    "--n_folds",
    type=int,
    default=10,
    help="Number of folds used in k-cross validation",
)
parser.add_argument(
    "--num_classifiers",
    type=int,
    default=10,
    help="Number of classifiers that will be used in random forest",
)
parser.add_argument(
    "--tree_max_depth",
    type=int,
    default=3,
    help="Maximal possible depth of single decision tree",
)
parser.add_argument(
    "--tree_min_entropy_diff",
    type=float,
    default=1e-2,
    help=(
        "Minimal difference between entropy value before and after"
        " split of a node"
    ),
)
parser.add_argument(
    "--tree_min_node_size",
    type=int,
    default=30,
    help="Minimal size of a single node in decision tree",
)
parser.add_argument(
    "--svm_lambda",
    type=float,
    default=0.05,
    help=(
        "Degree of importance that is given to misclassifications in"
        " SVM classifier"
    ),
)
parser.add_argument(
    "--svm_beta",
    type=float,
    default=0.01,
    help="Learning rate for SGD optimizer in SVM",
)
parser.add_argument(
    "--svm_min_epsilon",
    type=float,
    default=1e-17,
    help=(
        "Stop criterion as minimal gradient norm for SGD optimizer"
        " in SVM"
    ),
)
parser.add_argument(
    "--svm_max_steps",
    type=int,
    default=10000,
    help=(
        "Stop criterion as maximal iterationsd for SGD optimizer in SVM"
    ),
)
parser.add_argument(
    "--clf_max_num_attributes",
    type=int,
    default=None,
    help=(
        "Maximal number of attributes in dataset to consider in"
        " splitting single node of a tree/ to train svm on"
    ),
)


if __name__ == "__main__":
    args = parser.parse_args()
    results = cross_validate(
        dataset_name=args.dataset,
        n_folds=args.n_folds,
        num_classifiers=args.num_classifiers,
        tree_max_depth=args.tree_max_depth,
        tree_min_entropy_diff=args.tree_min_entropy_diff,
        tree_min_node_size=args.tree_min_node_size,
        svm_lambda=args.svm_lambda,
        svm_minimizer_params={
            "beta": args.svm_beta,
            "min_epsilon": args.svm_min_epsilon,
            "max_steps": args.svm_max_steps,
        },
        clf_max_num_attributes=args.clf_max_num_attributes,
    )

    print("Results:")
    print(f"Validation accuracy: {results['val_accuracy']:.3f}")
    print(f"Validation recall: {results['val_recall']:.3f}")
    print(f"Validation precision: {results['val_precision']:.3f}")
    print(f"Validation f1: {results['val_f1']:.3f}")
