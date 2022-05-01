import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from src.random_forest.hybrid_random_forest import RandomForest
from sklearn.datasets import load_breast_cancer
import random

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


def prepare_dataset(name):
    if name == "breast_cancer":
        data = load_breast_cancer()
        X = data.data
        y = data.target
    elif name == "ionosphere":
        data = pd.read_csv("./data/ionosphere.data", header=None)
        data.drop(columns=[0, 1], inplace=True)
        X = np.array(data.drop(columns=[34]))
        y = data[34].apply(lambda x: 1 if x == "g" else 0)
    elif name == "biodegrataion":
        data = pd.read_csv("./data/biodeg.csv", sep=";", header=None)
        X = np.array(data.drop(columns=[41]))
        y = data[41].apply(lambda x: 1 if x == "RB" else 0)
    else:
        raise Exception("Wrong dataset name!")
    return X, y


def cross_validate(
    dataset_name,
    num_classifiers,
    tree_max_depth,
    tree_min_entropy_diff,
    tree_min_node_size,
    svm_lambda,
    # svm_mapper_params,
    svm_minimizer_params,
    n_folds,
    clf_max_num_attributes=None,
):
    X, y = prepare_dataset(dataset_name)

    splitted = list()
    dataset_indexes = list(range(X.shape[0]))
    fold_size = X.shape[0] // n_folds
    for i in range(n_folds):
        fold = list()
        for i in range(fold_size):
            idx = random.randrange(len(dataset_indexes))
            fold.append(dataset_indexes.pop(idx))
        splitted.append(fold)

    splitted = np.array(splitted)
    val_accs = []
    val_recalls = []
    val_precisions = []
    val_f1 = []
    for i in range(splitted.shape[0]):
        print(f"Training of fold nr {i+1}...")
        random_forest = RandomForest(
            num_classifiers=num_classifiers,
            tree_max_depth=tree_max_depth,
            tree_min_entropy_diff=tree_min_entropy_diff,
            tree_min_node_size=tree_min_node_size,
            svm_lambda=svm_lambda,
            # svm_mapper_params=svm_mapper_params,
            svm_minimizer_params=svm_minimizer_params,
            clf_max_num_attributes=clf_max_num_attributes,
        )
        val_split = splitted[i]
        train_split = [
            e
            for j in range(splitted.shape[0])
            for e in splitted[j]
            if j != i
        ]
        random_forest.fit(X[train_split], y[train_split])
        val_preds = random_forest.predict(X[val_split])
        val_accs.append(accuracy_score(y[val_split], val_preds))
        val_recalls.append(recall_score(y[val_split], val_preds))
        val_precisions.append(precision_score(y[val_split], val_preds))
        val_f1.append(f1_score(y[val_split], val_preds))

    return {
        "val_accuracy": np.mean(val_accs),
        "val_recall": np.mean(val_recalls),
        "val_precision": np.mean(val_precisions),
        "val_f1": np.mean(val_f1),
    }


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
