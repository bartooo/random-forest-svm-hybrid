import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from src.random_forest.hybrid_random_forest import RandomForest
from sklearn.datasets import load_breast_cancer
import random

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
        val_recalls.append(recall_score(y[val_split], val_preds, zero_division=0))
        val_precisions.append(precision_score(y[val_split], val_preds, zero_division=0))
        val_f1.append(f1_score(y[val_split], val_preds, zero_division=0))

    return {
        "val_accuracy": np.mean(val_accs),
        "val_recall": np.mean(val_recalls),
        "val_precision": np.mean(val_precisions),
        "val_f1": np.mean(val_f1),
    }