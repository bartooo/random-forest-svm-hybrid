import logging
from experiments.experiments_utils import cross_validate
import numpy as np
import time
import pandas as pd

logging.basicConfig(
    filename=f"./experiments/logs/logs.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="   %H:%M:%S",
    level=logging.DEBUG,
)
logging.getLogger().addHandler(logging.StreamHandler())

N_RUNS = 25
N_FOLDS = 5
DATASETS = {"breast_cancer": 30, "ionosphere": 34, "biodegrataion": 41}


def run_models():
    MODELS = ["svm", "tree", "hybrid", "hybrid", "hybrid"]
    ONLY_SVM = [True, False, False, True, False]
    ONLY_TREE = [False, True, False, False, True]
    NUM_CLASSIFIERS = 14
    MAX_ATTRS = [None, None, 16, 16, 16]
    TREE_MAX_DEPTH = 6
    TREE_MIN_ENTROPY_DIFF = 0.01
    TREE_MIN_NODE_SIZE = 40
    SVM_LAMBDA = 0.005
    SVM_BETA = 0.01
    SVM_MIN_EPSILON = 1e-17
    SVM_MAX_STEPS = 10000
    results_df = pd.DataFrame()
    for dataset in DATASETS.keys():
        for model, only_svm_param, only_tree_param, max_attr in zip(
            MODELS, ONLY_SVM, ONLY_TREE, MAX_ATTRS
        ):
            times = []
            accs = []
            recalls = []
            precisions = []
            f1s = []

            name_of_model = model
            if model == "hybrid":
                if only_svm_param:
                    name_of_model += "_only_svm"
                elif only_tree_param:
                    name_of_model += "_only_tree"

            logging.info(
                "Experiment parameters: [name:"
                f" {name_of_model}][ds:"
                f" {dataset}] [n_clf: {NUM_CLASSIFIERS}]"
                " [tree_max_dpth:"
                f" {TREE_MAX_DEPTH}]"
                " [tree_min_entrpy_diff:"
                f" {TREE_MIN_ENTROPY_DIFF}]"
                " [tree_min_node_sz:"
                f" {TREE_MIN_NODE_SIZE}]"
                f" [svm_lambda: {SVM_LAMBDA}]"
                f" [svm_beta: {SVM_BETA}]"
                f" [svm_min_eps: {SVM_MIN_EPSILON}]"
                f" [svm_max_steps: {SVM_MAX_STEPS}]"
                f" [max_attr: {max_attr}]"
            )
            for i in range(N_RUNS):
                start = time.time()
                results = cross_validate(
                    dataset_name=dataset,
                    n_folds=N_FOLDS,
                    num_classifiers=NUM_CLASSIFIERS,
                    tree_max_depth=TREE_MAX_DEPTH,
                    tree_min_entropy_diff=TREE_MIN_ENTROPY_DIFF,
                    tree_min_node_size=TREE_MIN_NODE_SIZE,
                    svm_lambda=SVM_LAMBDA,
                    svm_minimizer_params={
                        "beta": SVM_BETA,
                        "min_epsilon": SVM_MIN_EPSILON,
                        "max_steps": SVM_MAX_STEPS,
                    },
                    clf_max_num_attributes=max_attr,
                    which_model=model,
                    is_only_svm=only_svm_param,
                    is_only_tree=only_tree_param,
                )
                end = time.time()
                times.append(end - start)
                accs.append(results["val_accuracy"])
                recalls.append(results["val_recall"])
                precisions.append(results["val_precision"])
                f1s.append(results["val_f1"])
                logging.info(
                    f"[{i+1}/{N_RUNS}] Acc:"
                    f" {results['val_accuracy']:.3f} Recall:"
                    f" {results['val_recall']:.3f} Precision:{results['val_precision']:.3f} F1:"
                    f" {results['val_f1']:.3f}"
                )

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        [
                            [
                                dataset,
                                name_of_model,
                                NUM_CLASSIFIERS,
                                TREE_MAX_DEPTH,
                                TREE_MIN_ENTROPY_DIFF,
                                TREE_MIN_NODE_SIZE,
                                SVM_LAMBDA,
                                max_attr,
                                np.mean(accs),
                                np.std(accs),
                                np.min(accs),
                                np.max(accs),
                                np.mean(recalls),
                                np.std(recalls),
                                np.min(recalls),
                                np.max(recalls),
                                np.mean(precisions),
                                np.std(precisions),
                                np.min(precisions),
                                np.max(precisions),
                                np.mean(f1s),
                                np.std(f1s),
                                np.min(f1s),
                                np.max(f1s),
                                np.mean(times),
                                np.std(times),
                            ]
                        ],
                        columns=[
                            "dataset",
                            "model_name",
                            "n_clf",
                            "tree_max_depth",
                            "tree_min_entropy_diff",
                            "tree_min_node_size",
                            "svm_lambda",
                            "max_attr",
                            "mean_acc",
                            "std_acc",
                            "min_acc",
                            "max_acc",
                            "mean_recall",
                            "std_recall",
                            "min_recall",
                            "max_recall",
                            "mean_precision",
                            "std_precision",
                            "min_precision",
                            "max_precision",
                            "mean_f1",
                            "std_f1",
                            "min_f1",
                            "max_f1",
                            "mean_time",
                            "std_time",
                        ],
                    ),
                ]
            )

            logging.info(
                "Mean acc:"
                f" {np.mean(accs):.3f} Mean"
                " precision:"
                f" {np.mean(precisions):.3f} Mean"
                " recall:"
                f" {np.mean(recalls):.3f} Mean f1:"
                f" {np.mean(f1s):.3f} Mean time:"
                f" {np.mean(times)}\n"
            )
            results_df.to_csv("./experiments/logs/results.csv")


def run_parameters():
    NUM_CLASSIFIERS = [2, 8, 14, 20]
    TREE_MAX_DEPTHS = [2, 6, 10]
    TREE_MIN_NODE_SIZES = [20, 60, 100]
    TREE_MIN_ENTROPY_DIFFS = [0, 1e-2]
    SVM_LAMBDAS = [0.001, 0.01, 0.1, 1]
    SVM_BETA = 0.01
    SVM_MIN_EPSILON = 1e-17
    SVM_MAX_STEPS = 10000
    results_df = pd.DataFrame()
    for dataset in DATASETS.keys():
        for n_clf in NUM_CLASSIFIERS:
            for tree_max_depth in TREE_MAX_DEPTHS:
                for tree_min_entropy_diff in TREE_MIN_ENTROPY_DIFFS:
                    for tree_min_node_size in TREE_MIN_NODE_SIZES:
                        for svm_lambda in SVM_LAMBDAS:
                            for max_attr in np.arange(
                                1, DATASETS[dataset], 8
                            ):
                                times = []
                                accs = []
                                recalls = []
                                precisions = []
                                f1s = []

                                logging.info(
                                    "Experiment parameters: [ds:"
                                    f" {dataset}] [n_clf: {n_clf}]"
                                    " [tree_max_dpth:"
                                    f" {tree_max_depth}]"
                                    " [tree_min_entrpy_diff:"
                                    f" {tree_min_entropy_diff}]"
                                    " [tree_min_node_sz:"
                                    f" {tree_min_node_size}]"
                                    f" [svm_lambda: {svm_lambda}]"
                                    f" [svm_beta: {SVM_BETA}]"
                                    f" [svm_min_eps: {SVM_MIN_EPSILON}]"
                                    f" [svm_max_steps: {SVM_MAX_STEPS}]"
                                    f" [max_attr: {max_attr}]"
                                )

                                for i in range(N_RUNS):

                                    start = time.time()
                                    results = cross_validate(
                                        dataset_name=dataset,
                                        n_folds=N_FOLDS,
                                        num_classifiers=n_clf,
                                        tree_max_depth=tree_max_depth,
                                        tree_min_entropy_diff=tree_min_entropy_diff,
                                        tree_min_node_size=tree_min_node_size,
                                        svm_lambda=svm_lambda,
                                        svm_minimizer_params={
                                            "beta": SVM_BETA,
                                            "min_epsilon": SVM_MIN_EPSILON,
                                            "max_steps": SVM_MAX_STEPS,
                                        },
                                        clf_max_num_attributes=max_attr,
                                    )
                                    end = time.time()
                                    times.append(end - start)
                                    accs.append(results["val_accuracy"])
                                    recalls.append(
                                        results["val_recall"]
                                    )
                                    precisions.append(
                                        results["val_precision"]
                                    )
                                    f1s.append(results["val_f1"])
                                    logging.info(
                                        f"[{i+1}/{N_RUNS}] Acc:"
                                        f" {results['val_accuracy']:.3f} Recall:"
                                        f" {results['val_recall']:.3f} Precision:{results['val_precision']:.3f} F1:"
                                        f" {results['val_f1']:.3f}"
                                    )

                                results_df = pd.concat(
                                    [
                                        results_df,
                                        pd.DataFrame(
                                            [
                                                [
                                                    dataset,
                                                    n_clf,
                                                    tree_max_depth,
                                                    tree_min_entropy_diff,
                                                    tree_min_node_size,
                                                    svm_lambda,
                                                    max_attr,
                                                    np.mean(accs),
                                                    np.std(accs),
                                                    np.mean(recalls),
                                                    np.std(recalls),
                                                    np.mean(precisions),
                                                    np.std(precisions),
                                                    np.mean(f1s),
                                                    np.std(f1s),
                                                    np.mean(times),
                                                    np.std(times),
                                                ]
                                            ],
                                            columns=[
                                                "dataset",
                                                "n_clf",
                                                "tree_max_depth",
                                                "tree_min_entropy_diff",
                                                "tree_min_node_size",
                                                "svm_lambda",
                                                "max_attr",
                                                "mean_acc",
                                                "std_acc",
                                                "mean_recall",
                                                "std_recall",
                                                "mean_precision",
                                                "std_precision",
                                                "mean_f1",
                                                "std_f1",
                                                "mean_time",
                                                "std_time",
                                            ],
                                        ),
                                    ]
                                )

                                logging.info(
                                    "Mean acc:"
                                    f" {np.mean(accs):.3f} Mean"
                                    " precision:"
                                    f" {np.mean(precisions):.3f} Mean"
                                    " recall:"
                                    f" {np.mean(recalls):.3f} Mean f1:"
                                    f" {np.mean(f1s):.3f} Mean time:"
                                    f" {np.mean(times)}\n"
                                )
                                results_df.to_csv(
                                    "./experiments/logs/results.csv"
                                )
