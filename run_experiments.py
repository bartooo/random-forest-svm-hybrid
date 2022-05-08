import logging
from utils import cross_validate
import numpy as np
import time
import pandas as pd

logging.basicConfig(
    filename=f"./logs/logs.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
logging.getLogger().addHandler(logging.StreamHandler())

N_RUNS = 25
N_FOLDS = 5
DATASETS = {"breast_cancer": 30, "ionosphere": 34, "biodegrataion": 41}
NUM_CLASSIFIERS = [2, 8, 14, 20]
TREE_MAX_DEPTHS = [2, 6, 10]
TREE_MIN_NODE_SIZES = [20, 60, 100]
TREE_MIN_ENTROPY_DIFFS = [0, 1e-2]
SVM_LAMBDAS = [0.001, 0.01, 0.1, 1]
SVM_BETA = 0.01
SVM_MIN_EPSILON = 1e-17
SVM_MAX_STEPS = 10000


def run():
    results_df = pd.DataFrame()
    for dataset in DATASETS.keys():
        for n_clf in NUM_CLASSIFIERS:
            for tree_max_depth in TREE_MAX_DEPTHS:
                for tree_min_entropy_diff in TREE_MIN_ENTROPY_DIFFS:
                    for tree_min_node_size in TREE_MIN_NODE_SIZES:
                        for svm_lambda in SVM_LAMBDAS:
                            for max_attr in np.arange(1, DATASETS[dataset], 8):
                                times = []
                                accs = []
                                recalls = []
                                precisions = []
                                f1s = []
                                
                                logging.info(
                                        f"Experiment parameters: [ds: {dataset}] [n_clf: {n_clf}] [tree_max_dpth: {tree_max_depth}] [tree_min_entrpy_diff: {tree_min_entropy_diff}] [tree_min_node_sz: {tree_min_node_size}] [svm_lambda: {svm_lambda}] [svm_beta: {SVM_BETA}] [svm_min_eps: {SVM_MIN_EPSILON}] [svm_max_steps: {SVM_MAX_STEPS}] [max_attr: {max_attr}]"
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
                                    recalls.append(results["val_recall"])
                                    precisions.append(results["val_precision"])
                                    f1s.append(results["val_f1"])
                                    logging.info(
                                        f"[{i}/{N_RUNS}] Acc: {results['val_accuracy']:.3f} Recall: {results['val_recall']:.3f} Precision:{results['val_precision']:.3f} F1: {results['val_f1']:.3f}"
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
                                                "std_time"
                                            ]
                                        ),
                                    ]
                                )
                                
                                logging.info(
                                    f"Mean acc: {np.mean(accs):.3f} Mean precision: {np.mean(precisions):.3f} Mean recall: {np.mean(recalls):.3f} Mean f1: {np.mean(f1s):.3f} Mean time: {np.mean(times)}\n"
                                )
                                results_df.to_csv('./logs/results.csv')

if __name__ == "__main__":
    run()