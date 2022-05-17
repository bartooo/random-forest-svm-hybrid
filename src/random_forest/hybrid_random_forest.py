import numpy as np
from sklearn import tree
from src.SVM.SVM import SVM
from .decision_tree import DecisionTree
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.utils.multiclass import unique_labels


class RandomForest:
    def __init__(
        self,
        num_classifiers: int,
        tree_max_depth: int,
        tree_min_entropy_diff: float,
        tree_min_node_size: int,
        svm_lambda: float,
        is_only_svm: bool = False,
        is_only_tree: bool = False,
        svm_mapper_params: dict = {},
        svm_minimizer_params: dict = {},
        clf_max_num_attributes: int = None,
    ):
        self.tree_max_depth = tree_max_depth
        self.tree_min_entropy_diff = tree_min_entropy_diff
        self.tree_min_node_size = tree_min_node_size
        self.tree_max_num_attributes = clf_max_num_attributes
        self.num_classifiers = num_classifiers
        self.svm_lambda = svm_lambda
        self.is_only_svm = is_only_svm
        self.is_only_tree = is_only_tree
        self.classifiers = []
        self.svm_mapper_params = svm_mapper_params
        self.svm_minimizer_params = svm_minimizer_params

    def _bootstrap_dataset(self, dataset: np.ndarray):
        chosen_idxs = np.random.choice(
            dataset.shape[0], size=dataset.shape[0], replace=True
        )
        return dataset[chosen_idxs]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attribute_labels: np.ndarray = None,
        class_labels: np.ndarray = None,
    ) -> None:
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        dataset = np.concatenate([X, y], axis=1)
        if (
            self.tree_max_num_attributes is not None
            and self.tree_max_num_attributes > dataset.shape[1] - 1
        ):
            raise Exception(
                "Invalid number of max attributes in Decision Tree"
            )
        for i in range(self.num_classifiers):
            print(
                f"[{i+1}/{self.num_classifiers}] Training classifier..."
            )
            boostrapped_dataset = self._bootstrap_dataset(dataset)
            if (
                i % 2 == 0 or self.is_only_tree
            ) and not self.is_only_svm:
                tree = DecisionTree(
                    max_depth=self.tree_max_depth,
                    min_entropy_diff=self.tree_min_entropy_diff,
                    min_node_size=self.tree_min_node_size,
                    max_num_attributes=boostrapped_dataset.shape[1] - 1
                    if self.tree_max_num_attributes is None
                    else self.tree_max_num_attributes,
                )
                tree.fit(
                    boostrapped_dataset[:, :-1],
                    boostrapped_dataset[:, -1],
                    attribute_labels=attribute_labels,
                    class_labels=class_labels,
                )
                self.classifiers.append(tree)
            else:
                svm = SVM(
                    lambd=self.svm_lambda,
                    minimizer_params=self.svm_minimizer_params,
                    mapper_params=self.svm_mapper_params,
                    rf_max_attributes=boostrapped_dataset.shape[1] - 1
                    if self.tree_max_num_attributes is None
                    else self.tree_max_num_attributes,
                )
                svm.fit(
                    boostrapped_dataset[:, :-1],
                    boostrapped_dataset[:, -1],
                )
                self.classifiers.append(svm)

    def _predict_sample(self, sample: np.ndarray) -> int:
        return np.bincount(
            np.squeeze(
                [
                    classifier.predict(sample).squeeze()
                    for classifier in self.classifiers
                ]
            )
        ).argmax()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        if len(self.classifiers) > 0:
            return np.array(
                [self._predict_sample(sample) for sample in X]
            )
        else:
            raise Exception("Random Forest is not trained yet")
