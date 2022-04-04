import numpy as np
from ..SVM.SVM import SVM
from .decision_tree import DecisionTree

class RandomForest:
    def __init__(
        self,
        num_classifiers: int,
        tree_max_depth: int,
        tree_min_entropy_diff: float,
        tree_min_node_size: int,
        tree_max_num_attributes: int = None,
    ):
        self.tree_max_depth = tree_max_depth
        self.tree_min_entropy_diff = tree_min_entropy_diff
        self.tree_min_node_size = tree_min_node_size
        self.tree_max_num_attributes = tree_max_num_attributes
        self.num_classifiers = num_classifiers
        self.classifiers = []

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
        if self.tree_max_num_attributes > dataset.shape[1] - 1:
            raise Exception("Invalid number of max attributes in Decision Tree")

        for i in range(self.num_classifiers):
            boostrapped_dataset = self._bootstrap_dataset(dataset)
            if i % 2 == 0:
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
                    lambd=1, minimizer_params={"beta": 0.01, "min_epsilon": 1e-20}
                )
                svm.fit(boostrapped_dataset[:, :-1], boostrapped_dataset[:, -1])
                self.classifiers.append(svm)

    def _predict_sample(self, sample: np.ndarray) -> int:
        map_predictions = {1: 1, 0: 0, -1: 1}
        return np.bincount(
            np.squeeze(
                [
                    map_predictions[prediction]
                    for classifier in self.classifiers
                    for prediction in classifier.predict(sample)
                ]
            )
        ).argmax()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        if len(self.classifiers) > 0:
            return np.array([self._predict_sample(sample) for sample in X])
        else:
            raise Exception("Random Forest is not trained yet")