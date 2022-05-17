import numpy as np
from typing import Tuple
from .node import Node


class DecisionTree:
    def __init__(
        self,
        max_depth: int,
        min_entropy_diff: float,
        min_node_size: int,
        max_num_attributes: int = None,
    ) -> None:
        self.root = None
        self.max_depth = max_depth
        self.min_entropy_diff = min_entropy_diff
        self.min_node_size = min_node_size
        self.max_num_attributes = max_num_attributes

    def _entropy(self, s: np.ndarray) -> float:
        """Calculates entropy of current dataset s
        according to formula:
        H(s) = sum(-p(x)*log2(p(x))),
        where p(x) is the proportion of the number of elements
        in class x to the number of elements in set s

        Args:
            s (np.ndarray): current dataset to calculate entropy on

        Returns:
            float: entropy of dataset s
        """
        y = s[:, -1]
        result = 0
        counts = np.unique(y, return_counts=True)[1]
        for count in counts:
            proportion = count / y.shape[0]
            result += -proportion * np.log2(proportion + 1e-5)
        return result

    def _find_split(
        self, s: np.ndarray
    ) -> Tuple[int, float, float, np.ndarray, np.ndarray]:
        """Finds best attribute and threshold to split node on

        Args:
            s (np.ndarray): current dataset of node

        Returns:
            Tuple[int, float, float, np.ndarray, np.ndarray]: split attribute index, threshold, split entropy, left subset, right subset
        """
        min_split_entropy = np.inf
        best_split_attr_idx = None
        best_threshold = None
        best_left_subset = None
        best_right_subset = None

        possible_attributes = (
            range(s.shape[1] - 1)
            if self.max_num_attributes is None
            else np.random.choice(
                range(s.shape[1] - 1), size=self.max_num_attributes, replace=False
            )
        )
        for attribute_idx in possible_attributes:
            for threshold in np.unique(s[:, attribute_idx]):

                left_subset = np.array(
                    [row for row in s if row[attribute_idx] < threshold]
                )
                right_subset = np.array(
                    [row for row in s if row[attribute_idx] >= threshold]
                )

                left_entropy = (
                    (len(left_subset) / len(s)) * self._entropy(left_subset)
                    if left_subset.shape[0] > 0
                    else 0
                )
                right_entropy = (
                    (len(right_subset) / len(s)) * self._entropy(right_subset)
                    if right_subset.shape[0] > 0
                    else 0
                )

                split_entropy = left_entropy + right_entropy
                if split_entropy < min_split_entropy:
                    min_split_entropy = split_entropy
                    best_split_attr_idx = attribute_idx
                    best_threshold = threshold
                    best_left_subset = left_subset
                    best_right_subset = right_subset

        return (
            best_split_attr_idx,
            best_threshold,
            min_split_entropy,
            best_left_subset,
            best_right_subset,
        )

    def _build_id3(
        self,
        dataset: np.ndarray,
        depth: int,
        orig_classes: np.ndarray,
        attribute_labels: np.ndarray = None,
        class_labels: np.ndarray = None,
    ) -> Node:
        """Builds decision tree according to ID3 algorithm

        Args:
            dataset (np.ndarray): Dataset from which tree will be built
            depth (int): Current depth of the tree
            orig_classes (np.ndarray): All classes in original dataset
            attribute_labels (np.ndarray, optional): Labels of attributes in dataset. Defaults to None.
            class_labels (np.ndarray, optional): Labels of classes in dataset. Defaults to None.

        Returns:
            Node: Next created node of the tree
        """
        if dataset.shape[0] == 0:
            return None

        X, y = dataset[:, :-1], dataset[:, -1].astype("int64")

        # all examples classified as one class
        if np.unique(y).shape[0] == 1:
            return Node(
                depth=depth,
                values=[y.tolist().count(c) for c in orig_classes],
                entropy=0.0,
                label=np.argmax([y.tolist().count(c) for c in orig_classes]),
                label_name=class_labels[
                    np.argmax([y.tolist().count(c) for c in orig_classes])
                ]
                if class_labels is not None
                else np.argmax([y.tolist().count(c) for c in orig_classes]),
                samples=dataset.shape[0],
            )

        # no attributes to split upon
        if X.shape[1] == 0:
            return Node(
                depth=depth,
                values=[y.tolist().count(c) for c in orig_classes],
                entropy=self._entropy(dataset),
                label=np.argmax([y.tolist().count(c) for c in orig_classes]),
                label_name=class_labels[
                    np.argmax([y.tolist().count(c) for c in orig_classes])
                ]
                if class_labels is not None
                else np.argmax([y.tolist().count(c) for c in orig_classes]),
                samples=dataset.shape[0],
            )

        (
            best_split_attr_idx,
            best_threshold,
            min_split_entropy,
            best_left_subset,
            best_right_subset,
        ) = self._find_split(dataset)

        # decide about splitting
        if (
            depth < self.max_depth
            and (self._entropy(dataset) - min_split_entropy) > self.min_entropy_diff
            and (best_left_subset.shape[0] > self.min_node_size)
            and (best_right_subset.shape[0] > self.min_node_size)
        ):
            root = Node(
                depth=depth,
                split_attribute_idx=best_split_attr_idx,
                split_attribute_name=attribute_labels[best_split_attr_idx]
                if attribute_labels is not None
                else None,
                split_threshold=best_threshold,
                entropy=self._entropy(dataset),
                samples=dataset.shape[0],
                values=[y.tolist().count(c) for c in orig_classes],
                label=np.argmax([y.tolist().count(c) for c in orig_classes]),
                label_name=class_labels[
                    np.argmax([y.tolist().count(c) for c in orig_classes])
                ]
                if class_labels is not None
                else np.argmax([y.tolist().count(c) for c in orig_classes]),
            )
            root.left = self._build_id3(
                best_left_subset,
                depth=depth + 1,
                orig_classes=orig_classes,
                attribute_labels=attribute_labels,
                class_labels=class_labels,
            )
            root.right = self._build_id3(
                best_right_subset,
                depth=depth + 1,
                orig_classes=orig_classes,
                attribute_labels=attribute_labels,
                class_labels=class_labels,
            )

        else:
            root = Node(
                depth=depth,
                entropy=self._entropy(dataset),
                samples=dataset.shape[0],
                values=[y.tolist().count(c) for c in orig_classes],
                label=np.argmax([y.tolist().count(c) for c in orig_classes]),
                label_name=class_labels[
                    np.argmax([y.tolist().count(c) for c in orig_classes])
                ]
                if class_labels is not None
                else np.argmax([y.tolist().count(c) for c in orig_classes]),
            )

        return root

    def visualize(self) -> None:
        queue = list()
        queue.append(self.root)

        while queue:
            v = queue.pop(0)
            print(v)
            if v.left is not None:
                queue.append(v.left)
            if v.left is not None:
                queue.append(v.right)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attribute_labels: np.ndarray = None,
        class_labels: np.ndarray = None,
    ) -> None:
        if attribute_labels is not None:
            if attribute_labels.shape[0] != X.shape[1]:
                raise Exception("Invalid shape of given attribute labels")

        if class_labels is not None:
            if class_labels.shape[0] != np.unique(y).shape[0]:
                raise Exception("Invalid shape of given class labels")

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        dataset = np.concatenate([X, y], axis=1)
        self.root = self._build_id3(
            dataset,
            depth=0,
            orig_classes=np.unique(y),
            attribute_labels=attribute_labels,
            class_labels=class_labels,
        )

    def _predict_sample(self, sample: np.ndarray) -> int:
        current_node = self.root
        current_prediction = current_node.label
        while current_node.split_attribute_idx is not None:
            if sample[current_node.split_attribute_idx] < current_node.split_threshold:
                current_prediction = current_node.left.label
                current_node = current_node.left
            else:
                current_prediction = current_node.right.label
                current_node = current_node.right

        return current_prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        if self.root is not None:
            return np.array([self._predict_sample(sample) for sample in X])
        else:
            raise Exception("Decision Tree is not trained yet")
