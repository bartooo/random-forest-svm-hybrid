import numpy as np

class Node:
    def __init__(
        self,
        depth: int,
        split_attribute_idx: int = None,
        split_attribute_name: str = None,
        split_threshold: float = None,
        entropy: float = None,
        samples: int = None,
        values: np.ndarray = None,
        label: int = None,
        label_name: str = None,
    ) -> None:
        self.depth = depth
        self.split_attribute_idx = split_attribute_idx
        self.split_attribute_name = split_attribute_name
        self.split_threshold = split_threshold
        self.entropy = entropy
        self.samples = samples
        self.values = values
        self.label = label
        self.label_name = label_name
        self.right = None
        self.left = None

    def __str__(self):
        node_str = f"Depth: {self.depth}, "
        if self.split_attribute_name is not None:
            node_str += f"Split attribute: {self.split_attribute_name}, threshold: {self.split_threshold}, "
        elif self.split_attribute_idx is not None:
            node_str += f"Split attribute index: {self.split_attribute_idx}, threshold: {self.split_threshold}, "
        if self.label_name is not None:
            node_str += f"entropy: {self.entropy:.3f}, samples: {self.samples}, values: {self.values}, label: {self.label_name}"
        else:
            node_str += f"entropy: {self.entropy:.3f}, samples: {self.samples}, values: {self.values}, label: {self.label}"
        return node_str
