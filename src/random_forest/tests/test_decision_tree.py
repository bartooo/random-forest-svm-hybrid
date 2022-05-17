"""
Authors: Bartosz Cywiński, Łukasz Staniszewski
"""
import numpy as np
from src.random_forest.decision_tree import DecisionTree

dummy = DecisionTree(0, 0, 0)
S = np.array([[0, 2, 5, 0], [0, 4, 5, 1], [0, -1, 5, 0]])


def test_entropy_calculation():
    assert np.round(dummy._entropy(np.array([S[0], S[2]])), 2) == 0
    assert np.round(dummy._entropy(np.array([S[1]])), 2) == 0
    assert np.round(dummy._entropy(np.array([S[0], S[1]])), 2) == 1


def test_finding_split_attr():
    (
        attr_idx,
        threshold,
        min_entropy,
        left_subset,
        right_subset,
    ) = dummy._find_split(S)
    assert attr_idx == 1
    assert threshold == 4
    assert np.round(min_entropy, 2) == 0
    assert np.array_equal(left_subset, [S[0], S[2]])
    assert np.array_equal(right_subset, [S[1]])
