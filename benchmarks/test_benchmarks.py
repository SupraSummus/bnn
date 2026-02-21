"""Tests for the benchmark infrastructure."""

import numpy as np

from benchmarks.baselines import (
    decision_tree_accuracy,
    majority_class_accuracy,
    random_forest_accuracy,
    random_guess_accuracy,
)
from benchmarks.bnn_model import build_network, evaluate, train_epoch
from benchmarks.datasets import (
    binarize_continuous,
    load_breast_cancer_dataset,
    load_parity,
)


def test_binarize_continuous():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = binarize_continuous(X)
    expected = np.array([[False, False], [False, False], [True, True]])
    np.testing.assert_array_equal(result, expected)


def test_load_parity_4():
    X_train, y_train, X_test, y_test = load_parity(4)
    assert X_train.shape == (8, 4)
    assert X_test.shape == (8, 4)
    assert X_train.dtype == np.bool_
    assert set(y_train).issubset({0, 1})


def test_load_breast_cancer():
    X_train, y_train, X_test, y_test = load_breast_cancer_dataset()
    assert X_train.ndim == 2
    assert X_train.dtype == np.bool_
    assert X_train.shape[0] > X_test.shape[0]


def test_random_guess_accuracy():
    y_train = np.array([0, 0, 1, 1, 2])
    y_test = np.array([0, 1, 2])
    assert random_guess_accuracy(y_train, y_test) == 1.0 / 3


def test_majority_class_accuracy():
    y_train = np.array([0, 0, 0, 1])
    y_test = np.array([0, 0, 1, 1])
    assert majority_class_accuracy(y_train, y_test) == 0.5


def test_decision_tree_accuracy():
    X_train, y_train, X_test, y_test = load_breast_cancer_dataset()
    acc = decision_tree_accuracy(X_train, y_train, X_test, y_test)
    assert 0.0 <= acc <= 1.0


def test_random_forest_accuracy():
    X_train, y_train, X_test, y_test = load_breast_cancer_dataset()
    acc = random_forest_accuracy(X_train, y_train, X_test, y_test)
    assert 0.0 <= acc <= 1.0


def test_build_network():
    net = build_network(input_features=10, n_classes=3, hidden_sizes=(8, 4))
    assert len(net.layers) == 3
    assert net.layers[0].input_features == 10
    assert net.layers[0].output_features == 8
    assert net.layers[1].input_features == 8
    assert net.layers[1].output_features == 4
    assert net.layers[2].input_features == 4
    assert net.layers[2].output_features == 3


def test_train_and_evaluate():
    X_train, y_train, X_test, y_test = load_parity(4)
    n_classes = 2
    net = build_network(
        input_features=4, n_classes=n_classes, hidden_sizes=(8,)
    )
    train_epoch(net, X_train, y_train, n_classes, batch_size=8)
    acc = evaluate(net, X_test, y_test, n_classes)
    assert 0.0 <= acc <= 1.0
