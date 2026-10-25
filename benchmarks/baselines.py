"""
Baseline models for benchmark comparison.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from bnn.tsetlin import TsetlinMachine


def random_guess_accuracy(y_train, y_test):
    """Expected accuracy of random guessing (uniform over classes)."""
    n_classes = len(np.unique(y_train))
    return 1.0 / n_classes


def majority_class_accuracy(y_train, y_test):
    """Accuracy of always predicting the most common class."""
    values, counts = np.unique(y_train, return_counts=True)
    majority_class = values[np.argmax(counts)]
    return np.mean(y_test == majority_class)


def decision_tree_accuracy(X_train, y_train, X_test, y_test):
    """Accuracy of a decision tree classifier."""
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def random_forest_accuracy(X_train, y_train, X_test, y_test):
    """Accuracy of a random forest classifier."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def logistic_regression_accuracy(X_train, y_train, X_test, y_test):
    """Accuracy of a linear model -- what a single layer could at best reach."""
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def tsetlin_accuracy(X_train, y_train, X_test, y_test, n_epochs=20, max_train=None, seed=None, **kwargs):
    """Accuracy of a Tsetlin Machine, the established fully discrete learner."""
    rng = np.random.default_rng(seed)
    if max_train is not None and len(X_train) > max_train:
        subset = rng.choice(len(X_train), max_train, replace=False)
        X_train, y_train = X_train[subset], y_train[subset]
    tm = TsetlinMachine(
        n_features=X_train.shape[1],
        n_classes=len(np.unique(y_train)),
        rng=rng,
        **kwargs,
    )
    for _ in range(n_epochs):
        tm.fit_epoch(X_train, y_train)
    return tm.score(X_test, y_test)
