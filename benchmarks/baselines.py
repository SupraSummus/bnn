"""
Baseline models for benchmark comparison.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


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
