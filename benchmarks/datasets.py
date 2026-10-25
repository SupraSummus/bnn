"""
Data loaders for benchmark datasets.

All loaders return (X_train, y_train, X_test, y_test) where:
- X arrays are boolean numpy arrays of shape (n_samples, n_features)
- y arrays are integer numpy arrays of shape (n_samples,)

Uses OpenML for Fashion-MNIST, Mushroom, and Adult Income when network
is available. Falls back to scikit-learn built-in datasets otherwise.
"""

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_classification,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def binarize_continuous(X, threshold=None):
    """Binarize continuous features by thresholding at the median."""
    if threshold is None:
        threshold = np.median(X, axis=0)
    return (X > threshold).astype(np.bool_)


def thermometer(X, n_thresholds=3):
    """
    Binarize each continuous column into `n_thresholds` bits: x > q for
    evenly spaced quantiles q. Keeps order information that a single
    median split throws away.
    """
    quantiles = np.linspace(0, 1, n_thresholds + 2)[1:-1]
    thresholds = np.quantile(X, quantiles, axis=0)
    return (X[:, np.newaxis, :] > thresholds[np.newaxis]).reshape(len(X), -1)


def binarize_openml(data):
    """
    Binarize an OpenML bunch fetched with as_frame=False: one-hot for
    categorical columns (missing values get no bit), thermometer for numeric.
    """
    categories = data.categories or {}
    columns = []
    for j, name in enumerate(data.feature_names):
        column = data.data[:, j]
        if name in categories:
            values = np.array(categories[name], dtype=object)
            columns.append(column[:, np.newaxis] == values[np.newaxis, :])
        else:
            columns.append(thermometer(column.astype(np.float32)[:, np.newaxis]))
    return np.concatenate(columns, axis=1).astype(np.bool_)


def _try_fetch_openml(name, version=1):
    """Try to fetch from OpenML, return None on network failure."""
    try:
        from sklearn.datasets import fetch_openml

        return fetch_openml(
            name, version=version, as_frame=False, parser="auto"
        )
    except OSError:
        return None


def load_fashion_mnist():
    """
    Load Fashion-MNIST (28x28 images, 10 classes).

    Falls back to sklearn digits (8x8 images, 10 classes) if
    OpenML is unavailable.
    """
    data = _try_fetch_openml("Fashion-MNIST", version=1)
    if data is not None:
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int32)
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        X_train = (X_train > 128).astype(np.bool_)
        X_test = (X_test > 128).astype(np.bool_)
        return X_train, y_train, X_test, y_test

    # Fallback: sklearn digits (8x8 images, 10 classes)
    print("    (using sklearn digits as Fashion-MNIST fallback)")
    data = load_digits()
    X = binarize_continuous(data.data.astype(np.float32))
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, y_train, X_test, y_test


def load_mushroom():
    """
    Load Mushroom dataset (binary classification).

    Falls back to a synthetic binary classification dataset if
    OpenML is unavailable.
    """
    data = _try_fetch_openml("mushroom", version=1)
    if data is not None:
        X = binarize_openml(data)
        y = LabelEncoder().fit_transform(data.target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, y_train, X_test, y_test

    # Fallback: synthetic binary classification
    print("    (using synthetic data as Mushroom fallback)")
    X, y = make_classification(
        n_samples=2000,
        n_features=22,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    X = binarize_continuous(X.astype(np.float32))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, y_train, X_test, y_test


def load_adult():
    """
    Load Adult Income dataset (binary classification).

    Falls back to sklearn wine dataset (3 classes) if
    OpenML is unavailable.
    """
    data = _try_fetch_openml("adult", version=2)
    if data is not None:
        X = binarize_openml(data)
        y = LabelEncoder().fit_transform(data.target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, y_train, X_test, y_test

    # Fallback: sklearn wine dataset
    print("    (using sklearn wine as Adult Income fallback)")
    data = load_wine()
    X = binarize_continuous(data.data.astype(np.float32))
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, y_train, X_test, y_test


def load_breast_cancer_dataset():
    """Load Breast Cancer Wisconsin dataset (binary classification)."""
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target

    X = binarize_continuous(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, y_train, X_test, y_test


def load_parity(n_bits):
    """
    Generate synthetic parity problem (XOR of all bits).

    Returns the full truth table as both train and test set. Parity can't
    be generalized from a partial table (any held-out pattern differs from
    a seen one in a single bit, which flips the label), so what we measure
    is whether the model can represent and find the function at all.
    """
    n_samples = 2**n_bits
    X = np.array(
        [
            [(i >> bit) & 1 for bit in range(n_bits)]
            for i in range(n_samples)
        ],
        dtype=np.bool_,
    )
    y = np.sum(X, axis=1) % 2  # parity = XOR of all bits
    return X, y, X, y
