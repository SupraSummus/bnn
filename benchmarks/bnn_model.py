"""
BNN model wrapper for benchmarks.

Wraps the Dense/Sequence layers into a train/evaluate interface
suitable for the benchmark suite.
"""

import numpy as np

from bnn.abstract import Sequence
from bnn.dense import Dense


def build_network(input_features, n_classes, hidden_sizes=(64, 32)):
    """Build a BNN network for classification."""
    layers = []
    prev_size = input_features
    for size in hidden_sizes:
        layers.append(
            Dense(
                input_features=prev_size,
                output_features=size,
                max_synapses=max(prev_size // 5, 4),
                threshold_change_chance=0.1,
                synapse_change_chance=0.1,
            )
        )
        prev_size = size
    layers.append(
        Dense(
            input_features=prev_size,
            output_features=n_classes,
            max_synapses=max(prev_size // 5, 4),
            threshold_change_chance=0.1,
            synapse_change_chance=0.1,
        )
    )
    return Sequence(layers)


def train_epoch(network, X_train, y_train, n_classes, batch_size=64):
    """Train the network for one epoch."""
    n_samples = X_train.shape[0]
    indices = np.random.permutation(n_samples)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        # One-hot encode targets
        targets = np.zeros((len(y_batch), n_classes), dtype=np.bool_)
        targets[np.arange(len(y_batch)), y_batch] = True

        # Forward pass
        output, state = network.forward(X_batch)

        # Compute error signals
        too_much = (output & ~targets).astype(np.float32)
        not_enough = (~output & targets).astype(np.float32)

        # Backward pass
        network.backward(state, too_much, not_enough)

        # Update
        network.update()


def evaluate(network, X, y, n_classes):
    """Evaluate classification accuracy."""
    output, _ = network.forward(X)
    predictions = np.argmax(output.astype(np.float32), axis=1)
    return np.mean(predictions == y)


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    n_epochs=20,
    hidden_sizes=(64, 32),
    batch_size=64,
):
    """Build, train, and evaluate a BNN on the given dataset."""
    n_classes = len(np.unique(y_train))
    input_features = X_train.shape[1]

    network = build_network(
        input_features=input_features,
        n_classes=n_classes,
        hidden_sizes=hidden_sizes,
    )

    for epoch in range(n_epochs):
        train_epoch(network, X_train, y_train, n_classes, batch_size=batch_size)

    accuracy = evaluate(network, X_test, y_test, n_classes)
    return accuracy
