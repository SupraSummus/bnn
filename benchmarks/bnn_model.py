"""
BNN model wrapper for benchmarks.

Wraps the Dense/CounterDense/Sequence layers into a train/evaluate interface
suitable for the benchmark suite.
"""

import numpy as np

from bnn.abstract import Sequence
from bnn.counter import CounterDense, refit_readout
from bnn.counter import train_epoch as train_counter_epoch
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


def build_counter_network(input_features, n_classes, hidden_sizes=(64, 32), rng=None, **layer_kwargs):
    """
    Build a CounterDense network for classification.

    Hidden layers only accept flips they can make cheaply (minimal disturbance)
    and revive units that stopped varying (homeostasis); the output layer, whose
    errors are exact, takes them all and also keeps a safety margin on correct
    answers. `layer_kwargs` go to every layer.
    """
    sizes = [input_features, *hidden_sizes]
    layers = [
        CounterDense(n_in, n_out, accept_distance=0, homeostasis=256, rng=rng, **layer_kwargs)
        for n_in, n_out in zip(sizes, sizes[1:])
    ]
    layers.append(CounterDense(sizes[-1], n_classes, stabilize_margin=2, rng=rng, **layer_kwargs))
    return Sequence(layers)


def train_epoch(network, X_train, y_train, n_classes, batch_size=64, rng=None):
    """Train the network for one epoch."""
    if rng is None:
        rng = np.random.default_rng()
    n_samples = X_train.shape[0]
    indices = rng.permutation(n_samples)

    if all(type(layer) is CounterDense for layer in network.layers):
        targets = np.arange(n_classes) == y_train[:, np.newaxis]
        train_counter_epoch(network.layers, X_train, targets, indices, batch_size)
        return

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


def predict(network, X):
    """
    Pick the class whose output neuron is furthest above (or least below) its
    threshold. Plain argmax over binary outputs would break ties towards class 0.
    """
    return np.argmax(network.margins(X), axis=1)


def evaluate(network, X, y, n_classes):
    """Evaluate classification accuracy."""
    return np.mean(predict(network, X) == y)


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    n_epochs=20,
    hidden_sizes=(64, 32),
    batch_size=64,
    model="dense",
    seed=None,
    layer_kwargs=None,
    consolidate=True,
):
    """Build, train, and evaluate a BNN on the given dataset."""
    n_classes = len(np.unique(y_train))
    input_features = X_train.shape[1]
    rng = np.random.default_rng(seed)

    if model == "dense":
        if seed is not None:
            np.random.seed(seed)  # Dense draws from the global generator
        network = build_network(
            input_features=input_features,
            n_classes=n_classes,
            hidden_sizes=hidden_sizes,
        )
    elif model == "counter":
        network = build_counter_network(
            input_features=input_features,
            n_classes=n_classes,
            hidden_sizes=hidden_sizes,
            rng=rng,
            **(layer_kwargs or {}),
        )
    else:
        raise ValueError(f"Unknown model {model!r}")

    tallied = network.layers if model == "counter" and consolidate else []
    for epoch in range(n_epochs):
        if epoch == n_epochs - 1:
            for layer in tallied:
                layer.start_tally()
        train_epoch(network, X_train, y_train, n_classes, batch_size=batch_size, rng=rng)
    for layer in tallied:
        layer.consolidate()
    if tallied and hidden_sizes:
        targets = np.arange(n_classes) == y_train[:, np.newaxis]
        refit_readout(network.layers, X_train, targets, rng.permutation(len(X_train)), batch_size)

    accuracy = evaluate(network, X_test, y_test, n_classes)
    return accuracy
