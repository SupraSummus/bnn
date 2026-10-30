"""
BNN model wrapper for benchmarks.

Wraps the Dense/CounterDense/Sequence layers into a train/evaluate interface
suitable for the benchmark suite.
"""

import numpy as np

from bnn.abstract import Sequence
from bnn.counter import CounterDense, Ratchet, readout_inputs, refit_readout
from bnn.counter import train_epoch as train_counter_epoch
from bnn.dense import Dense
from bnn.exact import polish_readout


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


def build_counter_network(
    input_features, n_classes, hidden_sizes=(64, 32), rng=None, hidden_kwargs=None, output_kwargs=None,
    **layer_kwargs,
):
    """
    Build a CounterDense network for classification.

    Hidden layers only accept flips they can make cheaply (minimal disturbance)
    and revive units that stopped varying (homeostasis); the output layer, whose
    errors are exact, takes them all and also keeps a safety margin on correct
    answers. Its margins are the loss's input, not a step to differentiate, so it passes
    its pressure down whole, summed, without dividing by sqrt(reach) (README, "The ladder").
    `layer_kwargs` go to every layer, `hidden_kwargs` and `output_kwargs` (over them)
    to the hidden ones and the output layer.
    """
    sizes = [input_features, *hidden_sizes]
    hidden_kwargs = {"accept_distance": 0, "homeostasis": 256, **layer_kwargs, **(hidden_kwargs or {})}
    output_kwargs = {"stabilize_margin": 2, "surrogate": "none", "normalize_reach": False,
                     **layer_kwargs, **(output_kwargs or {})}
    layers = [CounterDense(n_in, n_out, rng=rng, **hidden_kwargs) for n_in, n_out in zip(sizes, sizes[1:])]
    layers.append(CounterDense(sizes[-1], n_classes, rng=rng, **output_kwargs))
    return Sequence(layers)


def train_epoch(network, X_train, y_train, n_classes, batch_size=64, rng=None, ratchet=None):
    """Train the network for one epoch; a CounterDense net with a `ratchet` is checked as often as it asks."""
    if rng is None:
        rng = np.random.default_rng()
    n_samples = X_train.shape[0]
    indices = rng.permutation(n_samples)

    if all(type(layer) is CounterDense for layer in network.layers):
        targets = np.arange(n_classes) == y_train[:, np.newaxis]
        if ratchet is None:
            train_counter_epoch(network.layers, X_train, targets, indices, batch_size)
            return
        chunk = batch_size * ratchet.check_every
        for start in range(0, n_samples, chunk):
            train_counter_epoch(network.layers, X_train, targets, indices[start:start + chunk], batch_size)
            ratchet.check(network.layers)
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
    hidden_kwargs=None,
    output_kwargs=None,
    consolidate=True,
    polish_steps=32,
    ratchet_tolerance=0.25,
):
    """
    Build, train, and evaluate a BNN on the given dataset.
    CounterDense nets train under a `bnn.counter.Ratchet` (`ratchet_tolerance` None: without),
    are consolidated over their last epoch (`consolidate`), replaced by the best state the ratchet saw
    if that one is more accurate by more than the tolerance, and their output layer is then polished
    by up to `polish_steps` steps of exact coordinate descent (`bnn.exact.polish_readout`).
    """
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
            hidden_kwargs=hidden_kwargs,
            output_kwargs=output_kwargs,
            **(layer_kwargs or {}),
        )
    else:
        raise ValueError(f"Unknown model {model!r}")

    counter = model == "counter"
    tallied = network.layers if counter and consolidate else []
    ratchet = Ratchet(X_train, y_train, ratchet_tolerance) if counter and ratchet_tolerance is not None else None
    for epoch in range(n_epochs):
        if epoch == n_epochs - 1:
            for layer in tallied:
                layer.start_tally()
            if tallied and ratchet:
                ratchet.rebase(network.layers)
        train_epoch(network, X_train, y_train, n_classes, batch_size=batch_size, rng=rng, ratchet=ratchet)
    for layer in tallied:
        layer.consolidate()
    if tallied and hidden_sizes:
        targets = np.arange(n_classes) == y_train[:, np.newaxis]
        refit_readout(network.layers, X_train, targets, rng.permutation(len(X_train)), batch_size)
    if ratchet:
        ratchet.keep_best(network.layers)
    if counter and polish_steps:
        polish_readout(network.layers[-1], readout_inputs(network.layers, X_train), y_train, max_steps=polish_steps)

    accuracy = evaluate(network, X_test, y_test, n_classes)
    return accuracy
