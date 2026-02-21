#!/usr/bin/env python
"""
Metaparameter analysis for the Breast Cancer benchmark.

Varies metaparameters (epochs, hidden layer size, batch size, max synapses)
one at a time and plots classification accuracy for each setting.
Multiple random seeds are used per configuration to show mean ± std.

Usage:
    python -m benchmarks.plot_metaparams
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.bnn_model import evaluate, train_epoch
from benchmarks.datasets import load_breast_cancer_dataset
from bnn.abstract import Sequence
from bnn.dense import Dense

matplotlib.use("Agg")

# ---------- helpers --------------------------------------------------


def build_network(input_features, n_classes, hidden_sizes, max_synapses):
    """Build a BNN with explicit max_synapses control."""
    layers = []
    prev_size = input_features
    for size in hidden_sizes:
        layers.append(
            Dense(
                input_features=prev_size,
                output_features=size,
                max_synapses=max_synapses,
                treshold_change_chance=0.1,
                synapse_change_chance=0.1,
            )
        )
        prev_size = size
    layers.append(
        Dense(
            input_features=prev_size,
            output_features=n_classes,
            max_synapses=max_synapses,
            treshold_change_chance=0.1,
            synapse_change_chance=0.1,
        )
    )
    return Sequence(layers)


def train_and_eval(
    X_train,
    y_train,
    X_test,
    y_test,
    n_epochs,
    hidden_sizes,
    batch_size,
    max_synapses,
):
    """Single train-and-evaluate run."""
    n_classes = len(np.unique(y_train))
    network = build_network(
        input_features=X_train.shape[1],
        n_classes=n_classes,
        hidden_sizes=hidden_sizes,
        max_synapses=max_synapses,
    )
    for _ in range(n_epochs):
        train_epoch(network, X_train, y_train, n_classes, batch_size=batch_size)
    return evaluate(network, X_test, y_test, n_classes)


def sweep(param_name, param_values, fixed, data, n_seeds=5):
    """Sweep one parameter, return (values, means, stds)."""
    X_train, y_train, X_test, y_test = data
    means, stds = [], []
    for val in param_values:
        kwargs = dict(fixed)
        kwargs[param_name] = val
        accs = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            acc = train_and_eval(X_train, y_train, X_test, y_test, **kwargs)
            accs.append(acc)
        means.append(np.mean(accs))
        stds.append(np.std(accs))
        print(f"  {param_name}={val}  acc={means[-1]:.4f} ± {stds[-1]:.4f}")
    return np.array(means), np.array(stds)


# ---------- main -----------------------------------------------------


def main():
    print("Loading Breast Cancer dataset …")
    data = load_breast_cancer_dataset()
    X_train, y_train, X_test, y_test = data
    print(
        f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features, "
        f"{len(np.unique(y_train))} classes"
    )

    # Default (baseline) values for each metaparameter
    defaults = dict(
        n_epochs=30,
        hidden_sizes=(16,),
        batch_size=32,
        max_synapses=8,
    )

    sweeps = [
        ("n_epochs", [5, 10, 20, 30, 50, 80], "Number of epochs"),
        (
            "hidden_sizes",
            [(4,), (8,), (16,), (32,), (64,), (128,)],
            "Hidden layer size",
        ),
        ("batch_size", [8, 16, 32, 64, 128], "Batch size"),
        ("max_synapses", [2, 4, 8, 16, 30], "Max synapses per neuron"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "BNN accuracy vs metaparameters (Breast Cancer dataset)",
        fontsize=14,
    )

    for ax, (param_name, param_values, label) in zip(axes.flat, sweeps):
        print(f"\nSweeping {param_name} …")
        fixed = {k: v for k, v in defaults.items() if k != param_name}

        means, stds = sweep(param_name, param_values, fixed, data, n_seeds=5)

        # For hidden_sizes we plot the single-element tuple value
        if param_name == "hidden_sizes":
            x_vals = [s[0] for s in param_values]
        else:
            x_vals = param_values

        ax.errorbar(x_vals, means, yerr=stds, marker="o", capsize=4)
        ax.set_xlabel(label)
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "benchmarks/metaparams.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
