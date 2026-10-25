#!/usr/bin/env python
"""
Benchmark evaluation suite for the BNN project.

Runs both BNN variants (Dense -- the original rule, CounterDense -- integer
counters with margin-aware credit assignment) and a Tsetlin Machine against
multiple datasets and compares them with baseline classifiers. Stochastic
models are averaged over several seeds.

Usage:
    python -m benchmarks.run              # everything
    python -m benchmarks.run parity breast  # only benchmarks whose name contains one of these
"""

import sys
import time

import numpy as np

from benchmarks.baselines import (
    decision_tree_accuracy,
    logistic_regression_accuracy,
    majority_class_accuracy,
    random_forest_accuracy,
    random_guess_accuracy,
    tsetlin_accuracy,
)
from benchmarks.bnn_model import train_and_evaluate
from benchmarks.datasets import (
    load_adult,
    load_breast_cancer_dataset,
    load_fashion_mnist,
    load_mushroom,
    load_parity,
)

SEEDS = (0, 1, 2)

# Narrow hysteresis, dense initial wiring: hidden layers learn faster, but noisy on wide inputs.
SMALL_COUNTERS = {"connect_level": 2, "counter_max": 16, "init_range": 2}

COLUMNS = [
    ("random", "Random"),
    ("majority", "Majority"),
    ("logreg", "LogReg"),
    ("decision_tree", "DTree"),
    ("random_forest", "RForest"),
    ("tsetlin", "Tsetlin"),
    ("bnn", "BNN"),
    ("counter", "Counter"),
]


def _over_seeds(label, results, key, fn, seeds):
    t0 = time.time()
    scores = [fn(seed) for seed in seeds]
    results[key] = np.mean(scores)
    print(
        f"  {label:<16} {np.mean(scores):.4f} ± {np.std(scores):.4f}"
        f"  ({time.time() - t0:.1f}s, {len(seeds)} seeds)",
        flush=True,
    )


def run_benchmark(name, load_fn, dense_kwargs, counter_kwargs, tsetlin_kwargs, seeds=SEEDS):
    """Run a single benchmark: load data, train models, compute baselines."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    print("  Loading data...", flush=True)
    X_train, y_train, X_test, y_test = load_fn()
    n_classes = len(np.unique(y_train))
    print(
        f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features, "
        f"{n_classes} classes"
    )
    print(f"  Test:  {X_test.shape[0]} samples")

    results = {"name": name}
    data = (X_train, y_train, X_test, y_test)

    results["random"] = random_guess_accuracy(y_train, y_test)
    results["majority"] = majority_class_accuracy(y_train, y_test)
    print(f"  {'Random guess':<16} {results['random']:.4f}")
    print(f"  {'Majority class':<16} {results['majority']:.4f}")
    for key, label, fn in [
        ("logreg", "Logistic regr.", logistic_regression_accuracy),
        ("decision_tree", "Decision tree", decision_tree_accuracy),
        ("random_forest", "Random forest", random_forest_accuracy),
    ]:
        t0 = time.time()
        results[key] = fn(*data)
        print(f"  {label:<16} {results[key]:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    _over_seeds(
        "Tsetlin machine", results, "tsetlin",
        lambda seed: tsetlin_accuracy(*data, seed=seed, **tsetlin_kwargs),
        seeds,
    )
    _over_seeds(
        "BNN (Dense)", results, "bnn",
        lambda seed: train_and_evaluate(*data, model="dense", seed=seed, **dense_kwargs),
        seeds,
    )
    _over_seeds(
        "BNN (Counter)", results, "counter",
        lambda seed: train_and_evaluate(*data, model="counter", seed=seed, **counter_kwargs),
        seeds,
    )
    return results


def print_summary(all_results):
    """Print a formatted summary table of all benchmark results."""
    print(f"\n\n{'='*90}")
    print("  BENCHMARK RESULTS SUMMARY  (stochastic models: mean over seeds)")
    print(f"{'='*90}\n")

    header = f"{'Dataset':<16}" + "".join(f" {title:>8}" for _, title in COLUMNS)
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['name']:<16}" + "".join(f" {r[key]:>8.4f}" for key, _ in COLUMNS))
    print()

    for key, title in [("bnn", "BNN (Dense)"), ("counter", "BNN (Counter)")]:
        beats_random_all = all(r[key] > r["random"] for r in all_results)
        fashion = [r[key] > r["majority"] for r in all_results if r["name"] == "Fashion-MNIST"]
        tabular = [
            r["name"] for r in all_results
            if r["name"] != "Fashion-MNIST" and "parity" not in r["name"].lower()
            and r[key] > r["majority"]
        ]
        print(f"Acceptance criteria, {title}:")
        print(f"  > random on all benchmarks:   {'PASS' if beats_random_all else 'FAIL'}")
        if fashion:
            print(f"  > majority on Fashion-MNIST:  {'PASS' if fashion[0] else 'FAIL'}")
        print(
            f"  > majority on >= 1 tabular:   {'PASS' if tabular else 'FAIL'}"
            + (f" ({', '.join(tabular)})" if tabular else "")
        )


BENCHMARKS = [
    (
        "Fashion-MNIST",
        load_fashion_mnist,
        {"n_epochs": 5, "hidden_sizes": (128, 64), "batch_size": 128},
        {
            "n_epochs": 5, "hidden_sizes": (), "batch_size": 32,
            "layer_kwargs": {"connect_level": 8, "counter_max": 64, "init_range": 12},
        },
        {"n_epochs": 3, "max_train": 5000, "n_clauses": 100, "T": 20, "s": 10.0},
    ),
    (
        "Mushroom",
        load_mushroom,
        {"n_epochs": 20, "hidden_sizes": (32, 16), "batch_size": 64},
        {"n_epochs": 10, "hidden_sizes": (32,), "batch_size": 16},
        {"n_epochs": 5, "n_clauses": 100, "T": 15, "s": 3.9},
    ),
    (
        "Adult Income",
        load_adult,
        {"n_epochs": 15, "hidden_sizes": (32, 16), "batch_size": 64},
        {
            "n_epochs": 5, "hidden_sizes": (), "batch_size": 16,
            "layer_kwargs": {"connect_level": 8, "counter_max": 64, "init_range": 12},
        },
        {"n_epochs": 3, "n_clauses": 100, "T": 20, "s": 5.0},
    ),
    (
        "Breast Cancer",
        load_breast_cancer_dataset,
        {"n_epochs": 30, "hidden_sizes": (16, 8), "batch_size": 32},
        {"n_epochs": 30, "hidden_sizes": (), "batch_size": 32},
        {"n_epochs": 30, "n_clauses": 100, "T": 15, "s": 3.9},
    ),
    (
        "Parity-4",
        lambda: load_parity(4),
        {"n_epochs": 50, "hidden_sizes": (8,), "batch_size": 8},
        {"n_epochs": 300, "hidden_sizes": (32,), "batch_size": 2, "layer_kwargs": SMALL_COUNTERS},
        {"n_epochs": 300, "n_clauses": 32, "T": 8, "s": 20.0},
    ),
    (
        "Parity-8",
        lambda: load_parity(8),
        {"n_epochs": 50, "hidden_sizes": (16, 8), "batch_size": 32},
        {"n_epochs": 200, "hidden_sizes": (128,), "batch_size": 8, "layer_kwargs": SMALL_COUNTERS},
        {"n_epochs": 100, "n_clauses": 512, "T": 128, "s": 20.0},
    ),
    (
        "Parity-16",
        lambda: load_parity(16),
        {"n_epochs": 10, "hidden_sizes": (32, 16), "batch_size": 64},
        {"n_epochs": 20, "hidden_sizes": (256,), "batch_size": 32, "layer_kwargs": SMALL_COUNTERS},
        {"n_epochs": 1, "n_clauses": 100, "T": 50, "s": 20.0},
    ),
]


def main():
    np.random.seed(42)
    start_time = time.time()

    selected = [arg.lower() for arg in sys.argv[1:]]
    benchmarks = [
        b for b in BENCHMARKS
        if not selected or any(s in b[0].lower() for s in selected)
    ]

    all_results = []
    for name, load_fn, dense_kwargs, counter_kwargs, tsetlin_kwargs in benchmarks:
        try:
            result = run_benchmark(name, load_fn, dense_kwargs, counter_kwargs, tsetlin_kwargs)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    total_time = time.time() - start_time
    print_summary(all_results)
    print(f"\nTotal time: {total_time:.1f}s")

    if total_time > 1800:
        print("WARNING: Total time exceeds 30 minute limit!")
        sys.exit(1)


if __name__ == "__main__":
    main()
