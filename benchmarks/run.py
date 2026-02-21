#!/usr/bin/env python
"""
Benchmark evaluation suite for the BNN project.

Runs the BNN model against multiple datasets and compares
with baseline classifiers. Produces a summary results table.

Usage:
    python -m benchmarks.run
"""

import sys
import time

import numpy as np

from benchmarks.baselines import (
    decision_tree_accuracy,
    majority_class_accuracy,
    random_forest_accuracy,
    random_guess_accuracy,
)
from benchmarks.bnn_model import train_and_evaluate
from benchmarks.datasets import (
    load_adult,
    load_breast_cancer_dataset,
    load_fashion_mnist,
    load_mushroom,
    load_parity,
)


def run_benchmark(name, load_fn, bnn_kwargs):
    """Run a single benchmark: load data, train BNN, compute baselines."""
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

    # Random guess
    results["random"] = random_guess_accuracy(y_train, y_test)
    print(f"  Random guess:    {results['random']:.4f}")

    # Majority class
    results["majority"] = majority_class_accuracy(y_train, y_test)
    print(f"  Majority class:  {results['majority']:.4f}")

    # Decision tree
    print("  Training decision tree...", flush=True)
    results["decision_tree"] = decision_tree_accuracy(
        X_train, y_train, X_test, y_test
    )
    print(f"  Decision tree:   {results['decision_tree']:.4f}")

    # Random forest
    print("  Training random forest...", flush=True)
    results["random_forest"] = random_forest_accuracy(
        X_train, y_train, X_test, y_test
    )
    print(f"  Random forest:   {results['random_forest']:.4f}")

    # BNN
    print("  Training BNN...", flush=True)
    t0 = time.time()
    results["bnn"] = train_and_evaluate(
        X_train, y_train, X_test, y_test, **bnn_kwargs
    )
    elapsed = time.time() - t0
    print(f"  BNN:             {results['bnn']:.4f}  ({elapsed:.1f}s)")

    return results


def print_summary(all_results):
    """Print a formatted summary table of all benchmark results."""
    print(f"\n\n{'='*80}")
    print("  BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}\n")

    header = (
        f"{'Dataset':<25} {'Random':>8} {'Majority':>8} "
        f"{'DTree':>8} {'RForest':>8} {'BNN':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        print(
            f"{r['name']:<25} {r['random']:>8.4f} {r['majority']:>8.4f} "
            f"{r['decision_tree']:>8.4f} {r['random_forest']:>8.4f} "
            f"{r['bnn']:>8.4f}"
        )

    print()

    # Check acceptance criteria
    beats_random_all = all(r["bnn"] > r["random"] for r in all_results)
    beats_majority_fashion = None
    beats_majority_tabular = 0
    tabular_names = []

    for r in all_results:
        if r["name"] == "Fashion-MNIST":
            beats_majority_fashion = r["bnn"] > r["majority"]
        elif "parity" not in r["name"].lower():
            if r["bnn"] > r["majority"]:
                beats_majority_tabular += 1
                tabular_names.append(r["name"])

    print("Acceptance criteria:")
    print(
        f"  BNN > random on all benchmarks: "
        f"{'PASS' if beats_random_all else 'FAIL'}"
    )
    if beats_majority_fashion is not None:
        print(
            f"  BNN > majority on Fashion-MNIST: "
            f"{'PASS' if beats_majority_fashion else 'FAIL'}"
        )
    print(
        f"  BNN > majority on >= 1 tabular dataset: "
        f"{'PASS' if beats_majority_tabular >= 1 else 'FAIL'}"
        + (f" ({', '.join(tabular_names)})" if tabular_names else "")
    )


def main():
    np.random.seed(42)
    start_time = time.time()

    benchmarks = [
        (
            "Fashion-MNIST",
            load_fashion_mnist,
            {
                "n_epochs": 5,
                "hidden_sizes": (128, 64),
                "batch_size": 128,
            },
        ),
        (
            "Mushroom",
            load_mushroom,
            {
                "n_epochs": 20,
                "hidden_sizes": (32, 16),
                "batch_size": 64,
            },
        ),
        (
            "Adult Income",
            load_adult,
            {
                "n_epochs": 15,
                "hidden_sizes": (32, 16),
                "batch_size": 64,
            },
        ),
        (
            "Breast Cancer",
            load_breast_cancer_dataset,
            {
                "n_epochs": 30,
                "hidden_sizes": (16, 8),
                "batch_size": 32,
            },
        ),
        (
            "Parity-4",
            lambda: load_parity(4),
            {
                "n_epochs": 50,
                "hidden_sizes": (8,),
                "batch_size": 8,
            },
        ),
        (
            "Parity-8",
            lambda: load_parity(8),
            {
                "n_epochs": 50,
                "hidden_sizes": (16, 8),
                "batch_size": 32,
            },
        ),
        (
            "Parity-16",
            lambda: load_parity(16),
            {
                "n_epochs": 10,
                "hidden_sizes": (32, 16),
                "batch_size": 64,
            },
        ),
    ]

    all_results = []
    for name, load_fn, bnn_kwargs in benchmarks:
        try:
            result = run_benchmark(name, load_fn, bnn_kwargs)
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
