#!/usr/bin/env python
"""
Benchmark evaluation suite for the BNN project.

Runs CounterDense (integer counters with margin-aware credit assignment)
against multiple datasets. With --all also the reference models: Dense (the
original rule), a Tsetlin Machine and baseline classifiers. They don't change
with CounterDense work and take nearly all the time; README.md has their numbers. Stochastic models are averaged over several seeds.
Every model run is a separate task, spread over all cores.

Usage:
    python -m benchmarks.run                  # CounterDense on every dataset
    python -m benchmarks.run parity breast    # only benchmarks whose name contains one of these
    python -m benchmarks.run --all [names]    # with the reference models
"""

import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

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

LABELS = {
    "logreg": "Logistic regr.",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "tsetlin": "Tsetlin machine",
    "bnn": "BNN (Dense)",
    "counter": "BNN (Counter)",
}


def _timed(fn, *args, **kwargs):
    t0 = time.time()
    return fn(*args, **kwargs), time.time() - t0


def submit_benchmark(pool, data, dense_kwargs, counter_kwargs, tsetlin_kwargs, models=LABELS, seeds=SEEDS):
    """Queue the runs of `models` (column keys) on one benchmark. Returns {column key: [futures]}."""
    def once(fn):
        return lambda: [pool.submit(_timed, fn, *data)]

    def per_seed(fn, **kwargs):
        return lambda: [pool.submit(_timed, fn, *data, seed=seed, **kwargs) for seed in seeds]

    runs = {
        "logreg": once(logistic_regression_accuracy),
        "decision_tree": once(decision_tree_accuracy),
        "random_forest": once(random_forest_accuracy),
        "tsetlin": per_seed(tsetlin_accuracy, **tsetlin_kwargs),
        "bnn": per_seed(train_and_evaluate, model="dense", **dense_kwargs),
        "counter": per_seed(train_and_evaluate, model="counter", **counter_kwargs),
    }
    return {key: runs[key]() for key in models}


def report_benchmark(name, data, futures):
    """Wait for one benchmark's runs and print them. Times are compute time, summed over seeds."""
    X_train, y_train, X_test, y_test = data
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(
        f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features, "
        f"{len(np.unique(y_train))} classes"
    )
    print(f"  Test:  {X_test.shape[0]} samples")

    results = {
        "name": name,
        "random": random_guess_accuracy(y_train, y_test),
        "majority": majority_class_accuracy(y_train, y_test),
    }
    print(f"  {'Random guess':<16} {results['random']:.4f}")
    print(f"  {'Majority class':<16} {results['majority']:.4f}")
    for key, label in LABELS.items():
        if key not in futures:
            continue
        scores, seconds = zip(*(f.result() for f in futures[key]))
        results[key] = np.mean(scores)
        if len(scores) == 1:
            print(f"  {label:<16} {scores[0]:.4f}  ({seconds[0]:.1f}s)", flush=True)
        else:
            print(
                f"  {label:<16} {np.mean(scores):.4f} ± {np.std(scores):.4f}"
                f"  ({sum(seconds):.1f}s, {len(scores)} seeds)",
                flush=True,
            )
    return results


def print_summary(all_results):
    """Print a formatted summary table of all benchmark results."""
    print(f"\n\n{'='*90}")
    print("  BENCHMARK RESULTS SUMMARY  (stochastic models: mean over seeds)")
    print(f"{'='*90}\n")

    if not all_results:
        return
    columns = [(key, title) for key, title in COLUMNS if key in all_results[0]]
    header = f"{'Dataset':<16}" + "".join(f" {title:>8}" for _, title in columns)
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['name']:<16}" + "".join(f" {r[key]:>8.4f}" for key, _ in columns))
    print()

    for key, title in [("bnn", "BNN (Dense)"), ("counter", "BNN (Counter)")]:
        if key not in all_results[0]:
            continue
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
            "n_epochs": 5, "hidden_sizes": (4096,), "batch_size": 32,
            "layer_kwargs": {"connect_level": 16, "counter_max": 128, "init_range": 24, "threshold_level": 1},
            # the hidden layer moves little from its random start (see README, "Wider and deeper")
            "hidden_kwargs": {"connect_level": 256, "counter_max": 2048, "init_range": 384},
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
        {"n_epochs": 5, "hidden_sizes": (), "batch_size": 16},
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
        {"n_epochs": 300, "hidden_sizes": (32,), "batch_size": 2},
        {"n_epochs": 300, "n_clauses": 32, "T": 8, "s": 20.0},
    ),
    (
        "Parity-8",
        lambda: load_parity(8),
        {"n_epochs": 50, "hidden_sizes": (16, 8), "batch_size": 32},
        {"n_epochs": 200, "hidden_sizes": (128,), "batch_size": 8},
        {"n_epochs": 100, "n_clauses": 512, "T": 128, "s": 20.0},
    ),
    (
        "Parity-16",
        lambda: load_parity(16),
        {"n_epochs": 10, "hidden_sizes": (32, 16), "batch_size": 64},
        {"n_epochs": 20, "hidden_sizes": (256,), "batch_size": 32},
        {"n_epochs": 1, "n_clauses": 100, "T": 50, "s": 20.0},
    ),
]


def main():
    np.random.seed(42)
    start_time = time.time()

    models = list(LABELS) if "--all" in sys.argv[1:] else ["counter"]
    selected = [arg.lower() for arg in sys.argv[1:] if not arg.startswith("--")]
    benchmarks = [
        b for b in BENCHMARKS
        if not selected or any(s in b[0].lower() for s in selected)
    ]

    all_results = []
    # one BLAS thread per worker, the workers already use every core
    with ProcessPoolExecutor(initializer=threadpool_limits, initargs=(1,)) as pool:
        queued = []
        for name, load_fn, *model_kwargs in benchmarks:
            print(f"Loading {name}...", flush=True)
            data = load_fn()
            queued.append((name, data, submit_benchmark(pool, data, *model_kwargs, models=models)))

        for name, data, futures in queued:
            try:
                all_results.append(report_benchmark(name, data, futures))
            except Exception:
                traceback.print_exc()

    total_time = time.time() - start_time
    print_summary(all_results)
    print(f"\nTotal time: {total_time:.1f}s")

    if total_time > 1800:
        print("WARNING: Total time exceeds 30 minute limit!")
        sys.exit(1)


if __name__ == "__main__":
    main()
