# Notes for agents

* Tests: `pytest` (files `test_*.py` and `*_tests.py`, next to the code in `bnn/`).
* Benchmarks: `python -m benchmarks.run [name-substring ...]`, CounterDense only, ~2 min (Fashion-MNIST, 4096 hidden, is most of it).
  `--all` adds the reference models (baselines, Tsetlin, Dense; numbers in README), takes minutes; one process per core;
  Fashion-MNIST, Mushroom and Adult are fetched from OpenML (cached in `~/scikit_learn_data`, needs pandas<3).
  Without network they fall back to small sklearn stand-ins with a one-line note: check the "Train:" sizes.
* Results and design notes live in README.md; open problems in TODO.md -- update both when numbers change.
* Report accuracy over several seeds; single runs of these learners vary by ±0.1.
* `CounterDense` runs as numba kernels (`bnn/counter_kernels.py`, compiled on first use, cached in `__pycache__`).
  `bnn/counter_reference.py` is the same rule in numpy; a rule change goes into both,
  `bnn/counter_tests.py` checks they agree bit for bit.
* `python -m benchmarks.ladder` swaps the rule's parts for their gradient versions one by one (README, "The ladder").
* `flake8` reports isort errors that predate this code; don't mass-fix them in unrelated changes.
