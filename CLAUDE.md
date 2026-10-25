# Notes for agents

* Tests: `pytest` (files `test_*.py` and `*_tests.py`, next to the code in `bnn/`).
* Benchmarks: `python -m benchmarks.run [name-substring ...]`. Full run ~7 min, one process per core;
  Fashion-MNIST, Mushroom and Adult are fetched from OpenML (cached in `~/scikit_learn_data`, needs pandas<3).
  Without network they fall back to small sklearn stand-ins with a one-line note: check the "Train:" sizes.
* Results and design notes live in README.md; open problems in TODO.md -- update both when numbers change.
* Report accuracy over several seeds; single runs of these learners vary by ±0.1.
* `flake8` reports isort errors that predate this code; don't mass-fix them in unrelated changes.
