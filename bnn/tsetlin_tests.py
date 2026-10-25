import numpy as np

from bnn.tsetlin import TsetlinMachine


def test_clause_outputs():
    tm = TsetlinMachine(n_features=2, n_classes=2, n_clauses=2, n_states=10)
    tm.states[:] = 1  # exclude everything
    tm.states[0, 0, 0] = 20  # class 0, clause 0: x0
    tm.states[0, 1, 3] = 20  # class 0, clause 1: not x1
    X = np.array([[True, True], [True, False], [False, False]])
    out = tm.clause_outputs(X)
    np.testing.assert_array_equal(out[:, 0], [[True, False], [True, True], [False, True]])
    # class 1 has only empty clauses: silent at inference, firing while training
    assert not out[:, 1].any()
    assert tm.clause_outputs(X, training=True)[:, 1].all()


def test_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=bool)
    y = np.array([0, 1, 1, 0])
    tm = TsetlinMachine(n_features=2, n_classes=2, n_clauses=10, T=10, s=3.0, rng=np.random.default_rng(0))
    for _ in range(50):
        tm.fit_epoch(X, y)
    assert tm.score(X, y) == 1.0
