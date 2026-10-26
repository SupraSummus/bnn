import numpy as np

from bnn.counter import CounterDense
from bnn.exact import move_gains, move_terms, polish_readout


def errors_and_loss(margins, labels, hinge):
    errors = np.sum(margins.argmax(axis=1) != labels)
    gaps = margins[np.arange(len(labels)), labels][:, None] - margins
    other = np.arange(margins.shape[1]) != labels[:, None]
    return errors, np.sum(np.maximum(0, hinge - gaps) * other)


def test_move_terms_match_brute_force():
    rng = np.random.default_rng(0)
    # small margins: plenty of ties and gaps at the hinge
    margins = rng.integers(-3, 4, size=(200, 4))
    labels = rng.integers(0, 4, size=200)
    errors, loss, terms = move_terms(margins, labels, 2)
    assert (errors, loss) == errors_and_loss(margins, labels, 2)
    for b in range(len(labels)):
        base = errors_and_loss(margins[b:b + 1], labels[b:b + 1], 2)
        for o in range(4):
            for row, step in [(0, 1), (1, -1)]:
                moved = margins[b:b + 1].copy()
                moved[0, o] += step
                after = errors_and_loss(moved, labels[b:b + 1], 2)
                assert terms[b, row, o] == after[0] - base[0]
                assert terms[b, row + 2, o] == after[1] - base[1]


def test_move_gains_match_brute_force():
    rng = np.random.default_rng(1)
    inputs = rng.random((300, 6)) < 0.5
    weights = rng.integers(-1, 2, size=(3, 6))
    thresholds = rng.integers(-1, 2, size=3)
    labels = rng.integers(0, 3, size=300)

    def evaluate(w, t):
        return errors_and_loss(inputs.astype(np.int64) @ w.T - t, labels, 3)

    # chunks smaller than the set: the sum over chunks must be exact too
    errors, loss, error_gains, loss_gains = move_gains(
        inputs, inputs.astype(np.int64) @ weights.T - thresholds, labels, 3, chunk=64,
    )
    assert (errors, loss) == evaluate(weights, thresholds)
    for direction, step in enumerate([1, -1]):
        for o in range(3):
            for i in range(6 + 1):  # the last one is the threshold, with the opposite sign
                w, t = weights.copy(), thresholds.copy()
                if i < 6:
                    w[o, i] += step
                else:
                    t[o] -= step
                after = evaluate(w, t)
                assert error_gains[direction, i, o] == after[0] - errors
                assert loss_gains[direction, i, o] == after[1] - loss


def test_polish_readout_never_adds_errors():
    rng = np.random.default_rng(2)
    inputs = rng.random((500, 40)) < 0.4
    labels = (inputs[:, :8].sum(axis=1) + inputs[:, 30]) % 3
    layer = CounterDense(40, 3, rng=rng)
    margins = layer.margins(inputs)
    before = errors_and_loss(margins, labels, layer.hinge)
    steps = polish_readout(layer, inputs, labels)
    after = errors_and_loss(layer.margins(inputs), labels, layer.hinge)
    assert steps > 1
    assert after[0] < before[0]
    assert after[1] < before[1]
