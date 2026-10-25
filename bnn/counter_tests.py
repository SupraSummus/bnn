import numpy as np

from bnn.abstract import Sequence
from bnn.counter import CounterDense


def make_layer(weights, thresholds, **kwargs):
    weights = np.array(weights, dtype=np.int8)
    layer = CounterDense(
        weights.shape[1], weights.shape[0],
        rng=np.random.default_rng(0), **kwargs,
    )
    layer.counters = (weights * layer.connect_level).astype(np.int16)
    layer.thresholds = np.array(thresholds, dtype=np.int32)
    layer._sync_weights()
    return layer


def test_forward():
    layer = make_layer([[1, 0, 1, -1], [0, 1, -1, 1]], [1, 1])
    inputs = np.array([[True, False, True, False], [False, True, False, True]])
    outputs, _ = layer.forward(inputs)
    np.testing.assert_array_equal(outputs, [[True, False], [False, True]])
    np.testing.assert_array_equal(layer.margins(inputs), [[1, -2], [-2, 1]])


def test_synapse_needs_repeated_evidence():
    layer = make_layer([[0, 0]], [0], connect_level=3)
    inputs = np.array([[True, False]])
    for step in range(3):
        _, state = layer.forward(inputs)
        layer.backward(state, np.array([[1.0]]), np.array([[0.0]]))
        layer.update()
        if step < 2:
            assert layer.weights[0, 0] == 0
    assert layer.weights[0, 0] == -1


def test_backward_blames_only_near_threshold():
    # neuron 0 misses firing by one, neuron 1 by three
    layer = make_layer([[1, 1, 0, 0], [0, 0, 1, 1]], [1, 3], backprop_margin=1)
    inputs = np.array([[False, False, False, False]])
    _, state = layer.forward(inputs)
    too_much, not_enough = layer.backward(state, np.zeros((1, 2)), np.ones((1, 2)))
    # one flip of input 0 or 1 fixes neuron 0: each gets half; neuron 1 is too far, no blame
    np.testing.assert_array_almost_equal(not_enough, [[0.5, 0.5, 0, 0]])
    np.testing.assert_array_almost_equal(too_much, [[0, 0, 0, 0]])


def test_backward_counts_collateral_damage():
    # input 0 would fix neuron 0, but neuron 1 (fine, sitting right at threshold) would break
    layer = make_layer([[1, 1], [-1, 0]], [1, 0], backprop_margin=1)
    inputs = np.array([[False, False]])
    _, state = layer.forward(inputs)
    _, not_enough = layer.backward(state, np.zeros((1, 2)), np.array([[1.0, 0.0]]))
    assert not_enough[0, 0] == 0
    assert not_enough[0, 1] == 0.5


def test_learns_parity_4():
    X = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
    y = X.sum(axis=1) % 2
    targets = np.stack([y == 0, y == 1], axis=1)
    rng = np.random.default_rng(0)
    small = dict(connect_level=2, counter_max=16, init_range=2)
    network = Sequence([
        CounterDense(4, 32, accept_distance=0, rng=rng, **small),
        CounterDense(32, 2, stabilize_margin=2, rng=rng, **small),
    ])
    for _ in range(300):
        for batch in np.split(rng.permutation(16), 8):
            out, state = network.forward(X[batch])
            network.backward(state, out & ~targets[batch], ~out & targets[batch])
            network.update()
    # a single threshold unit can't get more than 11/16 right
    assert np.mean(network.margins(X).argmax(axis=1) == y) >= 14 / 16
