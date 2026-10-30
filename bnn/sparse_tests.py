import numpy as np

from bnn.abstract import Sequence
from bnn.counter import CounterDense
from bnn.sparse import SparseCounter


def as_sparse(dense: CounterDense, **kwargs) -> SparseCounter:
    """A SparseCounter with every input in a slot and dense's state: it should be the same layer."""
    layer = SparseCounter(
        dense.input_features, dense.output_features, dense.input_features,
        connect_level=dense.connect_level, counter_max=dense.counter_max, threshold_level=dense.threshold_level,
        window=dense.window, accept_distance=dense.accept_distance, stabilize_margin=dense.stabilize_margin,
        hinge=dense.hinge, homeostasis=dense.homeostasis, surrogate=dense.surrogate,
        normalize_reach=dense.normalize_reach, rng=np.random.default_rng(0), **kwargs,
    )
    layer.pointers[:] = np.arange(dense.input_features)
    layer.counters[:] = dense.counters
    layer.thresholds[:] = dense.thresholds
    layer.rng_state[:] = dense.rng_state
    layer.sync_weights()
    return layer


def train_batches(net, x, targets, batch_size=8):
    for start in range(0, len(x), batch_size):
        out, state = net.forward(x[start:start + batch_size])
        too_much, not_enough = net.layers[-1].output_pressure(state[-1], targets[start:start + batch_size])
        net.backward(state, too_much, not_enough)
        net.update()


def test_all_slots_is_counter_dense():
    rng = np.random.default_rng(1)
    x = rng.random((256, 12)) < 0.5
    targets = np.stack([x[:, 0] ^ x[:, 1], x[:, 2] & x[:, 3]], axis=1)
    dense = Sequence([
        CounterDense(12, 16, accept_distance=0, homeostasis=16, rng=rng),
        CounterDense(16, 2, stabilize_margin=2, surrogate="none", normalize_reach=False, rng=rng),
    ])
    sparse = Sequence([as_sparse(layer) for layer in dense.layers])
    for _ in range(5):
        train_batches(dense, x, targets)
        train_batches(sparse, x, targets)
    for d, s in zip(dense.layers, sparse.layers):
        np.testing.assert_array_equal(d.counters, s.counters)
        np.testing.assert_array_equal(d.thresholds, s.thresholds)
    np.testing.assert_array_equal(dense.margins(x), sparse.margins(x))
    _, state_d = dense.forward(x[:8])
    _, state_s = sparse.forward(x[:8])
    pressure = (np.zeros((8, 2)), np.ones((8, 2)))
    for got, want in zip(sparse.backward(state_s, *pressure), dense.backward(state_d, *pressure)):
        np.testing.assert_array_equal(got, want)


def test_margins_follow_pointers():
    layer = SparseCounter(5, 2, 2, connect_level=1, rng=np.random.default_rng(0))
    layer.pointers[:] = [[0, 3], [4, 1]]
    layer.counters[:] = [[1, -1], [1, 0]]
    layer.thresholds[:] = [0, 1]
    layer.sync_weights()
    x = np.array([[1, 0, 0, 1, 0], [1, 1, 0, 0, 1]], dtype=bool)
    np.testing.assert_array_equal(layer.margins(x), [[0, -1], [1, 0]])


def test_rewire_moves_absent_slots_onto_active_inputs():
    rng = np.random.default_rng(2)
    # slow thresholds and counters: the outputs keep asking until a slot moves onto the input that is on
    layer = SparseCounter(64, 4, 4, connect_level=32, init_range=0, threshold_level=1000, rewire=1.0, rng=rng)
    layer.pointers[:] = np.arange(16).reshape(4, 4) + 16
    layer.thresholds[:] = 1  # every output off and wanted on
    x = np.zeros((32, 64), dtype=bool)
    x[:, 7] = True
    targets = np.ones((32, 4), dtype=bool)
    for start in range(0, 32, 4):
        _, state = layer.forward(x[start:start + 4])
        layer.backward(state, *layer.output_pressure(state, targets[start:start + 4]))
        layer.update()
    # input 7 is the only one ever on: every output wired an exciting synapse to it, once
    assert layer.moved == 4
    for o in range(4):
        assert list(layer.pointers[o]).count(7) == 1
        assert layer.weights[o, list(layer.pointers[o]).index(7)] == 1
        assert len(set(layer.pointers[o])) == 4


def test_rewire_keeps_synapses():
    layer = SparseCounter(64, 1, 2, connect_level=2, rewire=1.0, rng=np.random.default_rng(3))
    layer.counters[:] = [[2, -2]]  # both slots hold synapses: nothing to move
    layer.sync_weights()
    before = layer.pointers.copy()
    x = np.ones((4, 64), dtype=bool)
    _, state = layer.forward(x)
    layer.backward(state, np.ones((4, 1)), np.zeros((4, 1)))
    layer.update()
    assert layer.moved == 0
    np.testing.assert_array_equal(layer.pointers, before)


def test_consolidate_keeps_majority_state():
    layer = SparseCounter(4, 1, 2, connect_level=2, rewire=1.0, rng=np.random.default_rng(4))
    layer.pointers[:] = [[0, 1]]
    layer.counters[:] = [[2, 0]]
    layer.sync_weights()
    layer.start_tally()
    x = np.array([[1, 0, 0, 0]], dtype=bool)
    # "too much" three times: the active slot loses its synapse on the first vote,
    # the inactive one (moving the other way) gains one on the second: on in 0 and 2 of 3 tallied states
    for _ in range(3):
        _, state = layer.forward(x)
        layer.backward(state, np.ones((1, 1)), np.zeros((1, 1)))
        layer.update()
    assert layer.moved == 0  # no rewiring while tallying
    np.testing.assert_array_equal(layer.pointers, [[0, 1]])
    layer.consolidate()
    np.testing.assert_array_equal(layer.weights, [[0, 1]])
