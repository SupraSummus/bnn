import numpy as np

from bnn.abstract import Sequence
from bnn.counter import CounterDense, train_epoch
from bnn.counter_reference import ReferenceCounterDense


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


def test_consolidate_keeps_average_state():
    layer = make_layer([[0, 0, 0]], [0])
    layer.start_tally()
    for weights, threshold in [([1, -1, 1], 2), ([1, -1, -1], 2), ([0, 0, 1], 3), ([1, -1, -1], 3)]:
        layer.counters = np.array([weights], dtype=np.int16) * layer.connect_level
        layer._sync_weights()
        layer.thresholds[:] = threshold
        layer.update()
    layer.consolidate()
    # +1 three times out of four; -1 three times; +1 and -1 twice each
    np.testing.assert_array_equal(layer.weights, [[1, -1, 0]])
    assert layer.thresholds[0] == 2  # 2.5 rounds to even


def random_inputs(rng, batch, features, density=0.4):
    return rng.random((batch, features)) < density


def assert_same_state(fast, reference):
    np.testing.assert_array_equal(fast.counters, reference.counters)
    np.testing.assert_array_equal(fast.weights, reference.weights)
    np.testing.assert_array_equal(fast.thresholds, reference.thresholds)
    np.testing.assert_array_equal(fast.threshold_counters, reference.threshold_counters)


def test_kernels_match_reference_step():
    # 150 inputs: several 64-bit words, the last one partial
    rng = np.random.default_rng(1)
    for kwargs in [
        {},
        {"accept_distance": 0, "backprop_margin": 3},
        {"stabilize_margin": 2, "connect_level": 2, "init_range": 3},
    ]:
        fast = CounterDense(150, 7, rng=np.random.default_rng(2), **kwargs)
        reference = ReferenceCounterDense(150, 7, rng=np.random.default_rng(2), **kwargs)
        fast.thresholds = rng.integers(-3, 25, 7).astype(np.int32)
        reference.thresholds = fast.thresholds.copy()
        for _ in range(5):
            inputs = random_inputs(rng, 16, 150)
            too_much = rng.random((16, 7)).astype(np.float32) * (rng.random((16, 7)) < 0.5)
            not_enough = rng.random((16, 7)).astype(np.float32) * (rng.random((16, 7)) < 0.5)
            out, state = fast.forward(inputs)
            ref_out, ref_state = reference.forward(inputs)
            np.testing.assert_array_equal(out, ref_out)
            np.testing.assert_array_equal(fast.margins(inputs), reference.margins(inputs))
            # blame sums shares over outputs; BLAS may add them in another order
            np.testing.assert_allclose(
                fast.backward(state, too_much, not_enough),
                reference.backward(ref_state, too_much, not_enough),
                rtol=1e-6,
            )
            fast.update()
            reference.update()
            assert_same_state(fast, reference)


def train(layer_class, X, y, hidden, epochs, batch_size, fused=False, **kwargs):
    rng = np.random.default_rng(0)
    sizes = [X.shape[1], *hidden]
    layers = [
        layer_class(n_in, n_out, accept_distance=0, rng=rng, **kwargs)
        for n_in, n_out in zip(sizes, sizes[1:])
    ]
    layers.append(layer_class(sizes[-1], 2, stabilize_margin=2, rng=rng, **kwargs))
    network = Sequence(layers)
    targets = np.stack([y == 0, y == 1], axis=1)
    for epoch in range(epochs):
        if epoch == epochs - 1:
            for layer in layers:
                layer.start_tally()
        order = rng.permutation(len(X))
        if fused:
            train_epoch(layers, X, targets, order, batch_size)
            continue
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            out, state = network.forward(X[batch])
            network.backward(state, (out & ~targets[batch]).astype(np.float32), (~out & targets[batch]).astype(np.float32))
            network.update()
    for layer in layers:
        layer.consolidate()
    return layers


def test_kernels_match_reference_training():
    # two output classes: the blame passed down sums at most two shares, so float order can't differ
    rng = np.random.default_rng(3)
    X = random_inputs(rng, 200, 70)
    y = (X[:, :5].sum(axis=1) + X[:, 60]) % 2
    for hidden in [(), (24,)]:
        fast = train(CounterDense, X, y, hidden, epochs=4, batch_size=8, connect_level=2, counter_max=16, init_range=2)
        reference = train(ReferenceCounterDense, X, y, hidden, epochs=4, batch_size=8, connect_level=2, counter_max=16, init_range=2)
        for f, r in zip(fast, reference):
            assert_same_state(f, r)


def test_fused_epoch_matches_layer_by_layer():
    rng = np.random.default_rng(4)
    X = random_inputs(rng, 150, 130)
    y = (X[:, :3].sum(axis=1) + X[:, 100]) % 2
    for hidden in [(), (20,), (20, 12)]:
        fused = train(CounterDense, X, y, hidden, epochs=3, batch_size=7, fused=True)
        stepped = train(CounterDense, X, y, hidden, epochs=3, batch_size=7)
        for f, s in zip(fused, stepped):
            assert_same_state(f, s)
