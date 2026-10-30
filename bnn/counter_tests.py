import numpy as np

from bnn.abstract import Sequence
from bnn.counter import CounterDense, Ratchet, train_epoch
from bnn.counter_reference import ReferenceCounterDense


def make_layer(weights, thresholds, layer_class=CounterDense, **kwargs):
    weights = np.array(weights, dtype=np.int8)
    layer = layer_class(
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


def test_pressure_passes_through_window():
    # distances to flipping 0, 1 and 2; only outputs closer than the window pass pressure, 1 / (distance + 1) of it
    layer = make_layer([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], [1, 2, 3], window=2)
    _, state = layer.forward(np.zeros((1, 6), dtype=bool))
    too_much, not_enough = layer.backward(state, np.zeros((1, 3)), np.ones((1, 3)))
    np.testing.assert_array_almost_equal(not_enough, [[1, 1, 0.5, 0.5, 0, 0]])
    np.testing.assert_array_almost_equal(too_much, np.zeros((1, 6)))


def test_pressure_counts_collateral_damage():
    # input 0 would turn output 0 on, but also turn off output 1, which is right and should stay on
    layer = make_layer([[1, 1], [-1, 0]], [1, 0])
    _, state = layer.forward(np.zeros((1, 2), dtype=bool))
    too_much, not_enough = layer.backward(state, np.zeros((1, 2)), np.ones((1, 2)))
    np.testing.assert_array_almost_equal(not_enough, [[0, 1]])
    np.testing.assert_array_almost_equal(too_much, [[0, 0]])


def test_pressure_from_several_outputs_scales_with_sqrt():
    layer = make_layer([[1], [1]], [1, 1])
    _, state = layer.forward(np.zeros((1, 1), dtype=bool))
    _, not_enough = layer.backward(state, np.zeros((1, 2)), np.ones((1, 2)))
    np.testing.assert_array_almost_equal(not_enough, [[np.sqrt(2)]])


def test_surrogates():
    # outputs at distance 0, 1 and 2 from firing, window 2; input 0 feeds the first two
    weights = [[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]]
    for surrogate, normalize_reach, want in [
        ("triangle", True, [1.5 / np.sqrt(2), 1, 0.5, 0]),
        ("box", False, [2, 1, 1, 0]),
        ("none", False, [2, 1, 1, 1]),
    ]:
        for layer_class in (CounterDense, ReferenceCounterDense):
            layer = make_layer(weights, [1, 2, 3], layer_class, window=2, surrogate=surrogate, normalize_reach=normalize_reach)
            _, state = layer.forward(np.zeros((1, 4), dtype=bool))
            _, not_enough = layer.backward(state, np.zeros((1, 3)), np.ones((1, 3)))
            np.testing.assert_array_almost_equal(not_enough, [want])


def test_output_pressure_hinge():
    # margins 5, 0 and -1
    for layer_class in (CounterDense, ReferenceCounterDense):
        layer = make_layer([[0], [0], [0]], [-5, 0, 1], layer_class=layer_class, hinge=2)
        _, state = layer.forward(np.zeros((2, 1), dtype=bool))
        targets = np.array([[True, True, True], [False, False, False]])
        too_much, not_enough = layer.output_pressure(state, targets)
        # right by 5 sends nothing; right by 0 (margin 0 or -1) and wrong push toward the target
        np.testing.assert_array_equal(not_enough - too_much, [[0, 1, 1], [-1, -1, -1]])


def test_homeostasis_revives_constant_unit():
    rng = np.random.default_rng(0)
    for layer_class in (CounterDense, ReferenceCounterDense):
        layer = make_layer([[1, 1]], [-5], layer_class=layer_class, homeostasis=8)
        inputs = rng.random((400, 2)) < 0.5
        for batch in np.split(inputs, 40):
            _, state = layer.forward(batch)
            layer.backward(state, np.zeros((10, 1)), np.zeros((10, 1)))
            layer.update()
        # always on at first; now its threshold sits at its mean input sum (1)
        assert layer.thresholds[0] == 1
        assert 0 < layer.forward(inputs)[0].mean() < 1


def test_learns_parity_4():
    X = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
    y = X.sum(axis=1) % 2
    targets = np.stack([y == 0, y == 1], axis=1)
    rng = np.random.default_rng(0)
    network = Sequence([
        CounterDense(4, 32, accept_distance=0, homeostasis=64, rng=rng),
        CounterDense(32, 2, stabilize_margin=2, rng=rng),
    ])
    for _ in range(300):
        for batch in np.split(rng.permutation(16), 8):
            _, state = network.forward(X[batch])
            network.backward(state, *network.layers[-1].output_pressure(state[-1], targets[batch]))
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
    np.testing.assert_array_equal(fast.homeo_mean, reference.homeo_mean)
    np.testing.assert_array_equal(fast.homeo_run, reference.homeo_run)


def test_kernels_match_reference_step():
    # 150 inputs: several 64-bit words, the last one partial
    rng = np.random.default_rng(1)
    for kwargs in [
        {},
        {"accept_distance": 0, "window": 30},
        {"stabilize_margin": 2, "connect_level": 2, "init_range": 3},
        {"accept_distance": 1, "homeostasis": 5},
        {"surrogate": "box", "normalize_reach": False, "window": 20},
        {"surrogate": "none", "accept_distance": 10, "stabilize_margin": 10},
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
            # the reference adds pressure in the kernels' order: exact, not just close
            for f, r in zip(fast.backward(state, too_much, not_enough), reference.backward(ref_state, too_much, not_enough)):
                np.testing.assert_array_equal(f, r)
            fast.update()
            reference.update()
            assert_same_state(fast, reference)


def train(layer_class, X, y, hidden, epochs, batch_size, fused=False, homeostasis=0, **kwargs):
    rng = np.random.default_rng(0)
    sizes = [X.shape[1], *hidden]
    layers = [
        layer_class(n_in, n_out, accept_distance=0, homeostasis=homeostasis, rng=rng, **kwargs)
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
            _, state = network.forward(X[batch])
            network.backward(state, *layers[-1].output_pressure(state[-1], targets[batch]))
            network.update()
    for layer in layers:
        layer.consolidate()
    return layers


def test_kernels_match_reference_training():
    rng = np.random.default_rng(3)
    X = random_inputs(rng, 200, 70)
    y = (X[:, :5].sum(axis=1) + X[:, 60]) % 2
    for hidden, kwargs in [((), {}), ((24,), {}), ((24, 12), {}), ((24, 12), {"surrogate": "none", "normalize_reach": False})]:
        fast = train(CounterDense, X, y, hidden, epochs=4, batch_size=8, homeostasis=16, **kwargs)
        reference = train(ReferenceCounterDense, X, y, hidden, epochs=4, batch_size=8, homeostasis=16, **kwargs)
        for f, r in zip(fast, reference):
            assert_same_state(f, r)


def test_fused_epoch_matches_layer_by_layer():
    rng = np.random.default_rng(4)
    X = random_inputs(rng, 150, 130)
    y = (X[:, :3].sum(axis=1) + X[:, 100]) % 2
    for hidden, homeostasis, kwargs in [
        ((), 0, {}), ((20,), 0, {}), ((20, 12), 0, {}), ((20, 12), 16, {}),
        ((20, 12), 16, {"surrogate": "box", "normalize_reach": False}),
    ]:
        fused = train(CounterDense, X, y, hidden, epochs=3, batch_size=7, fused=True, homeostasis=homeostasis, **kwargs)
        stepped = train(CounterDense, X, y, hidden, epochs=3, batch_size=7, homeostasis=homeostasis, **kwargs)
        for f, s in zip(fused, stepped):
            assert_same_state(f, s)


def test_ratchet_rolls_back_a_collapse():
    X = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
    y = X[:, 0].astype(np.int64)
    good = make_layer([[-1, 0, 0, 0], [1, 0, 0, 0]], [0, 1])  # class 1 iff bit 0
    layers = [good]
    ratchet = Ratchet(X, y, tolerance=0.25)
    ratchet.check(layers)
    assert ratchet.best_accuracy == 1
    # a small dip stays, a collapse goes back to the best state
    layers[0] = make_layer([[-1, -1, -1, 0], [1, 0, 0, 0]], [0, 1])  # wrong on 2 of 16
    ratchet.check(layers)
    assert ratchet.rollbacks == 0 and ratchet.accuracy(layers) == 0.875
    layers[0] = make_layer([[1, 0, 0, 0], [-1, 0, 0, 0]], [1, 0])  # all wrong
    ratchet.check(layers)
    assert ratchet.rollbacks == 1 and ratchet.accuracy(layers) == 1
    assert layers[0] is not ratchet.best[0]  # a copy: training it doesn't touch the saved state
