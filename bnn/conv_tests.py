import numpy as np

from bnn.conv import ConvCounter, OrPool


def test_patches_and_layout():
    conv = ConvCounter(2, 4, 5, filters=3, kernel=2, stride=2, rng=np.random.default_rng(0))
    assert (conv.out_height, conv.out_width) == (2, 2)
    image = np.zeros((1, 2 * 4 * 5), dtype=bool)
    image[0, 1 * 20 + 2 * 5 + 3] = True  # channel 1, row 2, column 3: patch (1, 1), offset (0, 1)
    patches = conv.patches(image)
    assert patches.shape == (4, 8)
    assert patches.sum() == 1 and patches[3, 4 + 1]


def test_shared_filter_matches_per_position_margins():
    rng = np.random.default_rng(1)
    conv = ConvCounter(2, 6, 6, filters=4, kernel=3, stride=1, rng=rng)
    x = rng.random((5, 72)) < 0.5
    out, _ = conv.forward(x)
    margins = conv.margins(x).reshape(5, 4, 4, 4)
    for f in range(4):
        for i in range(4):
            for j in range(4):
                patch = x.reshape(5, 2, 6, 6)[:, :, i:i + 3, j:j + 3].reshape(5, -1)
                want = patch.astype(int) @ conv.dense.weights[f].astype(int) - conv.dense.thresholds[f]
                np.testing.assert_array_equal(margins[:, f, i, j], want)
    np.testing.assert_array_equal(out, conv.margins(x) >= 0)


def test_pressure_sums_over_patches():
    conv = ConvCounter(1, 3, 3, filters=1, kernel=2, stride=1, window=10, vote_rate=0.25, rng=np.random.default_rng(2))
    conv.dense.counters[:] = conv.dense.connect_level  # all four synapses exciting
    conv.dense.thresholds[:] = 1
    conv.dense._sync_weights()
    x = np.zeros((1, 9), dtype=bool)  # every patch at distance 0 from firing
    _, state = conv.forward(x)
    _, not_enough = conv.backward(state, np.zeros((1, 4)), np.ones((1, 4)))
    # the centre pixel is in all four patches, corners in one
    np.testing.assert_array_almost_equal(not_enough.reshape(3, 3), [[1, 2, 1], [2, 4, 2], [1, 2, 1]])


def test_votes_scale_with_vote_rate():
    """With every row asking, a filter gets about rows × vote_rate votes per update."""
    rng = np.random.default_rng(3)
    conv = ConvCounter(1, 10, 10, filters=1, kernel=1, stride=1, counter_max=10**4, rng=rng)  # 100 positions
    conv.dense.counters[:] = 0
    conv.dense.thresholds[:] = 1  # off, one step from firing
    conv.dense._sync_weights()
    x = np.ones((40, 100), dtype=bool)  # 4000 rows, all asking to turn on
    _, state = conv.forward(x)
    conv.backward(state, np.zeros((40, 100)), np.ones((40, 100)))
    votes = conv.dense.counter_delta[0, 0]
    assert 20 <= votes <= 60  # 4000 / 100 = 40 expected


def test_or_pool():
    pool = OrPool(1, 4, 4, size=2)
    x = np.zeros((1, 16), dtype=bool)
    x[0, 5] = True  # row 1, column 1: block (0, 0)
    out, block = pool.forward(x)
    np.testing.assert_array_equal(out, [[True, False, False, False]])
    too_much, not_enough = pool.backward(block, np.array([[1.0, 0, 0, 0]]), np.array([[0, 2.0, 0, 0]]))
    assert too_much[0, 5] == 1 and too_much.sum() == 1  # only the unit that is on
    np.testing.assert_array_equal(not_enough.reshape(4, 4)[:2, 2:], 2)  # every unit of block (0, 1)
    assert not_enough.sum() == 8
