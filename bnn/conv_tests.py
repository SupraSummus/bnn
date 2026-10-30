import numpy as np

from bnn.abstract import Sequence
from bnn.conv import ConvCounter, OrPool, train_epoch
from bnn.counter import CounterDense


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


class ArrayConv(ConvCounter):
    """ConvCounter as first written: patches as rows of a CounterDense, numpy glue."""

    def margins(self, inputs):
        return self._to_image(self.dense.margins(inputs[:, self.index].reshape(-1, self.index.shape[1])))

    def forward(self, inputs):
        rows, state = self.dense.forward(inputs[:, self.index].reshape(-1, self.index.shape[1]))
        return self._to_image(rows), state

    def _to_image(self, rows):
        rows = rows.reshape(-1, self.positions, self.filters).transpose(0, 2, 1)
        return np.ascontiguousarray(rows).reshape(-1, self.output_features)

    def backward(self, state, too_much, not_enough):
        rate = np.float32(self.vote_rate)
        too_much, not_enough = self.dense.backward(state, self._to_rows(too_much) * rate, self._to_rows(not_enough) * rate)
        return self._scatter(too_much / rate), self._scatter(not_enough / rate)

    def _scatter(self, q_rows):
        out = np.zeros((q_rows.shape[0] // self.positions, self.input_features), np.float32)
        for b in range(out.shape[0]):
            for p in range(self.positions):
                for j in range(self.index.shape[1]):
                    out[b, self.index[p, j]] += q_rows[b * self.positions + p, j]
        return out


class ArrayPool(OrPool):
    def forward(self, inputs):
        block = inputs[:, self.index]
        return block.any(axis=2), block

    def backward(self, block, too_much, not_enough):
        out_tm = np.zeros((block.shape[0], self.input_features), np.float32)
        out_ne = np.zeros((block.shape[0], self.input_features), np.float32)
        flat = self.index.ravel()
        for b in range(block.shape[0]):
            out_tm[b] = np.bincount(flat, (block[b] * too_much[b][:, None]).ravel(), self.input_features)
            out_ne[b] = np.bincount(flat, np.broadcast_to(not_enough[b][:, None], block[b].shape).ravel(), self.input_features)
        return out_tm, out_ne


def _conv_net(kinds, seed):
    """conv 2x9x9 -> 6 filters 3x3 -> pool 2 (7 -> 3, one row and column left out) -> 5 filters 2x2 stride 1 -> readout."""
    rng = np.random.default_rng(seed)
    kw = dict(accept_distance=0, homeostasis=3, threshold_level=1, window=3, rng=rng)
    conv, pool = kinds
    layers = [conv(2, 9, 9, 6, 3, 1, **kw), pool(6, 7, 7, 2), conv(6, 3, 3, 5, 2, 1, vote_rate=0.5, **kw)]
    for layer in layers[::2]:
        layer.dense.counters[:] = rng.integers(-6, 7, layer.dense.counters.shape)
        layer.dense._sync_weights()
    return [*layers, CounterDense(layers[-1].output_features, 3, stabilize_margin=1, rng=rng)]


def test_fast_paths_match_array_version():
    """Kernels, and train_epoch without the first layer's input pressure: the same state, bit for bit."""
    rng = np.random.default_rng(5)
    x = rng.random((96, 2 * 81)) < 0.4
    targets = np.eye(3, dtype=bool)[rng.integers(0, 3, 96)]
    order = rng.permutation(96)
    want = _conv_net((ArrayConv, ArrayPool), 6)
    start, got, fused = (_conv_net((ConvCounter, OrPool), 6) for _ in range(3))
    for epoch in range(3):
        if epoch == 2:
            for layer in want + got + fused:
                layer.start_tally()
        for s in range(0, 96, 8):
            qs = []
            for net in (want, got):
                seq = Sequence(net)
                _, state = seq.forward(x[order[s:s + 8]])
                qs.append(seq.backward(state, *net[-1].output_pressure(state[-1], targets[order[s:s + 8]])))
                seq.update()
            for a, b in zip(*qs):
                np.testing.assert_array_equal(a, b)
        train_epoch(fused, x, targets, order, 8)
    for net in (got, fused):
        inputs = x
        for a, b in zip(want, net):
            np.testing.assert_array_equal(a.margins(inputs), b.margins(inputs))
            inputs = a.margins(inputs) >= 0
            a, b = getattr(a, "dense", a), getattr(b, "dense", b)
            if isinstance(a, CounterDense):
                for name in ("counters", "thresholds", "threshold_counters", "rng_state", "homeo_mean", "homeo_run"):
                    np.testing.assert_array_equal(getattr(a, name), getattr(b, name), name)
                np.testing.assert_array_equal(a.tally[0], b.tally[0])
    for a, b in zip(want[::2], start[::2]):  # every conv layer learned
        assert (a.dense.counters != b.dense.counters).any()
