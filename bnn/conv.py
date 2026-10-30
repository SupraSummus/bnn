"""
Convolution as weight sharing: a CounterDense layer applied to every patch of an image.

Patches become rows, so one filter's counters collect the votes of every position,
and the pressure on a pixel is the sum over the patches that contain it.
A filter sees positions × batch rows per update; to keep its counters' hysteresis,
a row votes with probability |pressure| × `vote_rate` (default 1 / positions: a dense neuron's rate per sample).
Images are flat, channel first: index c * height * width + row * width + column
(what `benchmarks.datasets.fashion_bits` gives, with 2 channels); so are the outputs, one channel per filter.
Kernels read patches straight from the image; `conv_tests.ArrayConv` is the same layer as numpy glue
around CounterDense, and the tests check the two agree bit for bit.
"""

import numpy as np
from numba import njit

from . import counter_kernels as kernels
from .abstract import ArrayBool, ArrayFloat, Layer
from .counter import CounterDense
from .counter_kernels import _popcount, may_vote, next_uniform, passes_down


@njit(cache=True)
def gather(x, index):
    """Patches (batch * positions, patch size) of images (batch, n_in)."""
    positions, size = index.shape
    out = np.empty((x.shape[0] * positions, size), np.bool_)
    for b in range(x.shape[0]):
        for p in range(positions):
            for j in range(size):
                out[b * positions + p, j] = x[b, index[p, j]]
    return out


@njit(cache=True)
def to_image(rows, positions):
    """(batch * positions, filters) -> (batch, filters * positions), channel first."""
    n_rows, filters = rows.shape
    out = np.empty((n_rows // positions, filters * positions), rows.dtype)
    for r in range(n_rows):
        b, p = divmod(r, positions)
        for f in range(filters):
            out[b, f * positions + p] = rows[r, f]
    return out


@njit(cache=True)
def conv_forward(x, index, exc, inh, thresholds):
    """
    Margins of every filter at every patch, as rows (batch * positions, filters),
    and the outputs as an image; patches are packed straight from the image.
    """
    batch = x.shape[0]
    positions, size = index.shape
    filters, words = exc.shape
    margins = np.empty((batch * positions, filters), np.int32)
    out = np.empty((batch, filters * positions), np.bool_)
    bits = np.empty(words, np.uint64)
    for b in range(batch):
        for p in range(positions):
            bits[:] = 0
            for j in range(size):  # no branch: pixels are unpredictable
                bits[j >> 6] |= np.uint64(x[b, index[p, j]]) << np.uint64(j & 63)
            r = b * positions + p
            if words == 1:  # small patches: a loop over filters alone vectorizes
                xb = bits[0]
                for o in range(filters):
                    margins[r, o] = _popcount(xb & exc[o, 0]) - _popcount(xb & inh[o, 0]) - thresholds[o]
            else:
                for o in range(filters):
                    s = 0
                    for w in range(words):
                        s += _popcount(bits[w] & exc[o, w]) - _popcount(bits[w] & inh[o, w])
                    margins[r, o] = s - thresholds[o]
        for o in range(filters):
            for p in range(positions):
                out[b, o * positions + p] = margins[b * positions + p, o] >= 0
    return margins, out


@njit(cache=True)
def row_pressure(too_much, not_enough, rate, positions):
    """Netted pressure rows (batch * positions, filters) from image-layout (too_much, not_enough), each scaled by rate."""
    batch, n_out = too_much.shape
    filters = n_out // positions
    out = np.empty((batch * positions, filters), np.float32)
    for b in range(batch):
        for p in range(positions):
            for o in range(filters):
                k = o * positions + p
                out[b * positions + p, o] = np.float32(not_enough[b, k] * rate) - np.float32(too_much[b, k] * rate)
    return out


@njit(cache=True)
def conv_votes(x, index, margins, pressure, accept_distance, stabilize_margin,
               counter_delta, threshold_delta, touched, rng_state):
    """The votes of counter_kernels.backward over the patch rows, patches read from the image `x`."""
    positions, size = index.shape
    for r in range(margins.shape[0]):
        b, p = divmod(r, positions)
        for o in range(margins.shape[1]):
            q = pressure[r, o]
            if q == 0:
                continue
            vote_ok = may_vote(q, margins[r, o], accept_distance, stabilize_margin)
            if vote_ok and next_uniform(rng_state) < abs(np.float64(q)):
                vote = np.int32(1) if q > 0 else np.int32(-1)
                for j in range(size):
                    counter_delta[o, j] += vote * (2 * np.int32(x[b, index[p, j]]) - 1)
                threshold_delta[o] -= vote
                touched[o] = True


@njit(cache=True)
def conv_blame(margins, pressure, weights, window, surrogate, normalize_reach, rate, index, n_in):
    """
    The input pressure of counter_kernels.backward over the patch rows, scaled back up by 1 / rate
    and summed onto the image as (too_much, not_enough) over the patches that contain a pixel.
    """
    positions, size = index.shape
    batch = margins.shape[0] // positions
    too_much = np.zeros((batch, n_in), np.float32)
    not_enough = np.zeros((batch, n_in), np.float32)
    q_in = np.zeros(size, np.float32)
    reach = np.zeros(size, np.int32)
    # weights as float32 and presence as int32: the inner loop vectorizes
    w_float = weights.astype(np.float32)
    w_present = (weights != 0).astype(np.int32)
    zero = np.float32(0)
    for r in range(margins.shape[0]):
        b, p = divmod(r, positions)
        q_in[:] = 0
        reach[:] = 0
        for o in range(margins.shape[1]):
            q = pressure[r, o]
            if q == 0:
                continue
            passed, c = passes_down(q, margins[r, o], window, surrogate)
            if passed:
                for j in range(size):  # absent synapses add 0, as in the reference
                    q_in[j] += c * w_float[o, j]
                    reach[j] += w_present[o, j]
        for j in range(size):
            q = q_in[j]
            if normalize_reach and reach[j] > 1:
                q /= np.float32(np.sqrt(np.float64(reach[j])))
            too_much[b, index[p, j]] += max(-q, zero) / rate
            not_enough[b, index[p, j]] += max(q, zero) / rate
    return too_much, not_enough


@njit(cache=True)
def pool_forward(x, index):
    batch = x.shape[0]
    n_out, size = index.shape
    out = np.zeros((batch, n_out), np.bool_)
    for b in range(batch):
        for u in range(n_out):
            on = False
            for k in range(size):  # no branches: pixels are unpredictable
                on |= x[b, index[u, k]]
            out[b, u] = on
    return out


@njit(cache=True)
def pool_backward(x, index, too_much, not_enough):
    """Blocks don't overlap: each input gets the pressure of its block's output, "too much" only if on."""
    batch, n_in = x.shape
    n_out, size = index.shape
    out_tm = np.zeros((batch, n_in), np.float32)
    out_ne = np.zeros((batch, n_in), np.float32)
    zero = np.float32(0)
    for b in range(batch):
        for u in range(n_out):
            tm, ne = np.float32(too_much[b, u]), np.float32(not_enough[b, u])
            for k in range(size):
                i = index[u, k]
                out_tm[b, i] = tm if x[b, i] else zero
                out_ne[b, i] = ne
    return out_tm, out_ne


class ConvCounter(Layer):
    def __init__(self, channels, height, width, filters, kernel, stride=1, vote_rate=None, **kwargs):
        """
        Valid convolution (no padding). `kwargs` go to the CounterDense that holds the filters
        (`input_features` = kernel * kernel * channels, one output per filter).
        """
        self.channels, self.height, self.width = channels, height, width
        self.filters, self.kernel, self.stride = filters, kernel, stride
        self.out_height = (height - kernel) // stride + 1
        self.out_width = (width - kernel) // stride + 1
        self.positions = self.out_height * self.out_width
        self.input_features = channels * height * width
        self.output_features = filters * self.positions
        c, di, dj = np.meshgrid(np.arange(channels), np.arange(kernel), np.arange(kernel), indexing="ij")
        patch = (c * height * width + di * width + dj).ravel()
        i, j = np.meshgrid(np.arange(self.out_height), np.arange(self.out_width), indexing="ij")
        corner = (i * stride * width + j * stride).ravel()
        self.index = np.ascontiguousarray(corner[:, None] + patch[None, :])  # (positions, patch size)
        self.vote_rate = 1 / self.positions if vote_rate is None else vote_rate
        self.dense = CounterDense(patch.size, filters, **kwargs)

    def patches(self, inputs: ArrayBool) -> ArrayBool:
        assert inputs.ndim == 2 and inputs.shape[1] == self.input_features
        return gather(np.ascontiguousarray(inputs, dtype=np.bool_), self.index)

    def margins(self, inputs: ArrayBool) -> np.ndarray:
        return to_image(self.forward(inputs)[1][1], self.positions)

    def forward(self, inputs: ArrayBool):
        assert inputs.ndim == 2 and inputs.shape[1] == self.input_features
        inputs = np.ascontiguousarray(inputs, dtype=np.bool_)
        d = self.dense
        margins, out = conv_forward(inputs, self.index, d.exc, d.inh, d.thresholds)
        return out, (inputs, margins)

    def output_pressure(self, state, targets: ArrayBool):
        too_much, not_enough = self.dense.output_pressure((None, None, state[1]), self._to_rows(targets))
        return to_image(too_much, self.positions), to_image(not_enough, self.positions)

    def _to_rows(self, image):
        return np.ascontiguousarray(image.reshape(-1, self.filters, self.positions).transpose(0, 2, 1)).reshape(-1, self.filters)

    def backward(self, state, too_much: ArrayFloat, not_enough: ArrayFloat, blame_inputs=True):
        """
        CounterDense.backward over the patch rows: votes are drawn with probability |pressure|,
        and the pressure passed down is linear in it, so pressure is scaled down by vote_rate
        for the votes and back up for what goes down. blame_inputs=False: votes only (a first layer).
        """
        inputs, margins = state
        d = self.dense
        _, _, _, window, accept_distance, stabilize_margin, homeostasis, _, surrogate, normalize_reach = d._params()
        rate = np.float32(self.vote_rate)
        pressure = row_pressure(too_much, not_enough, rate, self.positions)
        conv_votes(inputs, self.index, margins, pressure, accept_distance, stabilize_margin,
                   d.counter_delta, d.threshold_delta, d.touched, d.rng_state)
        q = None
        if blame_inputs:
            q = conv_blame(margins, pressure, d.weights, window, surrogate, normalize_reach != 0, rate,
                           self.index, self.input_features)
        if homeostasis > 0:
            kernels.homeostasis(margins, d.thresholds, d.threshold_counters, homeostasis,
                                d.homeo_run, d.homeo_last, d.homeo_mean)
        return q

    def update(self):
        self.dense.update()

    def start_tally(self):
        self.dense.start_tally()

    def consolidate(self):
        self.dense.consolidate()


class OrPool(Layer):
    """
    Binary max pooling: a unit is on iff any unit of its size × size block is on (blocks don't overlap).
    Pressure: "not enough" goes to every unit of the block (any of them would do),
    "too much" only to those that are on.
    """

    def __init__(self, channels, height, width, size=2):
        self.channels, self.size = channels, size
        self.out_height, self.out_width = height // size, width // size
        self.input_features = channels * height * width
        self.output_features = channels * self.out_height * self.out_width
        c, i, j, di, dj = np.meshgrid(np.arange(channels), np.arange(self.out_height), np.arange(self.out_width),
                                      np.arange(size), np.arange(size), indexing="ij")
        index = c * height * width + (i * size + di) * width + j * size + dj
        self.index = np.ascontiguousarray(index.reshape(self.output_features, size * size))

    def margins(self, inputs: ArrayBool) -> np.ndarray:
        return self.forward(inputs)[0].astype(np.int32) - 1  # >= 0 iff on

    def forward(self, inputs: ArrayBool):
        inputs = np.ascontiguousarray(inputs, dtype=np.bool_)
        return pool_forward(inputs, self.index), inputs

    def backward(self, inputs, too_much: ArrayFloat, not_enough: ArrayFloat):
        return pool_backward(inputs, self.index, too_much, not_enough)

    def update(self):
        pass

    def start_tally(self):
        pass

    def consolidate(self):
        pass


def train_epoch(layers, inputs: ArrayBool, targets: ArrayBool, order: np.ndarray, batch_size: int):
    """
    One epoch of a net (conv and pool layers, then CounterDense ones), batch by batch: forward,
    backward from the last layer's output_pressure(targets), update. The same as Sequence's
    forward, backward and update, minus the first layer's input pressure, which nobody reads.
    """
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        x, states = inputs[batch], []
        for layer in layers:
            x, state = layer.forward(x)
            states.append(state)
        q = layers[-1].output_pressure(states[-1], targets[batch])
        for layer, state in zip(layers[:0:-1], states[:0:-1]):
            q = layer.backward(state, *q)
        if isinstance(layers[0], ConvCounter):
            layers[0].backward(states[0], *q, blame_inputs=False)
        else:
            layers[0].backward(states[0], *q)
        for layer in layers:
            layer.update()
