"""
Convolution as weight sharing: a CounterDense layer applied to every patch of an image.

Patches become rows, so one filter's counters collect the votes of every position,
and the pressure on a pixel is the sum over the patches that contain it.
A filter sees positions × batch rows per update; to keep its counters' hysteresis,
a row votes with probability |pressure| × `vote_rate` (default 1 / positions: a dense neuron's rate per sample).
Images are flat, channel first: index c * height * width + row * width + column
(what `benchmarks.datasets.fashion_bits` gives, with 2 channels); so are the outputs, one channel per filter.
"""

import numpy as np
from numba import njit

from .abstract import ArrayBool, ArrayFloat, Layer
from .counter import CounterDense


@njit(cache=True)
def scatter(q_patches, index, n_in):
    """Sum patch pressure (batch * positions, patch size) back onto the image (batch, n_in)."""
    positions = index.shape[0]
    batch = q_patches.shape[0] // positions
    out = np.zeros((batch, n_in), np.float32)
    for b in range(batch):
        for p in range(positions):
            row = q_patches[b * positions + p]
            for j in range(index.shape[1]):
                out[b, index[p, j]] += row[j]
    return out


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
        return inputs[:, self.index].reshape(-1, self.index.shape[1])

    def _to_image(self, rows):
        """(batch * positions, filters) -> (batch, filters * positions), channel first."""
        return np.ascontiguousarray(rows.reshape(-1, self.positions, self.filters).transpose(0, 2, 1)).reshape(-1, self.output_features)

    def _to_rows(self, image):
        return np.ascontiguousarray(image.reshape(-1, self.filters, self.positions).transpose(0, 2, 1)).reshape(-1, self.filters)

    def margins(self, inputs: ArrayBool) -> np.ndarray:
        return self._to_image(self.dense.margins(self.patches(inputs)))

    def forward(self, inputs: ArrayBool):
        rows, state = self.dense.forward(self.patches(inputs))
        return self._to_image(rows), state

    def output_pressure(self, state, targets: ArrayBool):
        too_much, not_enough = self.dense.output_pressure(state, self._to_rows(targets))
        return self._to_image(too_much), self._to_image(not_enough)

    def backward(self, state, too_much: ArrayFloat, not_enough: ArrayFloat):
        # votes are drawn with probability |pressure|, and the pressure passed down is linear in it:
        # scale down for the votes, back up for what goes down
        rate = np.float32(self.vote_rate)
        too_much, not_enough = self.dense.backward(state, self._to_rows(too_much) * rate, self._to_rows(not_enough) * rate)
        return (scatter(np.ascontiguousarray(too_much / rate), self.index, self.input_features),
                scatter(np.ascontiguousarray(not_enough / rate), self.index, self.input_features))

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
        block = inputs[:, self.index]
        return block.any(axis=2), block

    def backward(self, block, too_much: ArrayFloat, not_enough: ArrayFloat):
        batch = block.shape[0]
        out_tm = np.zeros((batch, self.input_features), np.float32)
        out_ne = np.zeros((batch, self.input_features), np.float32)
        flat = self.index.ravel()
        for b in range(batch):
            out_tm[b] = np.bincount(flat, (block[b] * too_much[b][:, None]).ravel(), self.input_features)
            out_ne[b] = np.bincount(flat, np.broadcast_to(not_enough[b][:, None], block[b].shape).ravel(), self.input_features)
        return out_tm, out_ne

    def update(self):
        pass

    def start_tally(self):
        pass

    def consolidate(self):
        pass
