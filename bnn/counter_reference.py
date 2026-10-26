"""
Plain numpy version of CounterDense: the specification the compiled kernels in
`counter_kernels` follow. Slow on small batches (dozens of numpy calls each),
kept to test the kernels against and to try rule changes on.
"""

import numpy as np

from . import counter_kernels as kernels
from .abstract import ArrayBool, ArrayFloat
from .counter import NO_LIMIT, CounterDense


class ReferenceCounterDense(CounterDense):
    def margins(self, inputs: ArrayBool) -> np.ndarray:
        # float32 matmul is exact for integers below 2**24 and much faster than numpy's integer matmul
        sums = inputs.astype(np.float32) @ self.weights.T.astype(np.float32)
        return sums.astype(np.int32) - self.thresholds

    def forward(self, inputs: ArrayBool):
        assert inputs.ndim == 2, f"Expected 2D input, got {inputs.ndim}D"
        assert inputs.shape[1] == self.input_features, f"Expected {self.input_features} input features, got {inputs.shape[1]}"
        margins = self.margins(inputs)
        return margins >= 0, (inputs, margins)

    def output_pressure(self, state, targets: ArrayBool):
        _, margins = state
        distance = np.where(margins >= 0, margins, -margins - 1)
        right = (margins >= 0) == targets
        q = np.where(right & (distance >= self.hinge), 0, np.where(targets, 1, -1)).astype(np.float32)
        return np.maximum(-q, 0), np.maximum(q, 0)

    def backward(self, state, too_much: ArrayFloat, not_enough: ArrayFloat):
        inputs, margins = state
        outputs = margins >= 0
        assert too_much.shape == margins.shape
        assert not_enough.shape == margins.shape

        # Pressure > 0 asks for more excitation.
        q = np.asarray(not_enough, dtype=np.float32) - np.asarray(too_much, dtype=np.float32)
        # 0 means a single input change would flip the output
        distance = np.where(outputs, margins, -margins - 1)
        flip = (q > 0) != outputs
        accept = NO_LIMIT if self.accept_distance is None else self.accept_distance
        # Pressure against the state asks for a flip, taken only near threshold (minimal disturbance);
        # pressure with the state keeps fragile outputs where they are.
        may_vote = (q != 0) & np.where(flip, distance <= accept, distance < self.stabilize_margin)
        # a binary event with probability |q|, drawn (from the kernels' generator) only where it can count
        p = np.zeros(margins.shape)
        p[may_vote] = kernels.uniforms(self.rng_state, np.count_nonzero(may_vote))
        votes = np.where(may_vote & (p < np.abs(q).astype(np.float64)), np.sign(q), 0).astype(np.float32)

        # Local learning, perceptron on +-1 coded inputs: for a neuron that should turn on,
        # active inputs move towards excitation and inactive ones towards inhibition.
        x = inputs.astype(np.float32)
        self.counter_delta += (votes.T @ (2 * x - 1)).astype(np.int32)
        self.threshold_delta -= votes.sum(axis=0).astype(np.int32)

        # Pressure down: linear through the outputs within `window` of their threshold, each passing
        # 1 / (distance + 1) of its pressure along its synapses (a surrogate derivative of the step).
        # An input fed by `reach` such outputs gets their sum divided by sqrt(reach), so the signal
        # neither dies out nor blows up from layer to layer. Pressure on correct outputs is
        # collateral, in the same units: an input that would fix one output and break another gets 0.
        near = (q != 0) & (distance < self.window)
        c = np.where(near, (q.astype(np.float64) / (distance + 1)).astype(np.float32), np.float32(0))
        w = self.weights.astype(np.float32)
        q_in = np.zeros(inputs.shape, dtype=np.float32)
        for o in range(self.output_features):  # in the kernels' order, so sums agree to the bit
            q_in += c[:, o:o + 1] * w[o]
        reach = near.astype(np.int64) @ (self.weights != 0).astype(np.int64)
        scale = np.sqrt(np.maximum(reach, 1).astype(np.float64)).astype(np.float32)
        q_in = np.where(reach > 1, q_in / scale, q_in)

        if self.homeostasis > 0:
            self._homeostasis(margins)
        return np.maximum(-q_in, 0), np.maximum(q_in, 0)

    def _homeostasis(self, margins):
        base = self.thresholds.copy()
        for m in margins:
            out = m >= 0
            self.homeo_mean += (m.astype(np.float64) + base - self.homeo_mean) * kernels.HOMEOSTASIS_RATE
            self.homeo_run = np.where(out == self.homeo_last, self.homeo_run + 1, 1)
            self.homeo_last = out
            revive = self.homeo_run >= self.homeostasis
            self.thresholds[revive] = np.floor(self.homeo_mean[revive]).astype(np.int32)
            self.threshold_counters[revive] = 0
            self.homeo_run[revive] = 0

    def update(self):
        self.counters = np.clip(
            self.counters.astype(np.int32) + self.counter_delta,
            -self.counter_max, self.counter_max,
        ).astype(np.int16)

        self.threshold_counters += self.threshold_delta
        steps = np.trunc(self.threshold_counters / self.threshold_level).astype(np.int32)
        self.thresholds += steps
        self.threshold_counters -= steps * self.threshold_level

        self._sync_weights()
        self._reset_state()
        self._add_to_tally()

    def _add_to_tally(self):
        # the whole state after every update; CounterDense adds a row only when it is touched
        if self.tally is not None:
            self.tally[0] += self.weights
            self.tally[3] += 1
        super()._add_to_tally()
