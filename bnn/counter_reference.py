"""
Plain numpy version of CounterDense: the specification the compiled kernels in
`counter_kernels` follow. Slow on small batches (dozens of numpy calls each),
kept to test the kernels against and to try rule changes on.
"""

import numpy as np

from . import counter_kernels as kernels
from .abstract import ArrayBool, ArrayFloat
from .counter import CounterDense


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

    def backward(self, state, too_much: ArrayFloat, not_enough: ArrayFloat):
        inputs, margins = state
        outputs = margins >= 0
        assert too_much.shape == margins.shape
        assert not_enough.shape == margins.shape

        # Net out conflicting requests, keep those asking for an actual flip, sample binary events.
        wish = np.asarray(not_enough, dtype=np.float32) - np.asarray(too_much, dtype=np.float32)
        # draws only where there is a request (from the kernels' generator); p = 0 asks for nothing
        p = np.zeros(margins.shape)
        p[wish != 0] = kernels.uniforms(self.rng_state, np.count_nonzero(wish))
        want_on = ~outputs & (p < wish)
        want_off = outputs & (p < -wish)
        # 0 means a single input change would flip the output
        distance = np.where(outputs, margins, -margins - 1)
        if self.accept_distance is not None:
            want_on &= distance <= self.accept_distance
            want_off &= distance <= self.accept_distance

        # Local learning, perceptron on +-1 coded inputs: for a neuron that should turn on,
        # active inputs move towards excitation and inactive ones towards inhibition.
        x = inputs.astype(np.float32)
        on_votes = want_on
        off_votes = want_off
        if self.stabilize_margin > 0:
            fragile = (distance < self.stabilize_margin) & ~want_on & ~want_off
            on_votes = on_votes | (fragile & outputs)
            off_votes = off_votes | (fragile & ~outputs)
        votes = on_votes.astype(np.float32) - off_votes.astype(np.float32)
        self.counter_delta += (votes.T @ (2 * x - 1)).astype(np.int32)
        self.threshold_delta += off_votes.sum(axis=0) - on_votes.sum(axis=0)

        # Blame inputs. Turning on needs `-margin` more net excitation, turning off `margin + 1` less.
        # Inputs that can each provide one unit:
        #   turn on:  inactive exciters (should activate), active inhibitors (should deactivate)
        #   turn off: active exciters (should deactivate), inactive inhibitors (should activate)
        exc = (self.weights > 0).astype(np.float32)
        inh = (self.weights < 0).astype(np.float32)
        nx = 1.0 - x
        needed = np.where(want_on, -margins, margins + 1)
        helpers = np.where(want_on, nx @ exc.T + x @ inh.T, x @ exc.T + nx @ inh.T)
        fixable = (want_on | want_off) & (needed <= self.backprop_margin) & (needed <= helpers)
        # each helper gets the probability it is one of the `needed` flips
        share = np.divide(needed, helpers, out=np.zeros(needed.shape, dtype=np.float32), where=fixable)
        share_on = np.where(want_on, share, 0).astype(np.float32)
        share_off = np.where(want_off, share, 0).astype(np.float32)
        to_activate = share_on @ exc + share_off @ inh
        to_deactivate = share_on @ inh + share_off @ exc

        # Collateral damage: an input flip also flips every currently fine output sitting right at its
        # threshold (margin 0 drops on a -1 change, margin -1 rises on a +1 change).
        fine = ~(want_on | want_off)
        drop = (fine & (margins == 0)).astype(np.float32)
        rise = (fine & (margins == -1)).astype(np.float32)
        input_not_enough = nx * np.maximum(to_activate - drop @ inh - rise @ exc, 0)
        input_too_much = x * np.maximum(to_deactivate - drop @ exc - rise @ inh, 0)
        return input_too_much, input_not_enough

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
