"""
Binary threshold layer whose synapses are backed by hidden integer counters.

Every (neuron, input) pair has a bounded counter; the visible synapse is
exciting when counter >= connect_level, inhibiting when <= -connect_level,
absent otherwise. Errors move counters by one, so a synapse changes only after
consistent evidence. Thresholds get the same treatment.

Learned state is integer. Floats only carry error strengths between layers and
are turned into binary events (stochastic rounding) before they touch a counter.
"""

from typing import Optional

import numpy as np

from .abstract import ArrayBool, ArrayFloat, Layer


class CounterDense(Layer):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        connect_level: int = 4,
        counter_max: int = 32,
        init_range: Optional[int] = None,
        threshold_level: int = 4,
        backprop_margin: int = 2,
        accept_distance: Optional[int] = None,
        stabilize_margin: int = 0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            connect_level: |counter| needed for a synapse to exist.
            counter_max: counters are clipped to [-counter_max, counter_max].
            init_range: counters start uniform in [-init_range, init_range].
                Default 1.5 * connect_level connects about half the synapses,
                which gives blame paths to flow down.
            threshold_level: net error votes needed to move a threshold by 1.
            backprop_margin: a neuron needing more input flips than this to become
                correct doesn't blame its inputs, it fixes itself.
            accept_distance: ignore requests to flip a neuron further than this
                from flipping (minimal disturbance). 0 for hidden layers, None for
                the output layer, whose errors are all real.
            stabilize_margin: reinforce error-free outputs closer than this to
                flipping. Output layer only, where no error means correct.
        """
        assert 0 < connect_level <= counter_max
        self.input_features = input_features
        self.output_features = output_features
        self.connect_level = connect_level
        self.counter_max = counter_max
        self.threshold_level = threshold_level
        self.backprop_margin = backprop_margin
        self.accept_distance = accept_distance
        self.stabilize_margin = stabilize_margin
        self.rng = rng if rng is not None else np.random.default_rng()

        if init_range is None:
            init_range = connect_level + connect_level // 2
        self.counters = self.rng.integers(
            -init_range, init_range + 1,
            size=(output_features, input_features),
        ).astype(np.int16)
        self.thresholds = np.zeros(output_features, dtype=np.int32)
        self.threshold_counters = np.zeros(output_features, dtype=np.int32)
        self.tally = None
        self._sync_weights()
        self._reset_state()

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
        p = self.rng.random(margins.shape)
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
        steps = np.fix(self.threshold_counters / self.threshold_level).astype(np.int32)
        self.thresholds += steps
        self.threshold_counters -= steps * self.threshold_level

        self._sync_weights()
        self._reset_state()
        if self.tally is not None:
            self.tally[0] += self.weights
            self.tally[1] += self.thresholds
            self.tally[2] += 1

    def start_tally(self):
        """Count the visible state of every synapse and threshold over the following updates."""
        self.tally = [np.zeros(self.weights.shape, dtype=np.int32), np.zeros_like(self.thresholds), 0]

    def consolidate(self):
        """
        Keep each synapse's average state over the tally (rounded to -1/0/1)
        and the average threshold. The last state of a perceptron-like learner
        is a noisy sample; the average is what it converged around.
        Counters are reset to just past connect_level.
        """
        weights, thresholds, n = self.tally
        self.counters = (np.sign(weights) * (2 * np.abs(weights) >= n) * self.connect_level).astype(np.int16)
        self.thresholds = np.round(thresholds / n).astype(np.int32)
        self.tally = None
        self._sync_weights()

    def _sync_weights(self):
        self.weights = (
            (self.counters >= self.connect_level).astype(np.int8)
            - (self.counters <= -self.connect_level).astype(np.int8)
        )

    def _reset_state(self):
        self.counter_delta = np.zeros((self.output_features, self.input_features), dtype=np.int32)
        self.threshold_delta = np.zeros(self.output_features, dtype=np.int32)
