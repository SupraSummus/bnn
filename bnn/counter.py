"""
Binary threshold layer whose synapses are backed by hidden integer counters.

Every (neuron, input) pair has a bounded counter; the visible synapse is
exciting when counter >= connect_level, inhibiting when <= -connect_level,
absent otherwise. Votes move counters by one, so a synapse changes only after
consistent evidence. Thresholds get the same treatment.

Learning is driven by pressure: a signed float per output (> 0: wants more
excitation). The output layer gets +1/-1 toward its targets (zero where it is
right by a margin); every layer passes pressure down to its inputs through the
outputs near their threshold, and turns it into binary votes (stochastic
rounding) before it touches a counter. Learned state is integer.

The heavy lifting is in compiled kernels (`counter_kernels`, numba, bit-packed
synapses); `counter_reference` has the same rule in plain numpy.
"""

import copy
from typing import Optional

import numpy as np
from numba.typed import List

from . import counter_kernels as kernels
from .abstract import ArrayBool, ArrayFloat, Layer

NO_LIMIT = np.iinfo(np.int32).max


class CounterDense(Layer):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        connect_level: Optional[int] = None,
        counter_max: Optional[int] = None,
        init_range: Optional[int] = None,
        threshold_level: int = 4,
        window: Optional[int] = None,
        accept_distance: Optional[int] = None,
        stabilize_margin: int = 0,
        hinge: Optional[int] = None,
        homeostasis: int = 0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Margins (input sum minus threshold) spread like sqrt(input_features),
        so the defaults that are distances scale with it.

        Args:
            connect_level: |counter| needed for a synapse to exist.
                Default 0.7 * sqrt(input_features), at least 2.
            counter_max: counters are clipped to [-counter_max, counter_max]. Default 8 * connect_level.
            init_range: counters start uniform in [-init_range, init_range].
                Default 1.5 * connect_level connects about half the synapses,
                which gives pressure paths to flow down.
            threshold_level: net votes needed to move a threshold by 1.
            window: only outputs closer than this to flipping pass pressure to their inputs,
                each 1 / (distance + 1) of it. Default sqrt(input_features), about one margin spread.
            accept_distance: ignore pressure to flip an output further than this
                from flipping (minimal disturbance). 0 for hidden layers, None for
                the output layer, whose errors are all real.
            stabilize_margin: an output whose state agrees with its pressure but is closer
                than this to flipping votes to stay. Output layer only.
            hinge: output_pressure sends nothing for outputs right by at least this;
                they no longer push the layers below. Output layer only.
                Default sqrt(input_features) / 3, at least stabilize_margin.
            homeostasis: a unit whose output was the same on this many samples in a row
                moves its threshold to the running mean of its input sum (revives units that
                no pressure can reach). Hidden layers; 0 = off.
        """
        fan_in = np.sqrt(input_features)
        if connect_level is None:
            connect_level = max(2, int(round(0.7 * fan_in)))
        if counter_max is None:
            counter_max = 8 * connect_level
        if window is None:
            window = max(1, int(round(fan_in)))
        if hinge is None:
            hinge = max(stabilize_margin, int(round(fan_in / 3)))
        assert 0 < connect_level <= counter_max
        self.input_features = input_features
        self.output_features = output_features
        self.connect_level = connect_level
        self.counter_max = counter_max
        self.threshold_level = threshold_level
        self.window = window
        self.accept_distance = accept_distance
        self.stabilize_margin = stabilize_margin
        self.hinge = hinge
        self.homeostasis = homeostasis
        self.rng = rng if rng is not None else np.random.default_rng()

        if init_range is None:
            init_range = connect_level + connect_level // 2
        self.counters = np.clip(
            self.rng.integers(-init_range, init_range + 1, size=(output_features, input_features)),
            -counter_max, counter_max,
        ).astype(np.int16)
        # the kernels' own generator, seeded from rng: they draw one number per nonzero pressure
        self.rng_state = self.rng.integers(0, 2**64, size=4, dtype=np.uint64) | np.uint64(1)
        self.thresholds = np.zeros(output_features, dtype=np.int32)
        self.threshold_counters = np.zeros(output_features, dtype=np.int32)
        # homeostasis: run of identical outputs, last output, running mean of the input sum
        self.homeo_run = np.zeros(output_features, dtype=np.int64)
        self.homeo_last = np.zeros(output_features, dtype=np.bool_)
        self.homeo_mean = np.zeros(output_features, dtype=np.float64)
        self.tally = None
        self._sync_weights()
        self._reset_state()

    def margins(self, inputs: ArrayBool) -> np.ndarray:
        return kernels.margins(self._pack(inputs), self.exc, self.inh, self.thresholds)

    def forward(self, inputs: ArrayBool):
        assert inputs.ndim == 2, f"Expected 2D input, got {inputs.ndim}D"
        assert inputs.shape[1] == self.input_features, f"Expected {self.input_features} input features, got {inputs.shape[1]}"
        bits = self._pack(inputs)
        margins = kernels.margins(bits, self.exc, self.inh, self.thresholds)
        return margins >= 0, (inputs, bits, margins)

    def output_pressure(self, state, targets: ArrayBool):
        """(too_much, not_enough) for this layer as the output layer, from the forward state."""
        _, _, margins = state
        q = kernels.output_pressure(margins, np.ascontiguousarray(targets, dtype=np.bool_), self.hinge)
        return np.maximum(-q, 0), np.maximum(q, 0)

    def backward(self, state, too_much: ArrayFloat, not_enough: ArrayFloat):
        """
        Net out "too much" and "not enough" into pressure, turn it into binary votes
        on counters and thresholds, and pass pressure down to the inputs.
        `ReferenceCounterDense.backward` spells the rule out in array form.
        """
        inputs, _, margins = state
        assert too_much.shape == margins.shape
        assert not_enough.shape == margins.shape
        pressure = np.asarray(not_enough, dtype=np.float32) - np.asarray(too_much, dtype=np.float32)
        _, _, _, window, accept_distance, stabilize_margin, homeostasis, _ = self._params()
        q_in = kernels.backward(
            np.ascontiguousarray(inputs, dtype=np.bool_), margins, pressure, self.weights,
            window, accept_distance, stabilize_margin, True,
            self.counter_delta, self.threshold_delta, self.touched, self.rng_state,
        )
        if homeostasis > 0:
            kernels.homeostasis(margins, self.thresholds, self.threshold_counters, homeostasis,
                                self.homeo_run, self.homeo_last, self.homeo_mean)
        return np.maximum(-q_in, 0), np.maximum(q_in, 0)

    def update(self):
        tally_weights, tally_since, tally_count = self._tally_rows()
        kernels.update(
            self.counters, self.counter_delta, self.touched, self.counter_max, self.connect_level,
            self.weights, self.exc, self.inh,
            self.thresholds, self.threshold_counters, self.threshold_delta, self.threshold_level,
            tally_weights, tally_since, tally_count, self.tally is not None,
        )
        self._add_to_tally()

    def start_tally(self):
        """
        Count the visible state of every synapse and threshold over the following updates.
        Synapse rows are added lazily: a row's state since the update in `since` is
        added when the row changes or at consolidation, so a tallied update costs
        what an untallied one does.
        """
        n_out = self.output_features
        self.tally = [np.zeros(self.weights.shape, dtype=np.int32), np.zeros_like(self.thresholds), 0,
                      np.zeros(n_out, dtype=np.int64)]

    def consolidate(self):
        """
        Keep each synapse's average state over the tally (rounded to -1/0/1)
        and the average threshold. The last state of a perceptron-like learner
        is a noisy sample; the average is what it converged around.
        Counters are reset to just past connect_level.
        """
        self._flush_tally()
        weights, thresholds, n, _ = self.tally
        if n == 0:  # nothing tallied (a ratchet rolled back to where the tally started): keep the state
            self.tally = None
            return
        self.counters = (np.sign(weights) * (2 * np.abs(weights) >= n) * self.connect_level).astype(np.int16)
        self.thresholds = np.round(thresholds / n).astype(np.int32)
        self.tally = None
        self._sync_weights()

    def _params(self):
        """Integer settings in the order kernels.train_epoch unpacks them."""
        accept_distance = NO_LIMIT if self.accept_distance is None else self.accept_distance
        return (
            self.connect_level, self.counter_max, self.threshold_level,
            self.window, accept_distance, self.stabilize_margin, self.homeostasis, self.hinge,
        )

    def _add_to_tally(self):
        """After an update: thresholds and the count; synapse rows are added by the update kernel."""
        if self.tally is not None:
            self.tally[1] += self.thresholds
            self.tally[2] += 1

    def _tally_rows(self):
        """(weight sums, since, updates so far) for the update kernel, empty when not tallying."""
        if self.tally is None:
            return np.zeros((0, 0), dtype=np.int32), np.zeros(0, dtype=np.int64), 0
        return self.tally[0], self.tally[3], self.tally[2]

    def _pack(self, inputs):
        return kernels.pack_bits(np.ascontiguousarray(inputs, dtype=np.bool_))

    def _flush_tally(self):
        """Add every row's state since its last addition."""
        if self.tally is not None:
            weights, _, n, since = self.tally
            weights += self.weights * (n - since)[:, np.newaxis].astype(np.int32)
            since[:] = n

    def _sync_weights(self):
        """Refresh the visible synapses after counters were set from outside the update."""
        if getattr(self, "tally", None) is not None:
            self._flush_tally()
        self.weights = (
            (self.counters >= self.connect_level).astype(np.int8)
            - (self.counters <= -self.connect_level).astype(np.int8)
        )
        self.exc = kernels.pack_bits(self.weights > 0)
        self.inh = kernels.pack_bits(self.weights < 0)

    def _reset_state(self):
        # the update kernel clears these as it applies them
        self.counter_delta = np.zeros((self.output_features, self.input_features), dtype=np.int32)
        self.threshold_delta = np.zeros(self.output_features, dtype=np.int32)
        self.touched = np.zeros(self.output_features, dtype=np.bool_)  # rows with counter votes


def train_epoch(layers, inputs: ArrayBool, targets: ArrayBool, order: np.ndarray, batch_size: int):
    """
    One epoch of a stack of CounterDense layers as a single compiled loop.
    Same result as, for each batch of `order`: forward through the layers,
    backward from the last layer's output_pressure(targets), update
    (minus the first layer's input pressure, which nobody reads).
    Tallies layers that have started one.
    """
    tallying = layers[0].tally is not None
    assert all((layer.tally is not None) == tallying for layer in layers)

    def typed(values):
        return List([np.asarray(value) for value in values])

    counts = {layer.tally[2] for layer in layers} if tallying else {0}
    assert len(counts) == 1, "layers started their tallies at different updates"
    no_tally = [np.zeros((0, 0), dtype=np.int32), np.zeros(0, dtype=np.int32), 0, np.zeros(0, dtype=np.int64)]
    tallies = [layer.tally if tallying else no_tally for layer in layers]
    updates = kernels.train_epoch(
        np.ascontiguousarray(inputs, dtype=np.bool_), np.ascontiguousarray(targets, dtype=np.bool_),
        np.asarray(order, dtype=np.int64), batch_size,
        List([tuple(int(v) for v in layer._params()) for layer in layers]),
        *(typed(getattr(layer, name) for layer in layers) for name in kernels.LAYER_STATE),
        typed(t[0] for t in tallies), typed(t[1] for t in tallies), typed(t[3] for t in tallies),
        counts.pop(), tallying,
    )
    if tallying:
        for layer in layers:
            layer.tally[2] += updates


def readout_inputs(layers, inputs: ArrayBool) -> ArrayBool:
    """What the last layer reads: `inputs` through all the layers below it."""
    for layer in layers[:-1]:
        inputs = layer.margins(inputs) >= 0
    return inputs


def refit_readout(layers, inputs: ArrayBool, targets: ArrayBool, order: np.ndarray, batch_size: int):
    """
    After consolidation: one more epoch of the last layer alone on the code the consolidated layers
    below produce, tallied and consolidated. Each layer was averaged on its own, so the averaged
    hidden code is one no layer above has seen; the readout is fitted to the code it will get.
    """
    readout = layers[-1]
    readout.start_tally()
    train_epoch([readout], readout_inputs(layers, inputs), targets, order, batch_size)
    readout.consolidate()


class Ratchet:
    """
    Selection on top of learning. Deep nets learn by search, and the plasticity that finds a solution
    also walks away from it (parity-16 with 3×64 hidden: 0.9 and back to 0.5 within a few hundred updates).
    `check` measures training accuracy on a fixed sample, remembers the best state,
    and puts the layers back to it when accuracy fell more than `tolerance` below:
    wider than the noise of a healthy net, so only collapses roll back.
    """

    def __init__(self, inputs: ArrayBool, labels, tolerance=0.25, check_every=128, size=2048):
        # its own generator: the sample doesn't shift the training's random stream
        pick = np.random.default_rng(0).permutation(len(inputs))[:size]
        self.inputs, self.labels = inputs[pick], labels[pick]
        self.tolerance = tolerance
        self.check_every = check_every
        self.best, self.best_accuracy = None, -1.0
        # what a collapse rolls back to: the best state, or one with the running tally (see rebase)
        self.anchor, self.anchor_accuracy = None, -1.0
        self.rollbacks = 0

    def accuracy(self, layers):
        margins = layers[-1].margins(readout_inputs(layers, self.inputs))
        return np.mean(np.argmax(margins, axis=1) == self.labels)

    def check(self, layers):
        """Remember the state if it is the best yet; roll `layers` back (in place) on a collapse."""
        accuracy = self.accuracy(layers)
        if accuracy > self.anchor_accuracy:
            self.anchor, self.anchor_accuracy = copy.deepcopy(layers), accuracy
            if accuracy > self.best_accuracy:
                self.best, self.best_accuracy = self.anchor, accuracy  # never modified, so shared
        elif accuracy < self.anchor_accuracy - self.tolerance:
            layers[:] = copy.deepcopy(self.anchor)
            self.rollbacks += 1

    def rebase(self, layers):
        """Roll back to the current state from now on: a tally started, and older states don't have it."""
        self.anchor, self.anchor_accuracy = copy.deepcopy(layers), self.accuracy(layers)
        if self.anchor_accuracy > self.best_accuracy:
            self.best, self.best_accuracy = self.anchor, self.anchor_accuracy

    def keep_best(self, layers):
        """
        At the end: put back the best state if it beats `layers` by more than the tolerance.
        A consolidated net fits the training set a little worse than the best raw state
        and generalizes better, so only a consolidated collapse should lose.
        """
        if self.best is not None and self.best_accuracy > self.accuracy(layers) + self.tolerance:
            layers[:] = copy.deepcopy(self.best)
