"""
Binary threshold layer whose synapses are backed by hidden integer counters.

Every (neuron, input) pair has a bounded counter; the visible synapse is
exciting when counter >= connect_level, inhibiting when <= -connect_level,
absent otherwise. Errors move counters by one, so a synapse changes only after
consistent evidence. Thresholds get the same treatment.

Learned state is integer. Floats only carry error strengths between layers and
are turned into binary events (stochastic rounding) before they touch a counter.

The heavy lifting is in compiled kernels (`counter_kernels`, numba, bit-packed
synapses); `counter_reference` has the same rule in plain numpy.
"""

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
        self.counters = np.clip(
            self.rng.integers(-init_range, init_range + 1, size=(output_features, input_features)),
            -counter_max, counter_max,
        ).astype(np.int16)
        # the kernels' own generator, seeded from rng: they draw one number per nonzero error signal
        self.rng_state = self.rng.integers(0, 2**64, size=4, dtype=np.uint64) | np.uint64(1)
        self.thresholds = np.zeros(output_features, dtype=np.int32)
        self.threshold_counters = np.zeros(output_features, dtype=np.int32)
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

    def backward(self, state, too_much: ArrayFloat, not_enough: ArrayFloat):
        """
        Net out "too much" and "not enough", turn what asks for a flip into binary events,
        vote on counters and thresholds, and pass blame to the inputs that could make the flip.
        `ReferenceCounterDense.backward` spells the rule out in array form.
        """
        inputs, bits, margins = state
        assert too_much.shape == margins.shape
        assert not_enough.shape == margins.shape
        wish = np.asarray(not_enough, dtype=np.float32) - np.asarray(too_much, dtype=np.float32)
        _, _, _, backprop_margin, accept_distance, stabilize_margin = self._params()
        return kernels.backward(
            np.ascontiguousarray(inputs, dtype=np.bool_), bits, margins, wish,
            self.weights, self.exc, self.inh,
            backprop_margin, accept_distance, stabilize_margin, True,
            self.counter_delta, self.threshold_delta, self.touched, self.rng_state,
        )

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
        self.counters = (np.sign(weights) * (2 * np.abs(weights) >= n) * self.connect_level).astype(np.int16)
        self.thresholds = np.round(thresholds / n).astype(np.int32)
        self.tally = None
        self._sync_weights()

    def _params(self):
        """Scalar settings in the order kernels.train_epoch unpacks them."""
        accept_distance = NO_LIMIT if self.accept_distance is None else self.accept_distance
        return (
            self.connect_level, self.counter_max, self.threshold_level,
            self.backprop_margin, accept_distance, self.stabilize_margin,
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
    backward with too_much = out & ~targets and not_enough = ~out & targets, update
    (minus the first layer's input blame, which nobody reads).
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
