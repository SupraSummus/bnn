"""
CounterDense with a fixed number of synapse slots per neuron.

Every output has `n_slots` pointers into its inputs, each with a counter underneath:
the slot is an exciting synapse when its counter >= connect_level, inhibiting when
<= -connect_level, absent otherwise. Votes, thresholds, pressure and homeostasis follow
CounterDense restricted to the slots (with every input in a slot it is CounterDense,
`sparse_tests.py` checks bit for bit); work and memory are O(n_slots) per output, not O(inputs).

Rewiring (`rewire` > 0): a slot that holds no synapse can move. When an output votes,
with probability `rewire` it asks for a synapse on a random input that is on in that sample
(the one the vote would push toward existing: "not enough" -> an exciting synapse on an active input,
"too much" -> an inhibiting one). The update gives it the absent slot with the weakest counter,
at +-connect_level: it exists, and keeps existing only while votes agree with it.
Consolidation is CounterDense's, per slot; slots don't move while it tallies.
"""

from typing import Optional

import numpy as np
from numba import njit

from . import counter_kernels as kernels
from .abstract import ArrayBool, ArrayFloat, Layer
from .counter import NO_LIMIT, resolve_settings
from .counter_kernels import NONE, TRIANGLE, next_uniform

MAX_TRIES = 16  # random picks per rewire request before giving up on finding an active input


@njit(cache=True)
def margins(x, pointers, weights, thresholds):
    """Input sum minus threshold for every (sample, output); fires iff >= 0."""
    batch = x.shape[0]
    n_out, n_slots = pointers.shape
    out = np.empty((batch, n_out), np.int32)
    for b in range(batch):
        for o in range(n_out):
            s = np.int32(0)
            for j in range(n_slots):
                if x[b, pointers[o, j]]:
                    s += weights[o, j]
            out[b, o] = s - thresholds[o]
    return out


@njit(cache=True)
def backward(
    x, margins, pressure, pointers, weights,
    window, accept_distance, stabilize_margin, surrogate, normalize_reach, blame_inputs, rewire,
    counter_delta, threshold_delta, touched, grow_input, grow_sign, rng_state,
):
    """counter_kernels.backward over slots, plus rewire requests (`grow_input`, `grow_sign`; latest wins)."""
    batch, n_in = x.shape
    n_out, n_slots = pointers.shape
    q_in = np.zeros((batch if blame_inputs else 0, n_in), np.float32)
    reach = np.zeros(n_in, np.int64)
    for b in range(batch):
        if blame_inputs:
            reach[:] = 0
        for o in range(n_out):
            q = pressure[b, o]
            if q == 0:
                continue
            m = np.int64(margins[b, o])
            out = m >= 0
            distance = m if out else -m - 1
            flip = (q > 0) != out
            may_vote = distance <= accept_distance if flip else distance < stabilize_margin
            if may_vote and next_uniform(rng_state) < abs(np.float64(q)):
                vote = np.int32(1) if q > 0 else np.int32(-1)
                for j in range(n_slots):
                    counter_delta[o, j] += vote * (2 * np.int32(x[b, pointers[o, j]]) - 1)
                threshold_delta[o] -= vote
                touched[o] = True
                if rewire > 0 and next_uniform(rng_state) < rewire:
                    for _ in range(MAX_TRIES):
                        i = np.int64(next_uniform(rng_state) * n_in)
                        if x[b, i]:
                            grow_input[o] = i
                            grow_sign[o] = vote
                            break
            if blame_inputs and (surrogate == NONE or distance < window):
                c = np.float32(np.float64(q) / (distance + 1)) if surrogate == TRIANGLE else q
                for j in range(n_slots):
                    w = weights[o, j]
                    i = pointers[o, j]
                    q_in[b, i] += c * np.float32(w)
                    reach[i] += w != 0
        if blame_inputs and normalize_reach:
            for i in range(n_in):
                if reach[i] > 1:
                    q_in[b, i] /= np.float32(np.sqrt(np.float64(reach[i])))
    return q_in


@njit(cache=True)
def update(
    counters, counter_delta, touched, counter_max, connect_level, weights, pointers,
    thresholds, threshold_counters, threshold_delta, threshold_level, grow_input, grow_sign,
):
    """Apply and clear the votes; then move one absent slot per granted rewire request. Returns slots moved."""
    n_out, n_slots = counters.shape
    moved = 0
    for o in range(n_out):
        if touched[o]:
            touched[o] = False
            for j in range(n_slots):
                c = min(max(np.int32(counters[o, j]) + counter_delta[o, j], -counter_max), counter_max)
                counter_delta[o, j] = 0
                counters[o, j] = c
                weights[o, j] = np.int8(c >= connect_level) - np.int8(c <= -connect_level)
        i = grow_input[o]
        if i >= 0:
            grow_input[o] = -1
            weakest = -1
            for j in range(n_slots):
                if pointers[o, j] == i:
                    weakest = -2  # already wired there
                    break
                if weights[o, j] == 0 and (weakest == -1 or abs(counters[o, j]) < abs(counters[o, weakest])):
                    weakest = j
            if weakest >= 0:
                pointers[o, weakest] = i
                counters[o, weakest] = grow_sign[o] * connect_level
                weights[o, weakest] = grow_sign[o]
                moved += 1

        t = threshold_counters[o] + threshold_delta[o]
        threshold_delta[o] = 0
        step = np.int32(t / threshold_level)  # truncates towards zero
        thresholds[o] += step
        threshold_counters[o] = t - step * threshold_level
    return moved


class SparseCounter(Layer):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        n_slots: int,
        connect_level: Optional[int] = None,
        counter_max: Optional[int] = None,
        init_range: Optional[int] = None,
        threshold_level: int = 4,
        window: Optional[int] = None,
        accept_distance: Optional[int] = None,
        stabilize_margin: int = 0,
        hinge: Optional[int] = None,
        homeostasis: int = 0,
        surrogate: str = "triangle",
        normalize_reach: bool = True,
        rewire: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Settings as in CounterDense, with the fan-in of the defaults being `n_slots`
        (what a neuron actually sums). Pointers start at distinct random inputs.

        Args:
            n_slots: synapse slots (pointers) per output, at most input_features.
            rewire: chance that a vote also asks to move an absent slot onto an active input.
        """
        assert 0 < n_slots <= input_features
        assert surrogate in kernels.SURROGATES, surrogate
        connect_level, counter_max, init_range, window, accept_distance, stabilize_margin, hinge = resolve_settings(
            n_slots, connect_level, counter_max, init_range, window, accept_distance, stabilize_margin, hinge,
        )
        self.input_features = input_features
        self.output_features = output_features
        self.n_slots = n_slots
        self.connect_level = connect_level
        self.counter_max = counter_max
        self.threshold_level = threshold_level
        self.window = window
        self.accept_distance = accept_distance
        self.stabilize_margin = stabilize_margin
        self.hinge = hinge
        self.homeostasis = homeostasis
        self.surrogate = surrogate
        self.normalize_reach = normalize_reach
        self.rewire = rewire
        self.rng = rng if rng is not None else np.random.default_rng()

        self.pointers = np.argsort(self.rng.random((output_features, input_features)), axis=1)[:, :n_slots]
        self.pointers = np.ascontiguousarray(self.pointers, dtype=np.int64)
        self.counters = np.clip(
            self.rng.integers(-init_range, init_range + 1, size=(output_features, n_slots)),
            -counter_max, counter_max,
        ).astype(np.int16)
        self.rng_state = self.rng.integers(0, 2**64, size=4, dtype=np.uint64) | np.uint64(1)
        self.thresholds = np.zeros(output_features, dtype=np.int32)
        self.threshold_counters = np.zeros(output_features, dtype=np.int32)
        self.homeo_run = np.zeros(output_features, dtype=np.int64)
        self.homeo_last = np.zeros(output_features, dtype=np.bool_)
        self.homeo_mean = np.zeros(output_features, dtype=np.float64)
        self.counter_delta = np.zeros((output_features, n_slots), dtype=np.int32)
        self.threshold_delta = np.zeros(output_features, dtype=np.int32)
        self.touched = np.zeros(output_features, dtype=np.bool_)
        self.grow_input = np.full(output_features, -1, dtype=np.int64)
        self.grow_sign = np.zeros(output_features, dtype=np.int8)
        self.moved = 0  # slots rewired so far
        self.tally = None
        self.sync_weights()

    def sync_weights(self):
        """Refresh the visible synapses after counters were set from outside the update."""
        self.weights = (
            (self.counters >= self.connect_level).astype(np.int8)
            - (self.counters <= -self.connect_level).astype(np.int8)
        )

    def margins(self, inputs: ArrayBool) -> np.ndarray:
        return margins(np.ascontiguousarray(inputs, dtype=np.bool_), self.pointers, self.weights, self.thresholds)

    def forward(self, inputs: ArrayBool):
        inputs = np.ascontiguousarray(inputs, dtype=np.bool_)
        assert inputs.ndim == 2 and inputs.shape[1] == self.input_features
        m = margins(inputs, self.pointers, self.weights, self.thresholds)
        return m >= 0, (inputs, m)

    def output_pressure(self, state, targets: ArrayBool):
        _, m = state
        q = kernels.output_pressure(m, np.ascontiguousarray(targets, dtype=np.bool_), self.hinge)
        return np.maximum(-q, 0), np.maximum(q, 0)

    def backward(self, state, too_much: ArrayFloat, not_enough: ArrayFloat):
        inputs, m = state
        pressure = np.asarray(not_enough, dtype=np.float32) - np.asarray(too_much, dtype=np.float32)
        accept_distance = NO_LIMIT if self.accept_distance is None else self.accept_distance
        rewire = 0.0 if self.tally is not None else self.rewire
        q_in = backward(
            inputs, m, pressure, self.pointers, self.weights,
            self.window, accept_distance, self.stabilize_margin,
            kernels.SURROGATES.index(self.surrogate), self.normalize_reach, True, rewire,
            self.counter_delta, self.threshold_delta, self.touched, self.grow_input, self.grow_sign, self.rng_state,
        )
        if self.homeostasis > 0:
            kernels.homeostasis(m, self.thresholds, self.threshold_counters, self.homeostasis,
                                self.homeo_run, self.homeo_last, self.homeo_mean)
        return np.maximum(-q_in, 0), np.maximum(q_in, 0)

    def update(self):
        self.moved += update(
            self.counters, self.counter_delta, self.touched, self.counter_max, self.connect_level,
            self.weights, self.pointers,
            self.thresholds, self.threshold_counters, self.threshold_delta, self.threshold_level,
            self.grow_input, self.grow_sign,
        )
        if self.tally is not None:
            self.tally[0] += self.weights
            self.tally[1] += self.thresholds
            self.tally[2] += 1

    def start_tally(self):
        """Count the visible state of every slot and threshold over the following updates."""
        self.tally = [np.zeros(self.weights.shape, dtype=np.int32), np.zeros(self.output_features, dtype=np.int64), 0]

    def consolidate(self):
        """Keep each slot's average state over the tally (rounded to -1/0/1) and the average threshold."""
        weights, thresholds, n = self.tally
        self.tally = None
        if n == 0:
            return
        self.counters = (np.sign(weights) * (2 * np.abs(weights) >= n) * self.connect_level).astype(np.int16)
        self.thresholds = np.round(thresholds / n).astype(np.int32)
        self.sync_weights()
