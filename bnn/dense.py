import itertools
from typing import List, Tuple

import numpy as np

from .abstract import ArrayBool, ArrayFloat, Layer

ArrayInt = np.ndarray


class Dense(Layer):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        max_synapses=None,
        threshold_change_chance=0.1,
        synapse_change_chance=0.1,
    ):
        self.input_features = input_features
        self.output_features = output_features

        self.max_synapses = max_synapses
        if self.max_synapses is None:
            self.max_synapses = input_features // 10
        self.threshold_change_chance = threshold_change_chance
        self.synapse_change_chance = synapse_change_chance

        self.excitators: List[List[int]] = [[] for _ in range(output_features)]
        self.inhibitors: List[List[int]] = [[] for _ in range(output_features)]
        self.thresholds: ArrayInt = np.zeros(output_features, dtype=np.int32)

        # excitator_potential[i, j] stores how much an exciting synapse from input j to output i is beneficial.
        # Negative values indicate that the synapse should be removed.
        self.excitator_potential: ArrayFloat = None
        # Similarly for inhibitor_potential -- only for inhibitory synapses.
        self.inhibitor_potential: ArrayFloat = None
        # self.threshold_changes[i] stores how much and in which direction the threshold of output i should change
        self.threshold_changes: ArrayFloat = None
        self._reset_state()

    def forward(self, inputs: ArrayBool):
        assert inputs.ndim == 2, f"Expected 2D input, got {inputs.ndim}D"
        assert inputs.shape[1] == self.input_features, f"Expected {self.input_features} input features, got {inputs.shape[1]}"

        active_excitator_count = self._count_active(inputs, self.excitators)
        active_inhibitor_count = self._count_active(inputs, self.inhibitors)
        outputs = (active_excitator_count - active_inhibitor_count) >= self.thresholds

        return outputs, (inputs, active_excitator_count, active_inhibitor_count)

    def _count_active(self, inputs: ArrayBool, synapses: List[List[int]]) -> ArrayFloat:
        sources, owners = _flatten(synapses)
        counts = np.zeros((inputs.shape[0], self.output_features), dtype=np.float32)
        _scatter_add(counts, owners, inputs[:, sources])
        return counts

    def margins(self, inputs: ArrayBool) -> np.ndarray:
        _, (_, active_excitator_count, active_inhibitor_count) = self.forward(inputs)
        return (active_excitator_count - active_inhibitor_count - self.thresholds).astype(np.int32)

    def backward(
        self,
        state,
        too_much: ArrayFloat,
        not_enough: ArrayFloat,
    ) -> Tuple[ArrayFloat, ArrayFloat]:
        inputs, active_excitator_count, active_inhibitor_count = state

        assert too_much.shape == (inputs.shape[0], self.output_features)
        assert not_enough.shape == (inputs.shape[0], self.output_features)

        # Accumulate changes for update step
        self.threshold_changes += np.sum(too_much, axis=0)
        self.threshold_changes -= np.sum(not_enough, axis=0)
        # excitator potential is large for active inputs and for large not_enough feedback
        self.excitator_potential += np.sum(inputs[:, np.newaxis, :] * not_enough[:, :, np.newaxis], axis=0)
        # inhibitor potential is large for active inputs and for large too_much feedback
        self.inhibitor_potential += np.sum(inputs[:, np.newaxis, :] * too_much[:, :, np.newaxis], axis=0)

        # propagate down
        input_too_much = np.zeros(
            (inputs.shape[0], self.input_features), dtype=np.float32
        )
        input_not_enough = np.zeros(
            (inputs.shape[0], self.input_features), dtype=np.float32
        )

        # How many inputs can we blame for the "too much" feedback?
        # Well, we can blame 1) all excitatory synapses that are active and 2) all inhibitory synapses that are inactive
        # Point 2 is equal to the difference between the number of all inhibitory synapses and active inhibitory synapses
        inhibitor_count = np.array([len(s) for s in self.inhibitors], dtype=np.float32)
        inactive_inhibitor_count = inhibitor_count - active_inhibitor_count
        too_much_contributor_count = active_excitator_count + inactive_inhibitor_count
        too_much_contributions = np.divide(
            too_much, too_much_contributor_count,
            out=np.zeros_like(too_much_contributor_count),
            where=too_much_contributor_count != 0,
        )

        # And similarly for not enough feedback
        # 1) blame all excitatory synapses that are inactive and
        # 2) blame all inhibitory synapses that are active
        excitator_count = np.array([len(s) for s in self.excitators], dtype=np.float32)
        inactive_excitator_count = excitator_count - active_excitator_count
        not_enough_contributor_count = active_inhibitor_count + inactive_excitator_count
        not_enough_contributions = np.divide(
            not_enough, not_enough_contributor_count,
            out=np.zeros_like(not_enough_contributor_count),
            where=not_enough_contributor_count != 0,
        )

        # A synapse listed twice counts twice above but is blamed once (see TODO.md).
        exc, exc_owners = _flatten(self.excitators, unique=True)
        inh, inh_owners = _flatten(self.inhibitors, unique=True)
        excitations = inputs[:, exc]
        inhibitions = inputs[:, inh]
        _scatter_add(input_too_much, exc, excitations * too_much_contributions[:, exc_owners])
        _scatter_add(input_not_enough, inh, ~inhibitions * too_much_contributions[:, inh_owners])
        _scatter_add(input_not_enough, exc, ~excitations * not_enough_contributions[:, exc_owners])
        _scatter_add(input_too_much, inh, inhibitions * not_enough_contributions[:, inh_owners])

        return (
            input_too_much,
            input_not_enough,
        )

    def update(self):
        for i in range(self.output_features):
            change_threshold = 0.01

            if np.random.rand() < self.threshold_change_chance:
                if self.threshold_changes[i] > change_threshold:
                    plus_minus_one = 1
                elif self.threshold_changes[i] < -change_threshold:
                    plus_minus_one = -1
                else:
                    plus_minus_one = 0
                self.thresholds[i] += plus_minus_one

            changed = False
            if np.random.rand() < self.synapse_change_chance:
                potential = self.excitator_potential[i] - self.inhibitor_potential[i]
                max_j = np.argmax(np.abs(potential))
                if potential[max_j] > change_threshold:
                    self.excitators[i].append(max_j)
                    changed = True
                elif potential[max_j] < -change_threshold:
                    self.inhibitors[i].append(max_j)
                    changed = True

            if changed:
                self.excitators[i], self.inhibitors[i] = normalize_synapses(
                    self.excitators[i],
                    self.inhibitors[i],
                    self.max_synapses,
                )

        self._reset_state()

    def _reset_state(self):
        self.threshold_changes = np.zeros(self.output_features, dtype=np.float32)
        self.excitator_potential = np.zeros((self.output_features, self.input_features), dtype=np.float32)
        self.inhibitor_potential = np.zeros((self.output_features, self.input_features), dtype=np.float32)


def normalize_synapses(excitators, inhibitors, max_synapses):
    """
    * sort
    * remove self-cancelling synapses (both inhibitory and excitatory)
    * trim to size
    """
    new_excitators = []
    new_inhibitors = []

    excitators.sort()
    inhibitors.sort()

    ie = 0
    ii = 0
    while ie < len(excitators) and ii < len(inhibitors):
        if excitators[ie] == inhibitors[ii]:
            ie += 1
            ii += 1
        elif excitators[ie] < inhibitors[ii]:
            new_excitators.append(excitators[ie])
            ie += 1
        else:
            new_inhibitors.append(inhibitors[ii])
            ii += 1
    while ie < len(excitators):
        new_excitators.append(excitators[ie])
        ie += 1
    while ii < len(inhibitors):
        new_inhibitors.append(inhibitors[ii])
        ii += 1

    while len(new_excitators) > max_synapses:
        new_excitators.pop(np.random.randint(0, len(new_excitators)))
    while len(new_inhibitors) > max_synapses:
        new_inhibitors.pop(np.random.randint(0, len(new_inhibitors)))

    return new_excitators, new_inhibitors


def _flatten(synapses: List[List[int]], unique=False) -> Tuple[ArrayInt, ArrayInt]:
    """Per-output synapse lists as flat arrays: input ids and the output each one belongs to."""
    lengths = [len(s) for s in synapses]
    sources = np.fromiter(itertools.chain.from_iterable(synapses), dtype=np.intp, count=sum(lengths))
    owners = np.repeat(np.arange(len(synapses)), lengths)
    if unique and len(sources):
        width = sources.max() + 1
        owners, sources = np.divmod(np.unique(owners * width + sources), width)
    return sources, owners


def _scatter_add(out: ArrayFloat, columns: ArrayInt, values: np.ndarray):
    """out[:, columns] += values, summing repeated columns. Much faster than np.add.at on 2D."""
    bins = (np.arange(out.shape[0])[:, np.newaxis] * out.shape[1] + columns).ravel()
    out += np.bincount(bins, weights=values.ravel(), minlength=out.size).reshape(out.shape)
