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
        treshold_change_chance=0.1,
        synapse_change_chance=0.1,
    ):
        self.input_features = input_features
        self.output_features = output_features

        self.max_synapses = max_synapses
        if self.max_synapses is None:
            self.max_synapses = input_features // 10
        self.treshold_change_chance = treshold_change_chance
        self.synapse_change_chance = synapse_change_chance

        self.excitators: List[List[int]] = [[] for _ in range(output_features)]
        self.inhibitors: List[List[int]] = [[] for _ in range(output_features)]
        self.thresholds: ArrayInt = np.zeros(output_features, dtype=np.int32)

        # excitator_potential[i, j] stores how much an exciting synapse from input j to output i is beneficial.
        # Negative values indicate that the synapse should be removed.
        self.excitator_potential: ArrayFloat = None
        # Similarly for inhibitor_potential -- only for inhibitory synapses.
        self.inhibitor_potential: ArrayFloat = None
        # self.treshold_changes[i] stores how much and in which direction the threshold of output i should change
        self.threshold_changes: ArrayFloat = None
        self._reset_state()

    def forward(self, inputs: ArrayBool):
        assert inputs.ndim == 2, f"Expected 2D input, got {inputs.ndim}D"
        assert inputs.shape[1] == self.input_features, f"Expected {self.input_features} input features, got {inputs.shape[1]}"

        active_excitator_count = np.zeros((inputs.shape[0], self.output_features), dtype=np.float32)
        active_inhibitor_count = np.zeros((inputs.shape[0], self.output_features), dtype=np.float32)
        outputs = np.zeros((inputs.shape[0], self.output_features), dtype=np.bool)
        for i in range(self.output_features):
            active_excitator_count[:, i] = np.sum(inputs[:, self.excitators[i]], axis=1)
            active_inhibitor_count[:, i] = np.sum(inputs[:, self.inhibitors[i]], axis=1)
            outputs[:, i] = (active_excitator_count[:, i] - active_inhibitor_count[:, i]) >= self.thresholds[i]

        return outputs, (inputs, active_excitator_count, active_inhibitor_count)

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

        for i in range(self.output_features):
            excitations = inputs[:, self.excitators[i]]
            inhibitions = inputs[:, self.inhibitors[i]]

            # How many inputs can we blame for the "too much" feedback?
            # Well, we can blame 1) all excitatory synapses that are active and 2) all inhibitory synapses that are inactive
            # Point 2 is equal to the difference between the number of all inhibitory synapses and active inhibitory synapses
            inhibitor_count = len(self.inhibitors[i])
            inactive_inhibitor_count = inhibitor_count - active_inhibitor_count[:, i]
            too_much_contributor_count = active_excitator_count[:, i] + inactive_inhibitor_count
            too_much_contributions = np.divide(
                too_much[:, i], too_much_contributor_count,
                where=too_much_contributor_count != 0,
            )
            input_too_much[:, self.excitators[i]] += excitations * too_much_contributions[:, np.newaxis]
            input_not_enough[:, self.inhibitors[i]] += ~inhibitions * too_much_contributions[:, np.newaxis]

            # And similarly for not enough feedback
            # 1) blame all excitatory synapses that are inactive and
            # 2) blame all inhibitory synapses that are active
            excitator_count = len(self.excitators[i])
            inactive_excitator_count = excitator_count - active_excitator_count[:, i]
            not_enough_contributor_count = active_inhibitor_count[:, i] + inactive_excitator_count
            not_enough_contributions = np.divide(
                not_enough[:, i], not_enough_contributor_count,
                where=not_enough_contributor_count != 0,
            )
            input_not_enough[:, self.excitators[i]] += ~excitations * not_enough_contributions[:, np.newaxis]
            input_too_much[:, self.inhibitors[i]] += inhibitions * not_enough_contributions[:, np.newaxis]

        return (
            input_too_much,
            input_not_enough,
        )

    def update(self):
        for i in range(self.output_features):
            change_threshold = 0.01

            if np.random.rand() < self.treshold_change_chance:
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
