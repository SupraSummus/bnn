from typing import List, Tuple

import numpy as np

from .abstract import ArrayBool, ArrayFloat, Layer

ArrayInt = np.ndarray


class Dense(Layer):
    def __init__(self, input_features: int, output_features: int):
        self.input_features = input_features
        self.output_features = output_features
        self.excitators: List[List[int]] = [[] for _ in range(output_features)]
        self.inhibitors: List[List[int]] = [[] for _ in range(output_features)]
        self.thresholds: ArrayInt = np.zeros(output_features, dtype=np.int32)

        self.excitator_changes: List[List[Tuple[int, float]]] = [
            [] for _ in range(output_features)
        ]
        self.inhibitor_changes: List[List[Tuple[int, float]]] = [
            [] for _ in range(output_features)
        ]
        self.threshold_changes: ArrayFloat = np.zeros(output_features, dtype=np.float32)

    def forward(self, inputs: ArrayBool) -> ArrayBool:
        input_shape = inputs.shape
        flattened_inputs = inputs.reshape(-1, self.input_features)

        outputs = np.zeros(
            (flattened_inputs.shape[0], self.output_features), dtype=bool
        )

        for i in range(self.output_features):
            excitation = np.sum(flattened_inputs[:, self.excitators[i]], axis=1)
            inhibition = np.sum(flattened_inputs[:, self.inhibitors[i]], axis=1)
            outputs[:, i] = (excitation - inhibition) >= self.thresholds[i]

        return outputs.reshape(input_shape[:-1] + (self.output_features,))

    def backward(
        self,
        inputs: ArrayBool,
        outputs: ArrayBool,
        too_much: ArrayFloat,
        not_enough: ArrayFloat,
    ) -> Tuple[ArrayFloat, ArrayFloat]:
        input_shape = inputs.shape
        flattened_inputs = inputs.reshape(-1, self.input_features)
        flattened_too_much = too_much.reshape(-1, self.output_features)
        flattened_not_enough = not_enough.reshape(-1, self.output_features)

        input_too_much = np.zeros(
            (flattened_inputs.shape[0], self.input_features), dtype=np.float32
        )
        input_not_enough = np.zeros(
            (flattened_inputs.shape[0], self.input_features), dtype=np.float32
        )

        for i in range(self.output_features):
            # Propagate "too much" feedback
            for j in self.excitators[i]:
                input_too_much[:, j] += flattened_too_much[:, i]
            for j in self.inhibitors[i]:
                input_not_enough[:, j] += flattened_too_much[:, i]

            # Propagate "not enough" feedback
            for j in self.excitators[i]:
                input_not_enough[:, j] += flattened_not_enough[:, i]
            for j in self.inhibitors[i]:
                input_too_much[:, j] += flattened_not_enough[:, i]

            # Accumulate changes for update step
            self.threshold_changes[i] += np.sum(flattened_too_much[:, i]) - np.sum(
                flattened_not_enough[:, i]
            )

            active_inputs = flattened_inputs[:, self.excitators[i]].any(axis=0)
            potential_excitators = set(range(self.input_features)) - set(
                self.excitators[i]
            )
            potential_inhibitors = set(range(self.input_features)) - set(
                self.inhibitors[i]
            )

            self.excitator_changes[i].extend(
                [
                    (j, np.sum(flattened_not_enough[:, i]))
                    for j in potential_excitators
                    if active_inputs[j]
                ]
            )
            self.inhibitor_changes[i].extend(
                [
                    (j, np.sum(flattened_too_much[:, i]))
                    for j in potential_inhibitors
                    if active_inputs[j]
                ]
            )

        return (
            input_too_much.reshape(input_shape),
            input_not_enough.reshape(input_shape),
        )

    def update(self):
        # Update thresholds
        self.thresholds += np.sign(self.threshold_changes).astype(np.int32)
        self.threshold_changes.fill(0)

        # Update synapses
        for i in range(self.output_features):
            # Add/remove excitatory synapses
            self.excitators[i].extend(
                [j for j, change in self.excitator_changes[i] if change > 0]
            )
            self.excitators[i] = [
                j for j in self.excitators[i] if np.random.rand() > 0.1
            ]  # Random pruning

            # Add/remove inhibitory synapses
            self.inhibitors[i].extend(
                [j for j, change in self.inhibitor_changes[i] if change > 0]
            )
            self.inhibitors[i] = [
                j for j in self.inhibitors[i] if np.random.rand() > 0.1
            ]  # Random pruning

            self.excitator_changes[i].clear()
            self.inhibitor_changes[i].clear()
