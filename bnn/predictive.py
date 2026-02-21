"""
Predictive Coding Layer for Binary Neural Networks

This module implements a predictive coding extension to the BNN dense layer.
It adds a feedback (generative) pathway that enables top-down predictions,
turning the network into a hierarchical generative model where learning is
driven by local prediction errors.

Key concepts:
- Each layer has both feedforward (bottom-up) and feedback (top-down) connections
- The feedback pathway predicts expected input given the output
- Prediction errors (actual input vs predicted input) drive learning locally
- Error signals use the same binary dual-channel format (too_much / not_enough)

See docs/predictive_coding.md for the full theoretical analysis.
"""

from typing import List, Tuple

import numpy as np

from .abstract import ArrayBool, ArrayFloat, Layer
from .dense import Dense, normalize_synapses

ArrayInt = np.ndarray


class PredictiveCodingLayer(Layer):
    """
    A layer combining feedforward (bottom-up) and feedback (top-down) pathways.

    The feedforward pathway computes binary activations from sparse excitatory
    and inhibitory connections (exactly as in Dense).

    The feedback pathway generates top-down predictions of expected input
    activity, using sparse binary connections from output to input. Prediction
    errors (mismatches between actual and predicted input) provide additional
    local learning signals.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        max_synapses=None,
        treshold_change_chance=0.1,
        synapse_change_chance=0.1,
        prediction_weight=0.5,
    ):
        self.input_features = input_features
        self.output_features = output_features

        self.max_synapses = max_synapses
        if self.max_synapses is None:
            self.max_synapses = input_features // 10
        self.treshold_change_chance = treshold_change_chance
        self.synapse_change_chance = synapse_change_chance
        self.prediction_weight = prediction_weight

        # Feedforward pathway (bottom-up: input -> output)
        self.excitators: List[List[int]] = [[] for _ in range(output_features)]
        self.inhibitors: List[List[int]] = [[] for _ in range(output_features)]
        self.thresholds: ArrayInt = np.zeros(output_features, dtype=np.int32)

        # Feedback pathway (top-down: output -> input prediction)
        self.fb_excitators: List[List[int]] = [[] for _ in range(input_features)]
        self.fb_inhibitors: List[List[int]] = [[] for _ in range(input_features)]
        self.fb_thresholds: ArrayInt = np.zeros(input_features, dtype=np.int32)

        self._reset_state()

    def forward(self, inputs: ArrayBool):
        """
        Forward pass: compute output activations from inputs.

        Also computes top-down predictions of what the input *should* look like
        given the output, and the resulting prediction errors.
        """
        assert inputs.ndim == 2, f"Expected 2D input, got {inputs.ndim}D"
        assert inputs.shape[1] == self.input_features

        batch_size = inputs.shape[0]

        # --- Feedforward pass (same as Dense) ---
        active_excitator_count = np.zeros(
            (batch_size, self.output_features), dtype=np.float32
        )
        active_inhibitor_count = np.zeros(
            (batch_size, self.output_features), dtype=np.float32
        )
        outputs = np.zeros((batch_size, self.output_features), dtype=np.bool_)
        for i in range(self.output_features):
            active_excitator_count[:, i] = np.sum(
                inputs[:, self.excitators[i]], axis=1
            )
            active_inhibitor_count[:, i] = np.sum(
                inputs[:, self.inhibitors[i]], axis=1
            )
            outputs[:, i] = (
                active_excitator_count[:, i] - active_inhibitor_count[:, i]
            ) >= self.thresholds[i]

        # --- Feedback pass: predict input from output ---
        predicted_input = self._predict_input(outputs)

        # --- Prediction errors (local, at the input level) ---
        # too_much: input is active but was NOT predicted -> over-activation
        pred_too_much = inputs & ~predicted_input
        # not_enough: input is inactive but WAS predicted -> under-activation
        pred_not_enough = ~inputs & predicted_input

        return outputs, (
            inputs,
            outputs,
            active_excitator_count,
            active_inhibitor_count,
            predicted_input,
            pred_too_much.astype(np.float32),
            pred_not_enough.astype(np.float32),
        )

    def _predict_input(self, outputs: ArrayBool) -> ArrayBool:
        """
        Generate a top-down prediction of expected input given output activity.

        Uses the feedback pathway's sparse excitatory/inhibitory connections.
        """
        batch_size = outputs.shape[0]
        predicted = np.zeros((batch_size, self.input_features), dtype=np.bool_)

        for j in range(self.input_features):
            fb_exc_count = np.sum(outputs[:, self.fb_excitators[j]], axis=1)
            fb_inh_count = np.sum(outputs[:, self.fb_inhibitors[j]], axis=1)
            predicted[:, j] = (fb_exc_count - fb_inh_count) >= self.fb_thresholds[j]

        return predicted

    def backward(
        self,
        state,
        too_much: ArrayFloat,
        not_enough: ArrayFloat,
    ) -> Tuple[ArrayFloat, ArrayFloat]:
        """
        Backward pass: combine propagated errors with local prediction errors.

        The error signals passed to the input combine:
        1. Errors propagated backward through feedforward connections (as in Dense)
        2. Local prediction errors from the feedback pathway (weighted by
           prediction_weight)
        """
        (
            inputs,
            outputs,
            active_excitator_count,
            active_inhibitor_count,
            predicted_input,
            pred_too_much,
            pred_not_enough,
        ) = state

        batch_size = inputs.shape[0]

        assert too_much.shape == (batch_size, self.output_features)
        assert not_enough.shape == (batch_size, self.output_features)

        # --- Accumulate feedforward weight changes (same as Dense) ---
        self.threshold_changes += np.sum(too_much, axis=0)
        self.threshold_changes -= np.sum(not_enough, axis=0)
        self.excitator_potential += np.sum(
            inputs[:, np.newaxis, :] * not_enough[:, :, np.newaxis], axis=0
        )
        self.inhibitor_potential += np.sum(
            inputs[:, np.newaxis, :] * too_much[:, :, np.newaxis], axis=0
        )

        # --- Accumulate feedback weight changes ---
        # The feedback pathway learns from prediction errors at the input level
        self.fb_threshold_changes += np.sum(pred_too_much, axis=0)
        self.fb_threshold_changes -= np.sum(pred_not_enough, axis=0)
        self.fb_excitator_potential += np.sum(
            outputs[:, np.newaxis, :] * pred_not_enough[:, :, np.newaxis], axis=0
        )
        self.fb_inhibitor_potential += np.sum(
            outputs[:, np.newaxis, :] * pred_too_much[:, :, np.newaxis], axis=0
        )

        # --- Propagate errors to input (feedforward pathway, same as Dense) ---
        input_too_much = np.zeros(
            (batch_size, self.input_features), dtype=np.float32
        )
        input_not_enough = np.zeros(
            (batch_size, self.input_features), dtype=np.float32
        )

        for i in range(self.output_features):
            excitations = inputs[:, self.excitators[i]]
            inhibitions = inputs[:, self.inhibitors[i]]

            inhibitor_count = len(self.inhibitors[i])
            inactive_inhibitor_count = inhibitor_count - active_inhibitor_count[:, i]
            too_much_contributor_count = (
                active_excitator_count[:, i] + inactive_inhibitor_count
            )
            too_much_contributions = np.divide(
                too_much[:, i],
                too_much_contributor_count,
                where=too_much_contributor_count != 0,
            )
            input_too_much[:, self.excitators[i]] += (
                excitations * too_much_contributions[:, np.newaxis]
            )
            input_not_enough[:, self.inhibitors[i]] += (
                ~inhibitions * too_much_contributions[:, np.newaxis]
            )

            excitator_count = len(self.excitators[i])
            inactive_excitator_count = (
                excitator_count - active_excitator_count[:, i]
            )
            not_enough_contributor_count = (
                active_inhibitor_count[:, i] + inactive_excitator_count
            )
            not_enough_contributions = np.divide(
                not_enough[:, i],
                not_enough_contributor_count,
                where=not_enough_contributor_count != 0,
            )
            input_not_enough[:, self.excitators[i]] += (
                ~excitations * not_enough_contributions[:, np.newaxis]
            )
            input_too_much[:, self.inhibitors[i]] += (
                inhibitions * not_enough_contributions[:, np.newaxis]
            )

        # --- Add local prediction errors (weighted) ---
        input_too_much += self.prediction_weight * pred_too_much
        input_not_enough += self.prediction_weight * pred_not_enough

        return input_too_much, input_not_enough

    def update(self):
        """Update both feedforward and feedback weights."""
        self._update_feedforward()
        self._update_feedback()
        self._reset_state()

    def _update_feedforward(self):
        """Update feedforward (bottom-up) weights, same logic as Dense."""
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
                potential = (
                    self.excitator_potential[i] - self.inhibitor_potential[i]
                )
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

    def _update_feedback(self):
        """Update feedback (top-down) weights using prediction error signals."""
        for j in range(self.input_features):
            change_threshold = 0.01

            if np.random.rand() < self.treshold_change_chance:
                if self.fb_threshold_changes[j] > change_threshold:
                    plus_minus_one = 1
                elif self.fb_threshold_changes[j] < -change_threshold:
                    plus_minus_one = -1
                else:
                    plus_minus_one = 0
                self.fb_thresholds[j] += plus_minus_one

            changed = False
            if np.random.rand() < self.synapse_change_chance:
                potential = (
                    self.fb_excitator_potential[j] - self.fb_inhibitor_potential[j]
                )
                max_k = np.argmax(np.abs(potential))
                if potential[max_k] > change_threshold:
                    self.fb_excitators[j].append(max_k)
                    changed = True
                elif potential[max_k] < -change_threshold:
                    self.fb_inhibitors[j].append(max_k)
                    changed = True

            if changed:
                fb_max_synapses = self.output_features // 10
                if fb_max_synapses < 1:
                    fb_max_synapses = self.output_features
                self.fb_excitators[j], self.fb_inhibitors[j] = normalize_synapses(
                    self.fb_excitators[j],
                    self.fb_inhibitors[j],
                    fb_max_synapses,
                )

    def _reset_state(self):
        """Reset accumulated learning signals."""
        # Feedforward accumulators
        self.threshold_changes = np.zeros(self.output_features, dtype=np.float32)
        self.excitator_potential = np.zeros(
            (self.output_features, self.input_features), dtype=np.float32
        )
        self.inhibitor_potential = np.zeros(
            (self.output_features, self.input_features), dtype=np.float32
        )
        # Feedback accumulators
        self.fb_threshold_changes = np.zeros(self.input_features, dtype=np.float32)
        self.fb_excitator_potential = np.zeros(
            (self.input_features, self.output_features), dtype=np.float32
        )
        self.fb_inhibitor_potential = np.zeros(
            (self.input_features, self.output_features), dtype=np.float32
        )

    def get_prediction_error(self, inputs: ArrayBool, outputs: ArrayBool):
        """
        Compute and return the prediction error for given inputs and outputs.

        Returns the total prediction error (sum of too_much and not_enough
        across all input features and batch samples).
        """
        predicted = self._predict_input(outputs)
        too_much = np.sum(inputs & ~predicted)
        not_enough = np.sum(~inputs & predicted)
        return too_much + not_enough


class PredictiveSequence(Layer):
    """
    A sequence of PredictiveCodingLayers that supports iterative inference.

    In predictive coding, inference involves running multiple forward-backward
    passes to allow prediction errors to settle before updating weights. This
    class implements that iterative settling process.
    """

    def __init__(self, layers: list, inference_steps: int = 1):
        self.layers = layers
        self.inference_steps = inference_steps

    def forward(self, input_data: ArrayBool):
        """Run forward through all layers, returning output and state."""
        state = []
        for layer in self.layers:
            input_data, layer_state = layer.forward(input_data)
            state.append(layer_state)
        return input_data, state

    def backward(
        self,
        state,
        too_much: ArrayFloat,
        not_enough: ArrayFloat,
    ) -> Tuple[ArrayFloat, ArrayFloat]:
        """Run backward through all layers."""
        layers_with_state = list(zip(self.layers, state))
        for layer, layer_state in reversed(layers_with_state):
            too_much, not_enough = layer.backward(layer_state, too_much, not_enough)
        return too_much, not_enough

    def update(self) -> None:
        """Update all layers."""
        for layer in self.layers:
            layer.update()

    def train_step(
        self,
        inputs: ArrayBool,
        targets: ArrayBool,
    ) -> float:
        """
        Run a full training step with iterative inference.

        For each inference step:
        1. Forward pass
        2. Compute output error from targets
        3. Backward pass (propagating errors and accumulating prediction errors)

        Then update weights once.

        Returns the output error rate (fraction of wrong output bits).
        """
        total_errors = 0
        for _ in range(self.inference_steps):
            output, state = self.forward(inputs)
            too_much = (output > targets).astype(np.float32)
            not_enough = (output < targets).astype(np.float32)
            total_errors = np.sum(too_much) + np.sum(not_enough)
            self.backward(state, too_much, not_enough)

        self.update()

        error_rate = total_errors / (targets.shape[0] * targets.shape[1])
        return error_rate
