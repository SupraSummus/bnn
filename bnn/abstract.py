from abc import abstractmethod
from typing import Any, List, Tuple

import numpy as np

ArrayBool = np.ndarray  # For boolean arrays (inputs and outputs)
ArrayFloat = np.ndarray  # For floating-point arrays (error signals)
InternalState = Any  # For internal state of the layer


class Layer:
    """
    Abstract interface for a binary, sparse network layer
    """

    @abstractmethod
    def forward(self, inputs: ArrayBool) -> Tuple[ArrayBool, InternalState]:
        """
        Perform the forward pass of the layer.

        Args:
            inputs (ArrayBool): Input data to the layer.
                Shape: (batch_size, ...), where ... represents any number of dimensions.

        Returns:
            ArrayBool: Output of the layer.
                Shape: (batch_size, ...), where ... represents any number of dimensions.
        """
        pass

    @abstractmethod
    def backward(
        self,
        internal_state: InternalState,
        too_much: ArrayFloat,
        not_enough: ArrayFloat,
    ) -> Tuple[ArrayFloat, ArrayFloat]:
        """
        Perform the backward pass of the layer.

        Args:
            too_much (ArrayFloat): Feedback indicating where the output was too high.
                Shape: (batch_size, ...), matching the shape of the layer's output.
            not_enough (ArrayFloat): Feedback indicating where the output was too low.
                Shape: (batch_size, ...), matching the shape of the layer's output.

        Returns:
            Tuple[ArrayFloat, ArrayFloat]: Tuple of (too_much, not_enough) for the layer's input.
                Each array shape: (batch_size, ...), matching the shape of the layer's input.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the layer's parameters based on accumulated feedback.
        """
        pass


class Sequence(Layer):
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, input_data: ArrayBool):
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
        layers_with_state = list(zip(self.layers, state))
        for layer, state in reversed(layers_with_state):
            too_much, not_enough = layer.backward(state, too_much, not_enough)
        return too_much, not_enough

    def update(self) -> None:
        for layer in self.layers:
            layer.update()
