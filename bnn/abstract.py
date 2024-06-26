from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

ArrayBool = np.ndarray  # For boolean arrays (inputs and outputs)
ArrayFloat = np.ndarray  # For floating-point arrays (error signals)


class Layer(ABC):
    """
    Abstract interface for a binary, sparse network layer
    """

    @abstractmethod
    def forward(self, inputs: ArrayBool) -> ArrayBool:
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
        inputs: ArrayBool,
        outputs: ArrayBool,
        too_much: ArrayFloat,
        not_enough: ArrayFloat,
    ) -> Tuple[ArrayFloat, ArrayFloat]:
        """
        Perform the backward pass of the layer.

        Args:
            inputs (ArrayBool): Input data that was fed to this layer in the forward pass.
                Shape: (batch_size, ...), where ... represents any number of dimensions.
            outputs (ArrayBool): Output data that this layer produced in the forward pass.
                Shape: (batch_size, ...), where ... represents any number of dimensions.
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


class Network(Layer):
    """
    Network consists of multiple layers organized in a directed acyclic graph (DAG).
    Network as a whole can be treated as a single layer.
    """

    def __init__(self):
        self.layers: List[Layer] = []
        self.connections: Dict[str, List[str]] = {}  # To store the DAG structure

    def add_layer(self, layer: Layer, inputs: List[str] = None) -> str:
        layer_id = f"layer_{len(self.layers)}"
        self.layers.append(layer)
        self.connections[layer_id] = inputs or []
        return layer_id

    def forward(self, input_data: ArrayBool) -> ArrayBool:
        layer_outputs = {"input": input_data}
        for i, layer in enumerate(self.layers):
            layer_id = f"layer_{i}"
            if not self.connections[layer_id]:
                layer_input = input_data
            else:
                layer_input = np.concatenate(
                    [layer_outputs[inp] for inp in self.connections[layer_id]], axis=-1
                )
            layer_outputs[layer_id] = layer.forward(layer_input)
        return layer_outputs[f"layer_{len(self.layers) - 1}"]

    def backward(
        self,
        inputs: ArrayBool,
        outputs: ArrayBool,
        too_much: ArrayFloat,
        not_enough: ArrayFloat,
    ) -> Tuple[ArrayFloat, ArrayFloat]:
        layer_inputs = {"input": inputs}
        layer_outputs = {"input": inputs}
        feedback = {f"layer_{len(self.layers) - 1}": (too_much, not_enough)}

        # Forward pass to store inputs and outputs for each layer
        for i, layer in enumerate(self.layers):
            layer_id = f"layer_{i}"
            if not self.connections[layer_id]:
                layer_input = inputs
            else:
                layer_input = np.concatenate(
                    [layer_outputs[inp] for inp in self.connections[layer_id]], axis=-1
                )
            layer_inputs[layer_id] = layer_input
            layer_outputs[layer_id] = layer.forward(layer_input)

        # Backward pass
        for i in reversed(range(len(self.layers))):
            layer_id = f"layer_{i}"
            layer = self.layers[i]
            layer_too_much, layer_not_enough = feedback[layer_id]
            input_too_much, input_not_enough = layer.backward(
                layer_inputs[layer_id],
                layer_outputs[layer_id],
                layer_too_much,
                layer_not_enough,
            )
            for inp in self.connections[layer_id]:
                if inp not in feedback:
                    feedback[inp] = (input_too_much, input_not_enough)
                else:
                    prev_too_much, prev_not_enough = feedback[inp]
                    feedback[inp] = (
                        prev_too_much + input_too_much,
                        prev_not_enough + input_not_enough,
                    )

        return feedback.get(
            "input", (np.zeros_like(too_much), np.zeros_like(not_enough))
        )

    def update(self) -> None:
        for layer in self.layers:
            layer.update()
