import numpy as np

from bnn.abstract import Sequence
from bnn.dense import Dense, normalize_synapses


def test_dense_layer_forward_batch():
    layer = Dense(input_features=4, output_features=2)
    layer.excitators = [[0, 2], [1, 3]]
    layer.inhibitors = [[1], [2]]
    layer.thresholds = np.array([1, 1])

    # Test with a batch of samples
    inputs = np.array([[True, False, True, False], [False, True, False, True]])
    expected_output = np.array([[True, False], [False, True]])

    output, _ = layer.forward(inputs)
    np.testing.assert_array_equal(output, expected_output)


def test_dense_layer_backward():
    layer = Dense(input_features=4, output_features=2)
    layer.excitators = [[0, 2], [1, 3]]
    layer.inhibitors = [[1], [2]]
    layer.thresholds = np.array([1, 1])
    inputs = np.array([[True, False, True, False]])
    _, state = layer.forward(inputs)
    too_much, not_enough = np.array([[0.6, 0.0]]), np.array([[0.0, 0.9]])
    input_too_much, input_not_enough = layer.backward(state, too_much, not_enough)
    np.testing.assert_array_almost_equal(input_too_much, np.array([[
        0.2,
        0.0,
        0.2 + 0.3,
        0.0,
    ]]))
    np.testing.assert_array_almost_equal(input_not_enough, np.array([[
        0.0,
        0.2 + 0.3,
        0.0,
        0.0 + 0.3,
    ]]))


def test_sequence_forward():
    network = Sequence([
        Dense(input_features=4, output_features=3),
        Dense(input_features=3, output_features=2),
    ])

    # Test with a batch of samples
    inputs = np.array([[True, False, True, False], [False, True, False, True]])
    output, _ = network.forward(inputs)
    assert output.shape == (2, 2)
    assert output.dtype == bool


def test_sequence_backward():
    network = Sequence([
        Dense(input_features=4, output_features=3),
        Dense(input_features=3, output_features=2),
    ])

    # Test input
    inputs = np.array([[True, False, True, False]])

    # Run forward pass
    _, state = network.forward(inputs)

    # Create some dummy feedback
    too_much = np.array([[0.5, 0.0]])
    not_enough = np.array([[0.0, 0.5]])

    # Run backward pass
    input_too_much, input_not_enough = network.backward(
        state,
        too_much, not_enough,
    )

    # Check if the backward pass produces outputs of the correct shape
    assert input_too_much.shape == inputs.shape
    assert input_not_enough.shape == inputs.shape


def test_sequence_update():
    network = Sequence([
        Dense(input_features=4, output_features=3),
        Dense(input_features=3, output_features=2),
    ])

    # Run a forward and backward pass to accumulate some changes
    inputs = np.array([[True, False, True, False]])
    _, state = network.forward(inputs)
    too_much = np.array([[0.5, 0.0]])
    not_enough = np.array([[0.0, 0.5]])
    network.backward(state, too_much, not_enough)

    # Run update
    network.update()

    # The update method doesn't return anything, so we just check that it runs without errors
    assert True


def test_end_to_end_training():
    network = Sequence([
        Dense(
            input_features=4,
            output_features=3,
            max_synapses=4,
        ),
        Dense(
            input_features=3,
            output_features=2,
            max_synapses=3,
        ),
    ])

    # Create a simple dataset
    inputs = np.array([
        [True, False, True, False],
        [False, True, False, True],
        [True, True, False, False],
        [False, False, True, True],
    ])
    targets = np.array([
        [True, False],
        [False, True],
        [True, True],
        [False, False],
    ])

    # Train for a few epochs
    for epoch in range(100):
        output, state = network.forward(inputs)
        too_much = (output > targets).astype(float)
        not_enough = (output < targets).astype(float)
        network.backward(state, too_much, not_enough)
        network.update()

    # Test the network after training
    correct_predictions = 0
    for input_sample, target_sample in zip(inputs, targets):
        output, _ = network.forward(input_sample[np.newaxis, :])
        if np.array_equal(output[0], target_sample):
            correct_predictions += 1

    # Check if the network has learned something (at least 50% accuracy)
    assert correct_predictions >= 2

    assert False, correct_predictions


def test_normalize_synapses_remove_cancelling():
    e, i = normalize_synapses([0, 0, 1], [0, 1, 2], 1)
    assert e == [0]
    assert i == [2]


def test_normalize_synapses_sort():
    e, i = normalize_synapses([0, 2, 1], [], 100)
    assert e == [0, 1, 2]
    assert i == []
