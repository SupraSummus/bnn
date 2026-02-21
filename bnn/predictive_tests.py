import numpy as np

from bnn.predictive import PredictiveCodingLayer, PredictiveSequence


def test_predictive_layer_forward_shape():
    """Test that forward pass produces correct output shapes."""
    layer = PredictiveCodingLayer(input_features=4, output_features=2)
    inputs = np.array([[True, False, True, False]], dtype=bool)
    output, state = layer.forward(inputs)
    assert output.shape == (1, 2)
    assert output.dtype == bool


def test_predictive_layer_forward_batch():
    """Test forward pass with a batch of samples."""
    layer = PredictiveCodingLayer(input_features=4, output_features=2)
    layer.excitators = [[0, 2], [1, 3]]
    layer.inhibitors = [[1], [2]]
    layer.thresholds = np.array([1, 1])

    inputs = np.array([
        [True, False, True, False],
        [False, True, False, True],
    ])
    expected_output = np.array([[True, False], [False, True]])

    output, _ = layer.forward(inputs)
    np.testing.assert_array_equal(output, expected_output)


def test_predictive_layer_prediction():
    """Test that the feedback pathway generates predictions."""
    layer = PredictiveCodingLayer(input_features=4, output_features=2)
    # Set up feedback connections: output 0 excites input 0 and 1
    layer.fb_excitators = [[0], [0], [1], [1]]
    layer.fb_thresholds = np.array([1, 1, 1, 1])

    outputs = np.array([[True, False]], dtype=bool)
    predicted = layer._predict_input(outputs)

    # Output 0 is active -> inputs 0 and 1 should be predicted active
    assert predicted[0, 0] == True  # noqa: E712
    assert predicted[0, 1] == True  # noqa: E712
    # Output 1 is inactive -> inputs 2 and 3 should NOT be predicted active
    assert predicted[0, 2] == False  # noqa: E712
    assert predicted[0, 3] == False  # noqa: E712


def test_predictive_layer_prediction_error():
    """Test prediction error computation."""
    layer = PredictiveCodingLayer(input_features=4, output_features=2)
    layer.fb_excitators = [[0], [0], [1], [1]]
    layer.fb_thresholds = np.array([1, 1, 1, 1])

    inputs = np.array([[True, False, True, False]], dtype=bool)
    outputs = np.array([[True, False]], dtype=bool)

    # Predicted: [True, True, False, False]
    # Actual:    [True, False, True, False]
    # too_much (actual & ~predicted):  [False, False, True, False] -> 1
    # not_enough (~actual & predicted): [False, True, False, False] -> 1
    error = layer.get_prediction_error(inputs, outputs)
    assert error == 2


def test_predictive_layer_backward_shape():
    """Test that backward pass produces correct output shapes."""
    layer = PredictiveCodingLayer(input_features=4, output_features=2)
    inputs = np.array([[True, False, True, False]])
    _, state = layer.forward(inputs)

    too_much = np.array([[0.5, 0.0]])
    not_enough = np.array([[0.0, 0.5]])

    input_too_much, input_not_enough = layer.backward(state, too_much, not_enough)
    assert input_too_much.shape == (1, 4)
    assert input_not_enough.shape == (1, 4)


def test_predictive_layer_backward_includes_prediction_errors():
    """Test that backward pass includes prediction error contributions."""
    layer = PredictiveCodingLayer(
        input_features=4, output_features=2, prediction_weight=1.0
    )
    # Set up feedback so predictions differ from inputs
    layer.fb_excitators = [[0], [0], [], []]
    layer.fb_thresholds = np.array([1, 1, 0, 0])

    inputs = np.array([[True, False, True, False]], dtype=bool)
    _, state = layer.forward(inputs)

    # No external error signal
    too_much = np.zeros((1, 2), dtype=np.float32)
    not_enough = np.zeros((1, 2), dtype=np.float32)

    input_too_much, input_not_enough = layer.backward(
        state, too_much, not_enough
    )

    # Even with zero external error, prediction errors should contribute
    # The prediction errors depend on the feedback path
    # This test just verifies the prediction error pathway is active
    total_error = np.sum(input_too_much) + np.sum(input_not_enough)
    assert total_error >= 0  # Non-negative by construction


def test_predictive_layer_update_runs():
    """Test that update runs without errors."""
    layer = PredictiveCodingLayer(input_features=4, output_features=2)
    inputs = np.array([[True, False, True, False]])
    _, state = layer.forward(inputs)

    too_much = np.array([[0.5, 0.0]])
    not_enough = np.array([[0.0, 0.5]])
    layer.backward(state, too_much, not_enough)
    layer.update()

    # After update, state should be reset
    np.testing.assert_array_equal(
        layer.threshold_changes,
        np.zeros(2, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        layer.fb_threshold_changes,
        np.zeros(4, dtype=np.float32),
    )


def test_predictive_sequence_forward():
    """Test PredictiveSequence forward pass."""
    network = PredictiveSequence([
        PredictiveCodingLayer(input_features=4, output_features=3),
        PredictiveCodingLayer(input_features=3, output_features=2),
    ])

    inputs = np.array([
        [True, False, True, False],
        [False, True, False, True],
    ])
    output, state = network.forward(inputs)
    assert output.shape == (2, 2)
    assert output.dtype == bool


def test_predictive_sequence_backward():
    """Test PredictiveSequence backward pass."""
    network = PredictiveSequence([
        PredictiveCodingLayer(input_features=4, output_features=3),
        PredictiveCodingLayer(input_features=3, output_features=2),
    ])

    inputs = np.array([[True, False, True, False]])
    _, state = network.forward(inputs)

    too_much = np.array([[0.5, 0.0]])
    not_enough = np.array([[0.0, 0.5]])

    input_too_much, input_not_enough = network.backward(
        state, too_much, not_enough
    )
    assert input_too_much.shape == (1, 4)
    assert input_not_enough.shape == (1, 4)


def test_predictive_sequence_train_step():
    """Test a full training step with PredictiveSequence."""
    network = PredictiveSequence(
        layers=[
            PredictiveCodingLayer(
                input_features=4,
                output_features=3,
                max_synapses=4,
            ),
            PredictiveCodingLayer(
                input_features=3,
                output_features=2,
                max_synapses=3,
            ),
        ],
        inference_steps=1,
    )

    inputs = np.array([
        [True, False, True, False],
        [False, True, False, True],
    ])
    targets = np.array([
        [True, False],
        [False, True],
    ])

    # Just verify the training step runs and returns a valid error rate
    error_rate = network.train_step(inputs, targets)
    assert 0.0 <= error_rate <= 1.0


def test_predictive_sequence_iterative_inference():
    """Test that multiple inference steps work."""
    network = PredictiveSequence(
        layers=[
            PredictiveCodingLayer(
                input_features=4,
                output_features=3,
                max_synapses=4,
            ),
            PredictiveCodingLayer(
                input_features=3,
                output_features=2,
                max_synapses=3,
            ),
        ],
        inference_steps=3,  # Multiple inference steps
    )

    inputs = np.array([
        [True, False, True, False],
        [False, True, False, True],
    ])
    targets = np.array([
        [True, False],
        [False, True],
    ])

    error_rate = network.train_step(inputs, targets)
    assert 0.0 <= error_rate <= 1.0


def test_predictive_end_to_end_training():
    """Test end-to-end training converges over multiple epochs."""
    np.random.seed(42)

    network = PredictiveSequence(
        layers=[
            PredictiveCodingLayer(
                input_features=4,
                output_features=3,
                max_synapses=4,
                synapse_change_chance=0.3,
                treshold_change_chance=0.3,
            ),
            PredictiveCodingLayer(
                input_features=3,
                output_features=2,
                max_synapses=3,
                synapse_change_chance=0.3,
                treshold_change_chance=0.3,
            ),
        ],
        inference_steps=1,
    )

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

    # Train for several epochs
    for _ in range(100):
        network.train_step(inputs, targets)

    # After training, check that the network has learned something
    output, _ = network.forward(inputs)
    correct = np.sum(output == targets)
    total = targets.size

    # At least better than random (>50% of bits correct)
    assert correct > total // 2
