from itertools import product

import numpy
from tqdm import tqdm


class BN:
    def __init__(self, sources_dim, sources):
        self.excitators = []
        self.inhibitors = []
        self.sources_dim = sources_dim
        self.sources = sources
        for s in self.sources:
            assert len(s) == self.sources_dim
        self.activation_treshold = 0

    def get_absolute_sources_slice(self, positions, sources_relative):
        sources_absolute = sources_relative[:, numpy.newaxis] + positions[numpy.newaxis, :]
        sources_slice = tuple(
            sources_absolute[:, :, i]
            for i in range(self.sources_dim)
        )
        return sources_slice

    def get_signal_slice(self, positions):
        positions_slice = tuple(
            positions[:, i]
            for i in range(self.sources_dim)
        )
        return positions_slice

    def get_inputs(self, positions, signal, dtype=None):
        excitators_slice = self.get_absolute_sources_slice(positions, self.sources[self.excitators])
        inhibitors_slice = self.get_absolute_sources_slice(positions, self.sources[self.inhibitors])
        excitations = signal[excitators_slice]
        inhibitions = signal[inhibitors_slice]
        excitation_sum = numpy.sum(excitations, axis=0, dtype=dtype)
        inhibition_sum = numpy.sum(inhibitions, axis=0, dtype=dtype)
        return excitators_slice, inhibitors_slice, excitations, inhibitions, excitation_sum, inhibition_sum

    def infer(self, positions, signal, dtype=None):
        _, _, _, _, excitation_sum, inhibition_sum = self.get_inputs(positions, signal, dtype=dtype)
        signal[self.get_signal_slice(positions)] = (excitation_sum - inhibition_sum) >= self.activation_treshold

    def compute_error(self, positions, signal, too_much, not_enough, dtype=None):
        (
            excitators_slice, inhibitors_slice,
            excitations, inhibitions,
            excitation_sum, inhibition_sum,
        ) = self.get_inputs(positions, signal, dtype=dtype)
        too_much_contributor_count = excitation_sum + len(self.inhibitors) - inhibition_sum
        not_enough_contributor_count = inhibition_sum + len(self.excitators) - excitation_sum
        if numpy.issubdtype(dtype, numpy.integer):
            div_func = numpy.floor_divide
        else:
            div_func = numpy.divide
        signal_slice = self.get_signal_slice(positions)
        too_much_contributions = div_func(too_much[signal_slice], too_much_contributor_count, dtype=dtype)
        not_enough_contributions = div_func(not_enough[signal_slice], not_enough_contributor_count, dtype=dtype)
        too_much[excitators_slice] += excitations * too_much_contributions
        not_enough[inhibitors_slice] += ~inhibitions * too_much_contributions
        not_enough[excitators_slice] += ~excitations * not_enough_contributions
        too_much[inhibitors_slice] += inhibitions * not_enough_contributions

    def get_signal_error_correlation(
        self, positions, signal,
        too_much, not_enough,
        dtype=None,
    ):
        source_signal = signal[self.get_absolute_sources_slice(positions, self.sources)]
        positions_slice = self.get_signal_slice(positions)

        return (
            numpy.sum(
                too_much[positions_slice] * source_signal,
                axis=1,
                dtype=dtype,
            ),
            numpy.sum(
                not_enough[positions_slice] * source_signal,
                axis=1,
                dtype=dtype,
            ),
        )


class ConvBNN:
    def __init__(self, dim, margin, sample_depth, neuron_count, **kwargs):
        self.dim = dim
        self.margin = margin
        self.neuron_count = neuron_count
        self.sample_depth = sample_depth

        self.neurons = []
        for i in range(self.neuron_count):
            self.neurons.append(BN(
                sources_dim=self.dim + 1,
                sources=numpy.array(
                    list(product(
                        range(-i - sample_depth, 0),  # input, previous layers
                        *(  # convolution in each direction
                            [-1, 0, 1]
                            for _ in range(self.dim)
                        ),
                    )),
                ),
                **kwargs,
            ))

    def make_workplace(self, sample_size, batch_size, dtype=None):
        assert len(sample_size) == self.dim
        workplace = numpy.zeros((
            self.sample_depth + self.neuron_count,
            *(s + 2 * self.margin for s in sample_size),
            batch_size,
        ), dtype=dtype)
        return workplace

    def get_sample_size(self, workplace):
        depth, *sample_size, batch_size = workplace.shape
        assert len(sample_size) == self.dim
        assert depth >= self.sample_depth + len(self.neurons)
        return sample_size

    def get_sample_positions(self, sample_size):
        return numpy.array(list(product(*(
            range(self.margin, s - self.margin)
            for s in sample_size
        ))))

    def infer(self, signal, dtype=None):
        sample_size = self.get_sample_size(signal)
        sample_positions = self.get_sample_positions(sample_size)

        for i, neuron in tqdm(
            enumerate(self.neurons),
            total=len(self.neurons), desc='infer         ',
        ):
            pos = numpy.concatenate((
                [[i + self.sample_depth]] * sample_positions.shape[0],
                sample_positions,
            ), axis=1)
            neuron.infer(
                pos,
                signal,
                dtype=dtype,
            )

    def compute_error(self, signal, too_much, not_enough, dtype=None):
        sample_size = self.get_sample_size(signal)
        sample_positions = self.get_sample_positions(sample_size)

        for i, neuron in tqdm(
            reversed(list(enumerate(self.neurons))),
            total=len(self.neurons), desc='backprop error',
        ):
            pos = numpy.concatenate((
                [[i + self.sample_depth]] * sample_positions.shape[0],
                sample_positions,
            ), axis=1)
            neuron.compute_error(
                pos,
                signal, too_much, not_enough,
                dtype=dtype,
            )
