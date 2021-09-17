from itertools import product

import numpy


class BN:
    def __init__(self, sources_dim, sources):
        self.excitators = []
        self.inhibitors = []
        self.sources_dim = sources_dim
        self.sources = sources
        for s in self.sources:
            assert len(s) == self.sources_dim
        self.activation_treshold = 0

    def get_absolute_sources_slice(self, position, source_indices):
        sources_absolute = self.sources[source_indices] + position
        sources_slice = tuple(
            sources_absolute[:, i]
            for i in range(self.sources_dim)
        )
        return sources_slice

    def get_inputs(self, position, signal, dtype=None):
        excitators_slice = self.get_absolute_sources_slice(position, self.excitators)
        inhibitors_slice = self.get_absolute_sources_slice(position, self.inhibitors)
        excitations = signal[excitators_slice]
        inhibitions = signal[inhibitors_slice]
        excitation_sum = numpy.sum(excitations, axis=0, dtype=dtype)
        inhibition_sum = numpy.sum(inhibitions, axis=0, dtype=dtype)
        return excitators_slice, inhibitors_slice, excitations, inhibitions, excitation_sum, inhibition_sum

    def infer(self, position, signal):
        _, _, _, _, excitation_sum, inhibition_sum = self.get_inputs(position, signal)
        signal[position] = (excitation_sum - inhibition_sum) >= self.activation_treshold

    def compute_error(self, position, signal, too_much, not_enough, dtype=None):
        (
            excitators_slice, inhibitors_slice,
            excitations, inhibitions,
            excitation_sum, inhibition_sum,
        ) = self.get_inputs(position, signal, dtype=dtype)
        too_much_contributor_count = excitation_sum + len(self.inhibitors) - inhibition_sum
        not_enough_contributor_count = inhibition_sum + len(self.excitators) - excitation_sum
        if numpy.issubdtype(dtype, numpy.integer):
            div_func = numpy.floor_divide
        else:
            div_func = numpy.divide
        too_much_contributions = div_func(too_much[position], too_much_contributor_count, dtype=dtype)
        not_enough_contributions = div_func(not_enough[position], not_enough_contributor_count, dtype=dtype)
        too_much[excitators_slice] += excitations * too_much_contributions
        not_enough[inhibitors_slice] += ~inhibitions * too_much_contributions
        not_enough[excitators_slice] += ~excitations * not_enough_contributions
        too_much[inhibitors_slice] += inhibitions * not_enough_contributions

    def compute_sources_error_correlation(
        self, position, signal,
        too_much, not_enough,
        too_much_correlation, not_enough_correlation,
    ):
        source_signal = signal[self.sources + position]
        too_much_correlation += too_much[position] * source_signal
        not_enough_correlation += not_enough[position] * source_signal


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

    def make_workplace(self, input):
        sample_depth, *sample_size, batch_size = input.shape
        assert sample_depth == self.sample_depth
        assert len(sample_size) == self.dim
        workplace = numpy.zeros((
            self.sample_depth + self.neuron_count,
            *(s + 2 * self.margin for s in sample_size),
            batch_size,
        ), dtype=bool)
        return workplace

    def get_sample_size(self, workplace):
        depth, *sample_size, batch_size = workplace.shape
        assert len(sample_size) == self.dim
        assert depth >= self.sample_depth + len(self.neurons)
        return sample_size

    def get_sample_positions(self, sample_size):
        return product(*(
            range(self.margin, s - self.margin)
            for s in sample_size
        ))

    def infer(self, workplace):
        sample_size = self.get_sample_size(workplace)

        for i, neuron in enumerate(self.neurons):
            for pos in self.get_sample_positions(sample_size):
                neuron.infer((
                    i + self.sample_depth,
                    *pos,
                ), workplace)

    def compute_error(self, signal, too_much, not_enough):
        sample_size = self.get_sample_size(signal)
        for i, neuron in reversed(enumerate(self.neurons)):
            for pos in self.get_sample_positions(sample_size):
                neuron.compute_error((
                    i + self.sample_depth,
                    *pos,
                ), signal, too_much, not_enough)
