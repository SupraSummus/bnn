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

    def infer(self, position, result):
        print('infer', position)
        excitations = result[self.get_absolute_sources_slice(position, self.excitators)]
        excitation_sum = numpy.sum(excitations, axis=0)
        inhibitions = result[self.get_absolute_sources_slice(position, self.inhibitors)]
        inhibition_sum = numpy.sum(inhibitions, axis=0)
        result[position] = (excitation_sum - inhibition_sum) >= self.activation_treshold

    def compute_error(self, position, result, too_much, not_enough):
        print('compute_error', position)
        excitators_slice = self.get_absolute_sources_slice(position, self.excitators)
        inhibitors_slice = self.get_absolute_sources_slice(position, self.inhibitors)
        excitations = result[excitators_slice]
        inhibitions = result[inhibitors_slice]
        excitation_sum = numpy.sum(excitations, axis=0)
        inhibition_sum = numpy.sum(inhibitions, axis=0)
        too_much_contributor_count = excitation_sum + len(self.inhibitors) - inhibition_sum
        not_enough_contributor_count = inhibition_sum + len(self.excitators) - excitation_sum
        too_much[excitators_slice] += too_much[position] * excitations / too_much_contributor_count
        not_enough[inhibitors_slice] += too_much[position] * ~inhibitions / too_much_contributor_count
        not_enough[excitators_slice] += not_enough[position] * ~excitations / not_enough_contributor_count
        too_much[inhibitors_slice] += not_enough[position] * inhibitions / not_enough_contributor_count


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

    def infer(self, workplace):
        depth, *sample_size, batch_size = workplace.shape
        assert len(sample_size) == self.dim
        assert depth >= self.sample_depth + len(self.neurons)

        for i, neuron in enumerate(self.neurons):
            for pos in product(*(
                range(self.margin, s - self.margin)
                for s in sample_size
            )):
                neuron.infer((
                    i + self.sample_depth,
                    *pos,
                ), workplace)
