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
            sources_absolute[:,i]
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

"""
bnn = ConvBNN(
    dim=2,
    margin=1,
    sample_depth=3,
    neuron_count=7,
    synapses_count=4,
    activation_treshold=2,
)

sample_size = 5
batch_size = 2

input = numpy.random.rand(bnn.sample_depth, sample_size, sample_size, batch_size) > 0.5
input[0] = False
input[1] = True

workplace = bnn.make_workplace(input)
print(workplace.shape)
bnn.infer(workplace)

exit(0)

class BNN:
    def __init__(self, neuron_count, synapses_per_neuron, symmetries):
        self.neuron_count = neuron_count
        self.synapses_per_neuron = synapses_per_neuron
        self.symmetries = symmetries

        self.synapses = numpy.zeros(
            (self.neuron_count, self.synapses_per_neuron, len(self.symmetries) + 1),
            dtype=int,
        )
        #self.synapses_relative = numpy.zeros(
        #    (self.neuron_count, self.synapses_per_neuron),
        #    dtype=(int, len(self.symmetries)),
        #)
        self.tresholds = numpy.zeros((self.neuron_count,), dtype=int)

    def infer(self, input):
        input_size, *input_dim = input.shape
        result = numpy.zeros([self.neuron_count] + input_dim, dtype=bool)
        result[:input_size] = input

        indices = numpy.indices(
            [self.neuron_count] + input_dim,
        )
        print(input.shape, result.shape, indices.shape)
        #print('indices\n', indices)
        #3/0

        for i in range(self.neuron_count):
            synapses_diff = indices[:,i,:]
            print(synapses_diff)
            synapses_abs = self.synapses[i] + synapses_diff
            activations = result[synapses_abs]
            #print('l', i, '\n', activations)

            result[i]

        return result


bnn = BNN(
    neuron_count=7,
    synapses_per_neuron=4,
    flow=[  # conv 1d
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ],
    #flow=[  # recurrent 1d
    #    (None, (-inf, -1)),
    #    (-1, 0),
    #],
)
input = numpy.random.rand(2, 15) > 0.5
input[0] = True
#print('input\n', input)
result = bnn.infer(input)
print(result)
"""
