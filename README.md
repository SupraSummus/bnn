    poetry install
    poetry shell

    pytest

    mnist_get_data.sh
    mnist_preview

    python -m bnn.mnist

## Idea

I'd like to talk about my idea for binary, sparse neural nets that learn in a non-gradient-descent manner, but through correlation of inputs/outputs.

The basic idea is that we have, let's say, "binary neurons". Similar to how it works in classic neural nets, and drawing inspiration from biological nets, each neuron has many inputs and one output. Now the binary aspect kicks in - each input/synapse either "is" or "is-not". So the weight is 1 or 0. But also the inputs are divided into two groups. We have exciting inputs and inhibiting inputs. So in classic nets, it correlates with a weight of -1, 0, or 1. Then the neuron has an activation threshold which is an integer. So this is like a step function, aka a very sharp sigmoid. When the sum of inputs is below the threshold, the neuron does not fire (output 0); else it fires (output 1). So this is the binary part of the idea. Now for the sparse idea, when we describe to which outputs from the previous layer the neuron is connected, we don't use a dense list of weights, but we store only pointers. So we have one list of indices that excite the neuron and a list that inhibits the neuron.

For the training, I envision this like that: a neuron can be wrong in two distinct ways:

1. It is inactive but should be active -- aka "not enough excitation"
2. It is active but should be inactive -- aka "too much excitation"

This all assumes we are learning on training data - input and desired output pairs.

To correct case 1, we have options:

* 1.1 - We can add an exciting synapse to some input that is active
* 1.2 - We can remove an inhibiting synapse to some input that is active
* 1.3 - We can decrease the threshold

To correct case 2, options are:

* 2.1 - We can add an inhibiting synapse to an active input
* 2.2 - We can remove an exciting synapse to some input that is active
* 2.3 - We can increase the threshold

And this is for this local layer. We can do a sort of backpropagation to the previous layer. That is:

* 1.4 - We can tell the neuron to which we have an exciting synapse, but which is not firing, that it is excited not enough
* 1.5 - We can tell the neuron to which we have an inhibiting synapse and it is firing, that it is excited too much

Similar in case 2:

* 2.4 - We can select the neuron to which we have an inhibiting synapse, but which is not firing, that it is excited not enough
* 2.5 - We can select an active neuron to which we have an exciting synapse that it is excited too much

So to summarize, we have three classes of correcting our two kinds of errors ("too much", "not enough"):

* Adjust our synapses
* Adjust our threshold
* Tell neurons that provide inputs to our existing synapses that they made an error ("too much" or "not enough")

As for batch learning, yes there is a way to do it. Instead of performing learning steps for every example and randomizing here and there synapses to add or remove etc., we can compute probabilities of synapse add/delete operations and aggregate them across many examples. Also in a single example, the downpropagated signal "too much"/"not enough" can be aggregated in multiple pathways, because the neuron in layer X can get feedback from multiple neurons in layer X+1.
