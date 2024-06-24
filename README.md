    poetry install
    poetry shell

    pytest

    mnist_get_data.sh
    mnist_preview

    python -m bnn.mnist

## Idea

i'd like to talk about my idea for binary, sparse neural nets that learn in a non gradient-descent manner, but through correlation of inputs/outputs

So the basic idea is that we have, lets say, "binary neurons". Similarly how it works in classic neuarl nets, and drawing inspiration from biological nets, each neuron has many inputs and one output. And now the binary aspect kick in - each input/synapse either "is" or "is-not". So the weight is 1 or 0. But also the inputs are divided into tw groups. We have exciting inputs and inhibiting inputs. So in classic net it correlates with a weright -1, 0 or 1. Then neurn has an activation threshold which is an integer. So this is like a step function aka very sharp sigmoid. When the sum of inputs is below threshold the neron does not fire (output 0), else it fires (output 1). So this is binary part of the idea. Now for the sprse idea, when we describe to which outputs from previous layer the neuron is connected we dont use dense list of weights, but we store only pointers. So we have one list of indices that excite the neuron and a list that inhibits the neuron.

so for the training I envision this like that: neuron can be wrong in two distinct ways
1. it is inactive but should be active -- aka "not enough excitation"
2. it is active but should be inactive -- aka "too much excitation"
This all assumes we are learning on training data - input and desired output pairs.
To correct case 1. we have options:
1.1 - we can add exciting synapse to some input that is active
1.2 - we can remove inhibiting synapse to some input that is active
1.3 - we can decrease the threshold
To correct case 2. options are:
2.1 - we can add inhibiting synapse to active input
2.2 - we can remove exciting synapse to some input that is active
2.3 we can increase the treshold
And this is for this local layer. We can do sort of backpropagation to previous layer. that is:
1.4 - we can tell the neuron to which we have exciting synapse, but wchich is not firing, that is is excited not enough
1.5 - we can tell the neurn to which we have inhibiting synapse and it is firing, that it is excited too much
similar in case 2:
2.4 - we can select the neuron to which we have inhibiing synapse, but which is not firing that it is excited not enough
2.5 - we can select active neuron to whcih we have exciting sinapse that is is excited too much

So to summarize we have three classes of correcting our two kinds of errors ("too much", "not enoguh"):
* adjust aour syapses
* adjust our treshold
* tell neurons that provides inputs to our existing synapses that they made an error ("too much" or "not enough")
