    poetry install
    poetry shell

    pytest

    mnist_get_data.sh
    mnist_preview

    python -m bnn.mnist

    python -m benchmarks.run                 # all benchmarks, ~20 min on one core
    python -m benchmarks.run parity breast   # a subset

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

## CounterDense: what made it learn

`bnn/dense.py` implements the idea above literally.
`bnn/counter.py` keeps the same neuron
(binary in/out, ternary synapses, integer threshold, "too much"/"not enough" flowing down)
but changes how it decides.
Roughly in order of impact:

1. **Integer counter under every synapse.**
   The synapse exists only while `|counter| >= connect_level`, and errors move the counter by one,
   so synapses change only after consistent evidence.
   `Dense` decided from one batch and forgot, so learning faster made it worse.
2. **Inactive inputs learn too**, in the opposite direction (a perceptron on ±1 inputs).
   Otherwise a neuron can become more eager but never more selective.
3. **Margin-aware blame.**
   A neuron `n` flips away from correct blames its inputs only if `n <= backprop_margin`,
   spreading `n` expected flips over the inputs that could provide them,
   minus the outputs such a flip would break.
4. **Minimal disturbance** (`accept_distance=0` in hidden layers):
   only neurons one input away from flipping accept a request.
   Together with dense initial wiring, this is what made parity learnable.
5. **Output margin** (`stabilize_margin`): correct but fragile outputs get reinforced.
6. **Classes picked by margin**, not argmax over binary outputs (ties went to class 0).

The counter scale is a trade-off:
narrow (`connect_level=2`) helps hidden layers learn parity,
wide (`connect_level=8`) keeps 784-pixel inputs from oscillating.
Benchmarks set it per dataset.
Homeostatic thresholds, sparse potential pools and per-update step caps were tried and didn't help.
Open problems are in [TODO.md](TODO.md).

## Benchmark results

`python -m benchmarks.run`.
Inputs binarized: pixels > 128, one-hot categoricals, thermometer-coded numbers.
Stochastic models are averaged over 3 seeds; settings per dataset are in `benchmarks/run.py`.
Parity is scored on the full truth table.

| Dataset       | Majority | LogReg | DTree | RForest | Tsetlin | Dense | Counter |
|---------------|---------:|-------:|------:|--------:|--------:|------:|--------:|
| Fashion-MNIST | 0.100 | 0.789 | 0.722 | 0.831 | 0.674¹ | 0.185 | 0.655 |
| Mushroom      | 0.518 | 0.999 | 1.000 | 1.000 | 0.997 | 0.953 | 0.996 |
| Adult Income  | 0.761 | 0.851 | 0.804 | 0.828 | 0.831 | 0.761 | 0.819 |
| Breast Cancer | 0.632 | 0.921 | 0.947 | 0.947 | 0.921 | 0.561 | 0.924 |
| Parity-4      | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 0.979 |
| Parity-8      | 0.500 | 0.500 | 1.000 | 1.000 | 0.510 | 0.500 | 0.645 |
| Parity-16     | 0.500 | 0.500 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 |

¹ trained on 5000 samples to keep the run short.
Counter uses no hidden layer on Fashion-MNIST, Adult and Breast Cancer, where one still hurts.
It trains 2-13x faster than Dense.

## Biological parallels

* Counter under a binary synapse ≈ synaptic consolidation; HTM calls it permanence.
* Updates use only the neuron's inputs, its own margin and a signal from above ≈ three-factor learning rules.
* Minimal disturbance: only neurons near threshold change, which is where a real neuron is sensitive anyway.

## Related work

* Tsetlin Machine, Granmo 2018, [arXiv:1804.01508](https://arxiv.org/abs/1804.01508) -- baseline in `bnn/tsetlin.py`.
* SBPI: Baldassi et al., *Efficient supervised learning in networks with binary synapses*, PNAS 2007 -- hidden discrete states, margin-gated updates.
* MADALINE Rule II: Winter & Widrow 1988 -- minimal disturbance in multilayer threshold nets.
* HTM spatial pooler: Cui, Ahmad, Hawkins 2017 -- permanences, sparse binary activity.
* Friesen & Domingos, *Deep Learning as a Mixed Convex-Combinatorial Optimization Problem*, ICLR 2018 -- hidden binary targets as a combinatorial problem.
