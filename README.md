    poetry install
    poetry shell

    pytest

    mnist_get_data.sh
    mnist_preview

    python -m bnn.mnist

    python -m benchmarks.run                 # CounterDense on all datasets
    python -m benchmarks.run parity breast   # a subset
    python -m benchmarks.run --all           # with the reference models (slow)

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
(binary in/out, ternary synapses, integer threshold, "too much"/"not enough" flowing down as signed pressure)
but changes how it decides.
Roughly in order of impact:

1. **Integer counter under every synapse.**
   The synapse exists only while `|counter| >= connect_level`, and votes move the counter by one,
   so synapses change only after consistent evidence.
   `Dense` decided from one batch and forgot, so learning faster made it worse.
2. **Inactive inputs learn too**, in the opposite direction (a perceptron on ±1 inputs).
   Otherwise a neuron can become more eager but never more selective.
3. **Pressure flowing down.**
   Every output gets a signed pressure (> 0: wants more excitation),
   +1/-1 toward its target in the output layer, right outputs included: that is what protects them.
   A layer passes pressure to its inputs through the outputs closer than `window` to their threshold,
   each `1 / (distance + 1)` of it (a surrogate derivative of the step),
   summed per input and divided by the square root of how many such outputs feed it.
   Without the square root the signal grows ×3 per layer; the blame this replaced shrank it ×0.05.
   With it plain chains learn parity-8 at depth 1 to 8 (see "Wider and deeper").
4. **Minimal disturbance** (`accept_distance=0` in hidden layers):
   only neurons one input away from flipping take a flip.
   Together with dense initial wiring, this is what made parity learnable.
   Letting the whole window vote instead broke even 4 hidden layers.
5. **Output margin and hinge.** Correct but fragile outputs get reinforced (`stabilize_margin`),
   and outputs right by at least `hinge` (`sqrt(fan_in) / 3`) send no pressure.
   Without the hinge, samples the net already handles keep reshaping the layers below,
   and deep nets collapse after they break through (0.78 → 0.22 on parity-8).
6. **Homeostasis** (hidden layers): a unit whose output was the same on 256 samples in a row
   moves its threshold to the running mean of its input sum.
   Units reading another hidden layer drift far from threshold within a few epochs,
   and with minimal disturbance nothing ever brings them back: deep nets stay at chance without this.
   Pushing every unit toward 50% activity instead hurts everywhere.
7. **Classes picked by margin**, not argmax over binary outputs (ties went to class 0).
8. **Consolidation.**
   The state after the last update is a noisy sample:
   test accuracy on Fashion-MNIST swung by ±0.03 from epoch to epoch.
   So the last epoch is tallied, and each synapse keeps its average state (rounded to -1/0/1),
   each threshold its average.
   The result is still ternary: +0.04 on Fashion-MNIST, +0.02 on Adult, +0.19 on parity-8,
   -0.015 on Breast Cancer (455 samples).
   Each layer is averaged on its own, so in a deep net the averaged hidden code is one
   the output never saw (Adult with 3 hidden layers: 0.84 before, 0.76 after, in 1-2 seeds of 6).
   So the output layer is then refitted for one epoch on the consolidated code (`counter.refit_readout`).

Margins spread like `sqrt(fan_in)`, and so do the defaults of `connect_level` (`0.7 * sqrt(fan_in)`,
what Adult and Breast Cancer had been tuned to by hand), `window` and `hinge`.
Wide pixel inputs still want thresholds to move faster (`threshold_level=1`);
Fashion-MNIST keeps its hand-set `connect_level=16` (the default 28 gives the same 0.84).
Sparse potential pools and per-update step caps were tried and didn't help.
Open problems are in [TODO.md](TODO.md).

## Benchmark results

`python -m benchmarks.run --all` (without `--all` only the Counter column).
Inputs binarized: two bits per pixel (> 0, > 128), one-hot categoricals, thermometer-coded numbers.
Stochastic models are averaged over 3 seeds; settings per dataset are in `benchmarks/run.py`.
Parity is scored on the full truth table.

| Dataset       | Majority | LogReg | DTree | RForest | Tsetlin | Dense | Counter |
|---------------|---------:|-------:|------:|--------:|--------:|------:|--------:|
| Fashion-MNIST | 0.100 | 0.847 | 0.788 | 0.875 | 0.709¹ | 0.159 | 0.842 |
| Mushroom      | 0.518 | 0.999 | 1.000 | 1.000 | 0.997 | 0.953 | 1.000 |
| Adult Income  | 0.761 | 0.851 | 0.804 | 0.828 | 0.831 | 0.761 | 0.844 |
| Breast Cancer | 0.632 | 0.921 | 0.947 | 0.947 | 0.921 | 0.561 | 0.915 |
| Parity-4      | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 |
| Parity-8      | 0.500 | 0.500 | 1.000 | 1.000 | 0.510 | 0.500 | 1.000 |
| Parity-16     | 0.500 | 0.500 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 |

¹ trained on 5000 samples to keep the run short.
Counter uses no hidden layer on Fashion-MNIST, Adult and Breast Cancer.
On Adult and Breast Cancer 1-3 hidden layers of 32 now give the same (0.845-0.846, 0.906-0.915),
on Fashion-MNIST a hidden layer still hurts (below).
Over 10 seeds parity-4, -8 and -16 are solved every time, Mushroom in 9 (0.9999)
(before the pressure rule: parity-8 0.90 ± 0.05, parity-16 0.58 ± 0.04, Mushroom 0.995).

## Speed

`CounterDense` trains in numba kernels (`bnn/counter_kernels.py`);
`bnn/counter_reference.py` is the same rule in numpy, adding pressure in the kernels' order,
and the tests check the two agree bit for bit, deep nets and homeostasis included.

* Synapses and binary activity are packed into 64-bit words:
  a neuron's input sum is `popcount(x & exc) - popcount(x & inh)`.
* Backward looks only at outputs with nonzero pressure and touches whole input rows only for outputs
  that vote or pass pressure down (those within the window).
* A whole epoch is one compiled call (`counter.train_epoch`); per batch, the numpy glue cost more than the kernels.
* Consolidation tallies a synapse row only when the update kernel touches it
  (and the rest at the end), so the tallied epoch costs what any other does
  (Fashion-MNIST with 4096 hidden units: 13 s → 3.1 s).
* Each layer draws from its own xoshiro256+ stream, one number per output that may vote,
  so an epoch in one call and batch by batch give the same result.

## Wider and deeper

Seconds per epoch on Fashion-MNIST (60k samples, batch 32, one core):
no hidden layer 0.07, 1024 hidden 1.6, 4096 hidden 6.1.
Half of it is the hidden forward pass, linear in width × input words;
the rest is the hidden layer learning (each vote moves 1568 counters), which the old blame hardly did.
Evaluating 60k samples costs about an epoch (the full margin matrix is allocated).

* **Width on pixels is random features.** Fashion-MNIST, 5 epochs:
  a hidden layer frozen at its random initialization gives 0.82 / 0.85 / 0.868 with 1024 / 4096 / 16384 units
  (1 seed), beating one layer (0.84) and logistic regression (0.847).
  Learning that layer doesn't help: the blame rule gave 0.80 / 0.85, the pressure rule 0.806 / 0.806 (3 seeds).
* **Depth.** Parity-8, plain chains, `benchmarks.bnn_model.train_and_evaluate` defaults, 3 seeds:
  solved at depth 1 to 6 with widths 64 to 256, 0.97-1.00 at depth 8 with widths 32 to 128
  (seed-sensitive: another random stream gave 8×128 a seed at 0.66), 12×64 0.60.
  Deep nets sit at chance for a while and then break through within ~100 epochs;
  a linear probe on the first hidden layer shows the feature ~100 epochs before accuracy moves.
  Before the pressure rule 3 hidden layers didn't learn parity at all.
  Open problems and what didn't work are in TODO.md.

## Biological parallels

* Counter under a binary synapse ≈ HTM's permanence: a synapse changes only after repeated evidence.
* Fast learning followed by keeping the average state ≈ synaptic consolidation.
* Updates use only the neuron's inputs, its own margin and a signal from above ≈ three-factor learning rules.
* Minimal disturbance: only neurons near threshold change, which is where a real neuron is sensitive anyway.

## Related work

* Tsetlin Machine, Granmo 2018, [arXiv:1804.01508](https://arxiv.org/abs/1804.01508) -- baseline in `bnn/tsetlin.py`.
* SBPI: Baldassi et al., *Efficient supervised learning in networks with binary synapses*, PNAS 2007 -- hidden discrete states, margin-gated updates.
* MADALINE Rule II: Winter & Widrow 1988 -- minimal disturbance in multilayer threshold nets.
* Straight-through estimator: Bengio, Léonard, Courville 2013, [arXiv:1308.3432](https://arxiv.org/abs/1308.3432) -- a surrogate derivative for a step.
* Schoenholz et al., *Deep Information Propagation*, ICLR 2017 -- deep nets train only near the edge between a vanishing and an exploding signal.
* HTM spatial pooler: Cui, Ahmad, Hawkins 2017 -- permanences, sparse binary activity.
* Friesen & Domingos, *Deep Learning as a Mixed Convex-Combinatorial Optimization Problem*, ICLR 2018 -- hidden binary targets as a combinatorial problem.
