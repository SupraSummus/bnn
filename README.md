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
9. **Ratchet** (`counter.Ratchet`): every 128 updates, training accuracy on a fixed sample of 2048;
   if it fell more than 0.25 below the best seen, the net goes back to that state.
   Deep nets learn by search, and the plasticity that finds a solution also leaves it:
   parity-16 with 3×64 hidden reached 0.9 and fell back to 0.5 every few epochs (0.51 after consolidation);
   with the ratchet 0.96-0.97 (3 seeds). The tolerance is above the raw state's noise (up to 0.14 on Fashion-MNIST),
   so no benchmark in the table rolls back and none changes.
10. **Exact polish of the output layer** (`bnn/exact.py`, see "Exact moves" below):
   up to 32 steps of coordinate descent whose every move is evaluated exactly on the training set
   and never adds a training error. +0.005 on Fashion-MNIST, +0.009 on Breast Cancer, +0.001 on Adult (3 seeds each).

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
| Fashion-MNIST | 0.100 | 0.847 | 0.788 | 0.875 | 0.709¹ | 0.159 | 0.871 |
| Mushroom      | 0.518 | 0.999 | 1.000 | 1.000 | 0.997 | 0.953 | 1.000 |
| Adult Income  | 0.761 | 0.851 | 0.804 | 0.828 | 0.831 | 0.761 | 0.845 |
| Breast Cancer | 0.632 | 0.921 | 0.947 | 0.947 | 0.921 | 0.561 | 0.924 |
| Parity-4      | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 |
| Parity-8      | 0.500 | 0.500 | 1.000 | 1.000 | 0.510 | 0.500 | 1.000 |
| Parity-16     | 0.500 | 0.500 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 |

¹ trained on 5000 samples to keep the run short.
Counter uses no hidden layer on Adult and Breast Cancer,
where 1-3 hidden layers of 32 give the same (0.845-0.846, 0.906-0.915, before polishing).
On Fashion-MNIST it uses 4096 hidden units whose votes are gated like a gradient's ("The ladder"); one layer gets 0.847.
Over 10 seeds parity-4, -8 and -16 are solved every time, Mushroom in 9 (0.9999)
(before the pressure rule: parity-8 0.90 ± 0.05, parity-16 0.58 ± 0.04, Mushroom 0.995).

## Exact moves

A network of threshold units with ternary synapses is discrete, so the effect of a change
can be counted instead of linearized. Moving synapse `(o, i)` by one step shifts output `o`'s margin
by ±1 on exactly the samples where input `i` is on; a threshold is the weight of an input that is always on.
For every sample we know what ±1 on each margin does to its loss and to whether it is classified right
(`exact.move_terms`), so one matrix product gives the exact change of the training errors and of the loss
for every single move of a layer at once (`exact.move_gains`).
A gradient gives the slope; this gives the value of each discrete step.

`exact.polish_readout` does coordinate descent with these gains on the output layer after training.
A move is allowed only if it adds no training error; among those, the one lowering a multiclass hinge
(on the gaps `m[label] - m[other]`) most is taken. Moves made together interact,
so each step is checked exactly and cut in half until it helps: training errors never go up.
On Fashion-MNIST with one layer the gain on test comes within ~25 steps (0.842 → 0.847); ~300 steps get no further.
Without the error guard the hinge alone *lowered* accuracy on Adult with a hidden layer (0.845 → 0.83).
The online rule doesn't get there by itself: multiclass output pressure
(the right class up, every class within `hinge` of it down) changed nothing.
What the exact view did for hidden layers is in TODO.md.

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
  Learning that layer at the speed of the output layer doesn't help (pressure rule: 0.806 / 0.806, 3 seeds):
  the units collapse onto each other, and even *training* accuracy drops (1024 units: 0.853 frozen, 0.799 learned).
  Learned slowly it helps: with `connect_level=256` (init ±384) instead of 16 the hidden layer moves 4% of its
  synapses away from the random start, and 4096 units give 0.850, polished 0.855 (3 seeds; frozen and polished 0.846).
  Parity is the opposite: its random features are useless, and slower hidden counters stop it learning.
  ~55 s per seed, polishing the 4096-wide readout ~20.
  Superseded by the ladder's gating (below): 4096 units at the default speed 0.871, slowly 0.864.
* **Depth.** Parity-8, plain chains, `benchmarks.bnn_model.train_and_evaluate` defaults, 3 seeds:
  solved at depth 1 to 6 with widths 64 to 256, 0.97-1.00 at depth 8 with widths 32 to 128
  (seed-sensitive: another random stream gave 8×128 a seed at 0.66), 12×64 0.60.
  Deep nets sit at chance for a while and then break through within ~100 epochs;
  a linear probe on the first hidden layer shows the feature ~100 epochs before accuracy moves.
  Before the pressure rule 3 hidden layers didn't learn parity at all.
  Open problems and what didn't work are in TODO.md.

## Sparse: N pointers per neuron

The idea above stores synapses as pointers; `CounterDense` grew dense (about half of all pairs connected).
`bnn/sparse.py` (`SparseCounter`) goes back: every neuron has a fixed number N of slots,
each a pointer to an input with a counter under it, and the rule is CounterDense's on the slots
(with every input in a slot it is CounterDense bit for bit, `bnn/sparse_tests.py`).
Work and memory per neuron are O(N), not O(inputs).
With `rewire` > 0, a slot holding no synapse can move: when the neuron votes, it asks for a synapse
on an input that is on in that sample (cases 1.1 and 2.1 of the idea), at ±connect_level.

Fashion-MNIST, a hidden layer of random slots and a CounterDense readout, 3 epochs
(`python -m benchmarks.sparse 4096 8 fixed dense 0 3`), measured before "The ladder" (below):

* **Random and fixed is best.** Pointers and signs random, counters at ±2048 against connect_level 256
  (~1% of synapses still fade in 3 epochs), thresholds 0, only the readout learns:
  4096 units with N = 8 give 0.856 ± 0.002 (3 seeds), N = 16 0.853 ± 0.002;
  1 seed: N = 1, 2, 4, 32 0.61, 0.77, 0.842, 0.853; 16384 units with N = 8 0.861.
  The slowly learned dense layer of 4096 ("Wider and deeper") has 0.855 with ~100× the hidden synapses.
  That is the fruit fly's olfactory circuit (Dasgupta et al. 2017): a wide sparse random expansion and a learned readout.
* **Counting pointers as parameters:** 33k hidden pointers + 28k readout synapses for 0.856
  (logistic regression: 15.7k floats, 0.843; MLP with 256 hidden: 400k floats, 0.872).
  In bits (12 per pointer, 2 per possible readout synapse) ~58 KB, logistic regression in float32 63 KB.
* **Learning the sparse hidden layer hurts**, as it did dense: counters at the default speed 0.840,
  slower (connect_level 16, threshold_level 64) 0.839, rewiring 0.817-0.840 (p 0.01-0.1; at 0.1 every slot moves ~10 times in 3 epochs),
  thresholds moving with fixed synapses 0.846 (1 seed each). Test accuracy during training drops to 0.4-0.6:
  the readout chases a moving code.
* **Thresholds above 0 give the fly's sparse code:** N = 16 at threshold 4 has 7% of units on and gets 0.849
  with half the readout synapses (15k); threshold 8 leaves almost nothing on (0.62). 1 seed.
* **A sparse readout needs rewiring:** N = 256 slots per class 0.757 without, 0.796 with (p = 0.1),
  N = 1024 0.836 (dense, consolidated the same way: 0.853; the exact polish needs a dense layer). 1 seed each.

## Convolution

`bnn/conv.py`: `ConvCounter` is a CounterDense applied to every patch of an image (valid, with a stride).
Patches are rows, so a filter's counters collect the votes of every position, and the pressure on a pixel
is the sum over the patches that contain it. A filter sees positions × batch rows per update,
so a row votes with probability |pressure| / positions: without it a counter crosses ±2048 in one update.
`OrPool` is binary max pooling (on iff any unit of the block is on; "not enough" goes to the whole block,
"too much" to the units that are on).

Fashion-MNIST, 3 epochs, CounterDense readout (`python -m benchmarks.conv 64x5/1,p2 fixed 0 3 '{"threshold": 2}'`),
measured before "The ladder" (below):

* **Random, fixed filters** (signs random, about half the synapses present, a fixed threshold) and one pooling:
  64 filters 5×5, stride 1, pool 2: 0.872 ± 0.003 (3 seeds), from 1.6k filter synapses (and 63k in the readout);
  128 filters 0.877, 64 filters 3×3 0.873 (1 seed). The best dense net was then 0.856 (now 0.871).
  The threshold matters: a blank patch sums to 0, so at threshold 0 the background fires everywhere
  (32×5/2 without pooling: 0.844 at 0, 0.848 at 1, 0.852 at 3). Pooling adds ~0.02 (32×5/2 0.847, 32×5/1 + pool 0.852).
* **Deeper random stacks lose:** a second random conv layer (64×3×3 over 64 channels) after pooling 0.815-0.822
  (thresholds 4 and 8), without the second pooling and with 128 filters 0.847. 1 seed each.
* **Learned filters lose to random ones:** 32×5/1 + pool, learned (accept_distance 0, homeostasis, as hidden layers) 0.816,
  random 0.852 (1 seed): the same as dense and sparse hidden layers.
  Two conv layers with pooling, learned: 2×32 filters 0.776 (random 0.769), 2×64 0.774 (random 0.822). 1 seed each.

## The ladder

Why did hidden layers learn less than random ones? `benchmarks/ladder.py` assembles training from parts,
each in a rule version (CounterDense) and a gradient version (straight-through estimator, Adam), on the same discrete net.
One hidden layer of 1024 on Fashion-MNIST, hidden counters at the default speed (connect_level 16), 5 epochs, batch 32.
All rule parts: 0.789 ± 0.004 (3 seeds; CounterDense itself on the same setup 0.791, 0.781).
All gradient parts: 0.852 (2 seeds). The hidden layer frozen at its random start: 0.816 (2 seeds).

The rule with one part switched to its gradient version (seed 0, the rule 0.794):

| Part | Rule → gradient | Acc |
|---|---|---:|
| gating | hidden: flips at distance 0 → any vote within the window; output: flips, and stabilizing below 2 → every output with pressure | 0.844 |
| votes | binary, with probability \|signal\| → the signal as a float step | 0.809 |
| pressure down | outputs in the window, 1 / (distance + 1) each, ÷ √reach → the plain sum through the synapses | 0.808 |
| votes, with Adam | → float steps under Adam | 0.807 |
| loss | hinge → softmax cross-entropy | 0.792 |
| homeostasis | on → off | 0.762 |
| inactive inputs | move the other way → don't move | 0.760 |
| consolidation | on → off | 0.738 |

* **Minimal disturbance held hidden layers back.** Gradient gating gives 0.846 ± 0.002 (3 seeds);
  in the hidden layer alone 0.836, in the output alone 0.795. Hidden flips up to distance 2 / 8 / 40: 0.808 / 0.841 / 0.837
  (at 40 test accuracy during training dips to 0.32); any hidden vote within distance 8: 0.829.
* **With gradient gating and gradient pressure down, the rest of the rule beats gradient descent:**
  integer counters, binary votes, ±1 inputs, homeostasis and consolidation give 0.869 ± 0.003 (3 seeds),
  Adam on the same net 0.852; with float steps instead of votes 0.872 ± 0.001. With hidden flips up to 8 instead: 0.854 ± 0.001.
  The learned layer is now well above the random one (0.816); the old rule was below it.
* The parts gradient descent lacks are worth keeping: without the ±1 inputs, homeostasis or consolidation the rule loses 0.03-0.06.
* Going the other way (gradient descent, then rule parts one by one) needs a step size per rung once Adam is gone
  (cross-entropy with plain steps: 0.832 at 0.3, 0.797 at 1, diverging at 3; with the hinge, 1 diverges), so it is left out.

**In CounterDense** (`surrogate`, `normalize_reach`; `accept_distance` and `stabilize_margin` take `"window"`),
3 seeds each, the suite's ratchet and polish, so the numbers sit above the ladder's:

| Benchmark | Rule | Output passes down whole | + hidden gated within the window |
|---|---:|---:|---:|
| Fashion-MNIST, 1024 hidden | 0.820 | 0.832 | 0.862 |
| Fashion-MNIST, 4096 hidden | –² | 0.838 | **0.871** |
| Adult, 32 hidden | 0.846 | 0.845 | 0.846 |
| Breast Cancer, 32 hidden | 0.915 | 0.918 | 0.930 |
| Mushroom, 32 hidden | 1.000 | 1.000 | 1.000 |
| Parity-8, 128 / 4×64 | 1.000 / 1.000 | 1.000 / 1.000 | 0.934 / 0.729 |
| Parity-16, 256 / 3×64 | 1.000 / 0.979 | 1.000 / 0.986 | 0.502 / 0.500 |

² not measured at the default speed; the slowly learned layer the suite used had 0.855.

* **The output passing its pressure down whole** (`surrogate="none"`, `normalize_reach=False`) helps or is neutral everywhere,
  and is now the output layer's default (`build_counter_network`). Its margins are the loss's input, not a step to differentiate.
* **Gating hidden votes within the window helps on pixels and breaks parity**, shallow or deep. Narrower doesn't reconcile them:
  flips only, up to distance 1 / 2 / 4 / the window, give Fashion-MNIST (1024) 0.837 / 0.844 / 0.850 / 0.848,
  parity-16 3×64 0.500 at every one, parity-8 4×64 0.68-0.69, Adult 0.844 / 0.841 / 0.840 / 0.834.
  So it is set per benchmark (`hidden_kwargs`), for Fashion-MNIST only, as the hidden speed was;
  the pixels' gain comes from votes for units already on the right side too (the window's flips alone: 0.848, all votes: 0.862).
* Hidden layers passing down without a surrogate (`"box"`, no √reach) change nothing here (one hidden layer;
  with depth only parity, which the gating already breaks).

## Biological parallels

* Counter under a binary synapse ≈ HTM's permanence: a synapse changes only after repeated evidence.
* Fast learning followed by keeping the average state ≈ synaptic consolidation.
* Updates use only the neuron's inputs, its own margin and a signal from above ≈ three-factor learning rules.
* Minimal disturbance: only neurons near threshold change, which is where a real neuron is sensitive anyway.

## Related work

* Tsetlin Machine, Granmo 2018, [arXiv:1804.01508](https://arxiv.org/abs/1804.01508) -- baseline in `bnn/tsetlin.py`.
* Dasgupta, Stevens, Navlakha, *A neural algorithm for a fundamental computing problem*, Science 2017 -- the fly's sparse random expansion.
* SET: Mocanu et al., *Scalable training of artificial neural networks with adaptive sparse connectivity*, Nature Communications 2018 -- fixed synapse budget, weak synapses regrown elsewhere.
* SBPI: Baldassi et al., *Efficient supervised learning in networks with binary synapses*, PNAS 2007 -- hidden discrete states, margin-gated updates.
* MADALINE Rule II: Winter & Widrow 1988 -- minimal disturbance in multilayer threshold nets.
* Straight-through estimator: Bengio, Léonard, Courville 2013, [arXiv:1308.3432](https://arxiv.org/abs/1308.3432) -- a surrogate derivative for a step.
* Schoenholz et al., *Deep Information Propagation*, ICLR 2017 -- deep nets train only near the edge between a vanishing and an exploding signal.
* HTM spatial pooler: Cui, Ahmad, Hawkins 2017 -- permanences, sparse binary activity.
* Friesen & Domingos, *Deep Learning as a Mixed Convex-Combinatorial Optimization Problem*, ICLR 2018 -- hidden binary targets as a combinatorial problem.
