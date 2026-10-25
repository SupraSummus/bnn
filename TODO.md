# TODO

## CounterDense

* Hidden layers on wide inputs hurt.
  On Fashion-MNIST a neuron with hundreds of synapses is rarely one input from its threshold,
  so minimal disturbance almost never lets it learn.
  Try blame/acceptance scaled to fan-in, or local (convolutional) wiring.
* On Adult Income (76% majority) a hidden layer collapses to always answering the majority class.
* Parity-4 is fully solved in about 80% of seeds, parity-8 in some, parity-16 never.
* Batches sum counter updates, which defeats the hysteresis on wide inputs.
* No single counter scale works everywhere (see README).

## Code

* `Dense` and the old `BN`/`ConvBNN` (`bnn/__init__.py`, used by `bnn/mnist.py`) are now dominated by `CounterDense`.
  Decide whether to keep them as history or remove.
* PR #1 (predictive coding) builds on `Dense`; port to `CounterDense` or close.
* `benchmarks/plot_metaparams.py` still sweeps `Dense` only.
* Binarization quantiles (`thermometer`, median split) are fit on train+test; harmless (label-free) but not clean.
