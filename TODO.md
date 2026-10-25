# TODO

## CounterDense

* Hidden layers on wide inputs still don't beat a single layer on Fashion-MNIST.
  Tried so far (10k-sample subset, 1-bit pixels, unless noted):
  * Useful diagnostic: freeze a trained output layer and train only the hidden one.
    With the current blame, training accuracy *drops* (0.71 → 0.35):
    the hidden signal is harmful there, not just rare.
    A flip requested for one sample moves synapses that shift the unit on thousands of others.
  * Exact flip gains (how the output loss changes if one hidden unit flips),
    updates on active inputs only,
    and synapses whose counter resets after each change (Bop, Helwegen et al. 2019)
    make that diagnostic go down (0.715 → 0.766 train).
    Trained jointly or alternating with the output: ~0.735 on full data vs ~0.72 for one layer, 6x slower.
    Doesn't learn parity.
  * Fixed ±1 readout with Tsetlin-style feedback allocation: same plateau.
    Fails parity -- all units of a class get the same target, nothing breaks the symmetry.
  * Recruiting units that fix current errors, then freezing them (Upstart-like, Frean 1990):
    train 0.72 → 0.85 over 30 rounds, test 0.72 → ~0.75.
    Most promising; parity needs the recursive version.
* The output layer oscillates: test accuracy swings ±0.03 between epochs,
  with any counter scale or with plasticity decaying over epochs.
  That hides hidden-layer gains smaller than this.
* `threshold_level=4` collapses an output layer reading a wide sparse code
  (1024 units, 32 active: 0.1-0.5 accuracy, vs 0.72 with 1),
  while parity wants 4. How it should depend on fan-in is open.
* On Adult Income (76% majority) a hidden layer collapses to always answering the majority class.
* Parity-4 is fully solved in about 80% of seeds, parity-8 in some, parity-16 never.
* Batches sum counter updates, which defeats the hysteresis on wide inputs.
* A counter sitting at ±`connect_level` toggles its synapse on every vote, so there's no real hysteresis.
  Resetting the counter after a change (Bop) helps wide hidden layers but hurts parity-4 (9/10 → 3/10 seeds solved).
* No single counter scale works everywhere (see README).

## Code

* `Dense` and the old `BN`/`ConvBNN` (`bnn/__init__.py`, used by `bnn/mnist.py`) are now dominated by `CounterDense`.
  Decide whether to keep them as history or remove.
* `Dense.update` can add a synapse that is already there.
  `forward` then counts it twice, `backward` blames it once.
  Pick one meaning (weight 2, or dedupe in `normalize_synapses`).
* `Dense` is sensitive to the summation order of its synapse potentials.
  `np.sum` adds batch rows in order; a matmul adds the same terms in another order
  and dropped Mushroom from 0.93 to 0.84 (8 seeds). Unexplained.
  That `np.sum` over a (batch, outputs, inputs) temporary is now most of Dense's time
  (Dense on Fashion-MNIST is the slowest benchmark task);
  a Python loop over batch rows gives bit-identical sums ~1.5x faster.
* PR #1 (predictive coding) builds on `Dense`; port to `CounterDense` or close.
* `benchmarks/plot_metaparams.py` still sweeps `Dense` only.
* `poetry.lock` predates scikit-learn, matplotlib and pandas in `pyproject.toml`; run `poetry lock`.
* Binarization quantiles (`thermometer`, median split) are fit on train+test; harmless (label-free) but not clean.
