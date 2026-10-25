# TODO

## CounterDense

* Hidden layers on wide inputs still don't beat a single layer.
  With consolidation, 3 seeds: Fashion-MNIST with 256 hidden 0.69 (one layer 0.844),
  Adult with 32 hidden still collapses to the majority class, Breast Cancer 0.86 vs 0.91.
  Tried (10k-sample subset, 1-bit pixels, no consolidation, unless noted):
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
  * Recruiting units that fix the current errors, then freezing them (Upstart-like, Frean 1990),
    each new unit also seeing earlier ones (as in Cascade-Correlation):
    learns parity-4 in some seeds (mean 0.89; without seeing earlier units 0.5).
    Full data, 2-bit pixels, consolidated, 1 seed: Fashion-MNIST 0.840 → 0.845-0.850 after 10 rounds,
    Adult 0.839 → ~0.84; training accuracy rises more (0.88 on Fashion-MNIST).
    Not worth its code yet.
* Consolidation happens once, after training.
  An online version (a slow trace that takes over from the fast state) is untested.
* `threshold_level=4` collapses an output layer reading a wide sparse code
  (1024 units, 32 active: 0.1-0.5 accuracy, vs 0.72 with 1),
  while parity wants 4. How it should depend on fan-in is open.
* Parity-4 is fully solved in about 80% of seeds, parity-8 in some (0.84 mean with consolidation), parity-16 never.
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
