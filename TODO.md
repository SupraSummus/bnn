# TODO

## CounterDense

* Depth. The pressure rule trains plain chains on parity-8 at depth 1-8 (README). Open:
  * Parity-16 with 3×64 learns and falls back (raw accuracy 0.92, 0.52, 0.86, 0.50 ... every few epochs,
    2048 updates an epoch), and the average over the last epoch is a 0.5 net. 2×256 is solved.
    Not the homeostasis run length (256, 2048 and 16384 behave the same).
  * Parity-8: depth 8 is seed-sensitive (8×128 gave 1.00 in 3 seeds, 0.84 in 3 others), 12×64 doesn't break through (0.60).
  * Both look like a constant learning rate over many updates.
    Untested: annealing the vote probability, online consolidation (below).
  * Diagnostics that helped: accepted votes per sample, layer by layer (the signal's branching ratio, should stay near 1);
    correlation of a unit's pressure with the effect of actually flipping it;
    a linear probe on each layer's code, which shows a feature ~100 epochs before accuracy moves.
  * Tried and dropped:
    a per-sample flip limit (the old signal vanished, it didn't explode);
    RMS-normalized pressure (flat branching, but no learning past 4 layers), narrower windows, voting across the window;
    exact flip tests, Madaline Rule II style (2 layers only, 20× slower);
    against collapse: protecting fragile hidden units, lower vote rates, bigger counters (the hinge works);
    homeostasis toward 50% activity, re-centering only at init, moving thresholds on requests too far to accept;
    an output reading every hidden layer (solves wide parity by shortcut: any one layer can be cut).
* Hidden layers on pixels still don't beat a single layer:
  Fashion-MNIST 0.806 with 1024 or 4096 hidden units (one layer 0.84; a frozen random layer of 4096 0.85).
  What the hidden layer learns there is worth less than its random initialization.
  Tried before the pressure rule (10k-sample subset, 1-bit pixels, no consolidation, unless noted):
  * Useful diagnostic: freeze a trained output layer and train only the hidden one.
    With the old blame, training accuracy *drops* (0.71 → 0.35):
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
  while parity wants 4. How it should depend on fan-in is open
  (connect_level, window and hinge now follow sqrt(fan_in); the threshold step doesn't).
* Batches sum counter updates, which defeats the hysteresis on wide inputs.
* A counter sitting at ±`connect_level` toggles its synapse on every vote, so there's no real hysteresis.
  Resetting the counter after a change (Bop) helps wide hidden layers but hurts parity-4 (9/10 → 3/10 seeds solved).
* Counter scale from fan-in (`0.7 * sqrt(fan_in)`) works on every benchmark but Fashion-MNIST,
  which keeps `threshold_level=1` (and its hand-set `connect_level=16`, the default gives the same).

## Code

* Kernels are single-threaded; benchmarks use the cores by running models side by side.
  For one big model, `margins` parallelizes over samples (prange) and the counter votes over outputs.
* `CounterDense` settings split by role: `accept_distance` and `homeostasis` are for hidden layers,
  `stabilize_margin` and `hinge` for the output (`build_counter_network` sets them). A role flag could set all four.
* `CounterDense.forward/backward/update` now serve only tests and mixing with other layer types;
  training goes through `counter.train_epoch`. If no other layer type comes back, the per-batch path could go.

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
