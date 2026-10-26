# TODO

## CounterDense

* Depth. The pressure rule trains plain chains on parity-8 at depth 1-8 (README). Open:
  * Parity-16 with 3×64 learned and fell back every few epochs (raw 0.92, 0.52, 0.86, 0.50 ..., consolidated 0.5);
    the ratchet holds it at 0.97 (README). What falls back is the input layer: at 0.96 accuracy it still changes
    11% of its code bits every 128 updates, and freezing it alone at a good state keeps 0.96-0.97.
    A slower input layer (connect_level ×2, ×4) never leaves chance: its churn is the search.
    Also stops the fall, not in the code: hidden votes ×(running error rate) (0.94 without the ratchet),
    but Mushroom then freezes before its last errors are fixed (1.000 → 0.995-0.998, with a floor on the factor too).
    Didn't help: stabilizing votes in hidden layers, consolidating every 64 or 512 updates (stays at chance),
    equalizing how often units vote. A ratchet on the input layer alone is untested.
  * Parity-8: depth 8 is seed-sensitive (8×128 gave 1.00 in 3 seeds, 0.84 in 3 others), 12×64 doesn't break through (0.60).
    12×64 loses no information (every layer's code stays injective on the 256 inputs).
    Its middle layers' plasticity blocks it: initialized as exact identities and frozen, 12×64 learns in ~20 epochs.
    A copy of an input sits at distance 0 on every sample, so minimal disturbance makes it the most plastic unit;
    once one wire breaks, homeostasis turns the copies above it into constants.
    Slower identities: synapses ×4, ×16 no; synapses and thresholds ×4 1 seed of 2.
    Copies with a margin (k redundant wires, threshold k/2) are untried.
  * Untested: annealing the vote probability.
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
* Hidden layers on pixels beat a single layer only when learned slowly (README, "Wider and deeper"):
  4096 units at `connect_level=256` 0.855, one layer 0.847, random forest 0.875.
  The speed is set per benchmark (`hidden_kwargs`): parity wants the opposite (hidden counters ×4: parity-8 1.00 → 0.68).
  It should follow from whether random features are already good, not be hand-set.
  * At the default speed the units collapse onto each other (mean |correlation| 0.16 → 0.31,
    5% have a twin at > 0.93): units with similar readout columns get the same pressure.
    A hidden counter on pixels drifts ~17 per epoch (rms), so `connect_level` 16 leaves the random features within an epoch.
  * 1024 units, 1 seed each: frozen 0.818, default 0.791, connect_level 256 0.834, 4096 0.830 (0.6% of synapses change:
    the thresholds do most of it). Recycling near-twins every epoch 0.805.
    Exact flip gains as the hidden pressure (the readout column is all that changes): 256 / 1024 units
    0.783 → 0.803 / 0.791 → 0.813, nothing on Adult and parity; letting only the k best units vote per sample: no change.
  * Two slow layers do worse (4096+1024 0.821, 4096+4096 0.841), 16384 slow units 0.865 (frozen 0.863),
    Adult with 1024 slow units 0.847 (none 0.845). 1 seed each.
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
* Exact moves below the last hidden layer: flipping a unit and propagating the flip up
  gives no signal (parity-8 3×32 0.61, parity-16 3×64 0.50): a single flip is almost always absorbed above.
  Something between that and the surrogate (the flip's effect on the next layer's margins) is untested.
  Full-batch exact descent on a hidden layer predicts each unit's gain alone and overestimates the joint effect ×5.
* Function-preserving moves find little slack in trained nets: output synapses whose removal changes neither
  training errors nor loss (~6% on Adult, none on Breast Cancer; no effect on test), unused hidden units (re-randomized:
  no change). One-hot groups have an exact symmetry (all the group's weights +1 and the threshold +1),
  and the ±1 votes make whole groups drift, but re-centering them changes nothing (Adult 0.844).
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
