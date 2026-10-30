# TODO

## CounterDense

* The ladder's gating helps hidden layers on pixels (Fashion-MNIST 0.855 → 0.871) and breaks parity
  even at flip distance 1, so it is set per benchmark. Why is now known (README, "Depth, and why parity needs
  minimal disturbance"): a vote's width must be below the scale on which the target changes along the margin,
  one step for parity. Something a unit measures itself should set it: whether its votes at neighbouring margins agree.
  Summed over a layer the disagreement doesn't show (votes at distance 0 and 1 agree in sign for 0.7-0.8 of counters
  on parity too); it lives in the few units missing one input, which are the ones that learn.
* Depth on dense Fashion-MNIST gives nothing, to gradient descent either (README).
  The conv net with a 2×2 readout is where depth pays (TODO, "Convolution").
* Gradient descent with the usual window-wide surrogate can't learn parity on this net (0.50 everywhere);
  gated at the threshold, like the rule, it gets 0.61-0.81. So the parity benchmarks test a narrow surrogate
  and full wiring (a unit must see every input) more than credit assignment in general.

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
* Before the ladder's gating, hidden layers on pixels beat a single layer only when learned slowly
  (units at the default speed collapsed onto each other: those with similar readout columns get the same pressure).
  Tried then, no or small gain: recycling near-twins, exact flip gains as the hidden pressure (+0.02),
  letting only the k best units vote, Bop counters, a fixed ±1 readout, Upstart-like recruiting, two slow layers
  (details in TODO.md before commit 98f3013). A diagnostic worth keeping: freeze a trained readout, train only the hidden layer,
  and see whether training accuracy goes up.
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

## SparseCounter

* Sparse hidden layers don't learn: random and fixed beats every way of learning them tried (README, "Sparse"),
  as dense ones did before the ladder's gating, which is untried here. Untried: rewiring only units the readout ignores
  (a unit with no readout synapse is free to move), rewiring by the unit's own pressure history instead of per vote,
  a slower rewire chance (< 0.01) with counters fixed.
* Sparse readout: N = 1024 of 4096 0.836, dense 0.853 (1 seed). The slots a class starts with are random;
  regrowth picks a random active unit, not one that correlates with the class.
* Only Fashion-MNIST so far. Parity needs learned features (random ones are useless there),
  so it is the test for learning sparse hidden layers. Deep sparse stacks are untested.
* `SparseCounter` trains through the per-batch path (`Sequence.forward/backward/update`), not one compiled epoch;
  4096 × 8 with a dense readout takes ~15 s per epoch.
* Its kernels repeat CounterDense's vote and pass-down loops over slots, as `bnn/conv.py`'s do over patches
  (the decisions are shared: `counter_kernels.may_vote`, `passes_down`). One slot-based rule
  with the dense layer as all slots, bit-packed, would remove the copies, if it can keep the dense speed.

## Convolution

* Learned filters lose to random ones (README, "Convolution"), and random second layers lose information.
  So the deeper convolutional nets that make CNNs strong don't work here yet. The ladder's gating is untried here.
  Rows vote at 1 / positions; slower filter counters, or votes only from the positions the readout uses, are untried.
* `ConvCounter` trains through the per-batch path (a 32×5 layer: ~2 min per epoch); random filters are computed once.
* `OrPool`'s pressure is a guess ("not enough" to the whole block); the alternative is the unit nearest its threshold.

## Code

* Kernels are single-threaded; benchmarks use the cores by running models side by side.
  For one big model, `margins` parallelizes over samples (prange) and the counter votes over outputs.
* `CounterDense` settings split by role: `accept_distance` and `homeostasis` are for hidden layers,
  `stabilize_margin`, `hinge`, `surrogate` and `normalize_reach` for the output (`build_counter_network` sets them).
  A role flag could set them all.

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
