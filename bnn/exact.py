"""
Exact analysis of single moves in a ternary threshold layer.

A layer's output is discrete, so the effect of moving one synapse by one step
(or a threshold by one) on a training set is not a linearization, as a gradient
is, but can be counted exactly: the move shifts the margin of one output by +-1
on the samples where its input is on (on all samples for a threshold), and the
loss of every sample is a known function of its margins.

`polish_readout` uses this after training: coordinate descent on the output
layer that never increases the number of training errors and, among the moves
that keep it, takes the one that lowers a multiclass hinge loss the most.
"""

import numpy as np
from numba import njit

from .abstract import ArrayBool


@njit(cache=True)
def _is_correct(m, c):
    """np.argmax(m) == c: the first index wins ties."""
    for o in range(m.shape[0]):
        if m[o] > m[c] or (m[o] == m[c] and o < c):
            return False
    return True


@njit(cache=True)
def move_terms(margins, labels, hinge):
    """
    Per sample and output, what a margin moving by +-1 does to that sample.
    Errors count argmax mistakes (ties as np.argmax breaks them); the loss is a
    multiclass hinge, sum over wrong classes o of max(0, hinge - (m[label] - m[o])).

    Returns (errors, loss, terms) with terms of shape (batch, 4, outputs):
    change of the sample's error when m[b, o] goes up / down by one, then
    change of its loss when m[b, o] goes up / down by one.
    """
    batch, n_out = margins.shape
    terms = np.zeros((batch, 4, n_out), np.float32)
    errors = 0
    loss = 0
    m = np.empty(n_out, np.int64)
    for b in range(batch):
        c = labels[b]
        for o in range(n_out):
            m[o] = margins[b, o]
        right = _is_correct(m, c)
        errors += not right
        below = 0  # gaps under the hinge: raising m[c] lowers the loss by this
        at_most = 0  # gaps at or under: lowering m[c] raises it by this
        for o in range(n_out):
            if o == c:
                continue
            gap = m[c] - m[o]
            loss += max(0, hinge - gap)
            below += gap < hinge
            at_most += gap <= hinge
            terms[b, 2, o] = 1.0 if gap <= hinge else 0.0
            terms[b, 3, o] = -1.0 if gap < hinge else 0.0
        terms[b, 2, c] = -below
        terms[b, 3, c] = at_most
        for o in range(n_out):
            m[o] += 1
            terms[b, 0, o] = np.float32(right) - np.float32(_is_correct(m, c))
            m[o] -= 2
            terms[b, 1, o] = np.float32(right) - np.float32(_is_correct(m, c))
            m[o] += 1
    return errors, loss, terms


def move_gains(inputs: ArrayBool, margins, labels, hinge, chunk=8192):
    """
    Exact change of (errors, loss) on the whole set for every single move of a layer
    whose margins are `margins` on `inputs`. A threshold is the weight of an input that is always on,
    taken with the opposite sign: slot `inputs` (one past the last input) is "threshold - 1".
    Returns (errors, loss, error_gains, loss_gains), the gains of shape (2, inputs + 1, outputs):
    [0] a weight one up, [1] one down.
    Samples no move can affect are skipped, the rest go through a float32 matmul `chunk` rows at a time
    (sums of small integers, exact).
    """
    errors, loss, terms = move_terms(margins, labels, hinge)
    batch, _, n_out = terms.shape
    flat = terms.reshape(batch, 4 * n_out)
    active = np.flatnonzero(flat.any(axis=1))
    gains = np.zeros((inputs.shape[1] + 1, 4 * n_out), np.float32)
    for start in range(0, len(active), chunk):
        rows = active[start:start + chunk]
        gains[:-1] += inputs[rows].T.astype(np.float32) @ flat[rows]
    gains[-1] = flat.sum(axis=0)
    gains = gains.reshape(-1, 4, n_out).transpose(1, 0, 2)
    return errors, loss, gains[:2], gains[2:]


def polish_readout(layer, inputs: ArrayBool, labels, hinge=None, max_steps=1000):
    """
    Coordinate descent on a trained output layer, all moves exact on (inputs, labels):
    a move is allowed if it doesn't add a training error, and moves are ranked by how much
    they lower the multiclass hinge (an error removed outranks any loss).
    Several moves at once interact (on shared samples, and across outputs through the argmax),
    so a step takes the best `k` moves of every output, keeps them if they lowered (errors, loss),
    and otherwise falls back to the best half of them, down to the single best move, which is exact.
    `k` doubles after a step that kept all its moves and halves otherwise.
    Training errors never go up.

    Sets the layer's counters to +-connect_level on changed synapses and returns the number of steps.
    """
    if hinge is None:
        hinge = layer.hinge
    n_in = layer.input_features
    weights = layer.weights.astype(np.int64)
    thresholds = layer.thresholds.astype(np.int64)
    margins = layer.margins(inputs).astype(np.int64)

    def apply(moves, sign):
        for _, o, i, step in moves:
            step *= sign
            if i < n_in:
                weights[o, i] += step
                margins[:, o] += step * inputs[:, i]
            else:
                thresholds[o] -= step
                margins[:, o] += step

    big = 2.0 * hinge * margins.size  # one error outweighs any change of the loss
    k = 1
    steps = 0
    for steps in range(1, max_steps + 1):
        errors, loss, error_gains, loss_gains = move_gains(inputs, margins, labels, hinge)
        allowed = error_gains <= 0
        allowed[0, :n_in] &= weights.T < 1
        allowed[1, :n_in] &= weights.T > -1
        score = np.where(allowed, loss_gains + big * error_gains, np.inf)
        down = score.argmin(axis=0)  # per input and output, the better direction
        best = score.min(axis=0)
        moves = sorted(
            (best[i, o], o, i, 1 - 2 * down[i, o])
            for o, column in enumerate(np.argsort(best, axis=0, kind="stable")[:k].T)
            for i in column if best[i, o] < 0
        )
        if not moves:
            break
        apply(moves, 1)
        kept = len(moves)
        while kept > 1 and move_terms(margins, labels, hinge)[:2] >= (errors, loss):
            apply(moves[kept // 2:kept], -1)
            kept //= 2
        k = k * 2 if kept == len(moves) else max(1, k // 2)

    changed = weights != layer.weights
    layer.counters[changed] = (np.sign(weights[changed]) * layer.connect_level).astype(np.int16)
    layer.thresholds = thresholds.astype(np.int32)
    layer.threshold_counters[:] = 0
    layer._sync_weights()
    return steps
