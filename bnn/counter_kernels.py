"""
Compiled (numba) kernels for CounterDense.

Binary inputs and ternary weights are packed into 64-bit words, so a neuron's
input sum is popcount(x & exc) - popcount(x & inh) over I/64 words.
The backward pass looks only at outputs with nonzero pressure and touches a whole
input row only for outputs that vote or pass pressure down (those within the window).
Random numbers come from a per-layer xoshiro256+ state, one per output that may vote.
`train_epoch` runs a stack of layers for a whole epoch: per batch, the numpy glue
would cost more than the kernels.

`counter_reference.ReferenceCounterDense` is the same algorithm in plain numpy;
the tests check the two agree bit for bit.
"""

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic

WORD = 64

# per-layer arrays train_epoch takes, each as a list over layers, in this order
LAYER_STATE = (
    "counters", "counter_delta", "touched", "weights", "exc", "inh",
    "thresholds", "threshold_counters", "threshold_delta", "rng_state",
    "homeo_run", "homeo_last", "homeo_mean",
)
# rate of the running mean of its input sum a hidden unit keeps for homeostasis, per sample
HOMEOSTASIS_RATE = 1.0 / 64
# how an output passes pressure down (CounterDense.surrogate), by index
SURROGATES = ("triangle", "box", "none")
TRIANGLE, BOX, NONE = range(3)


@intrinsic
def _popcount(typingctx, x):
    def codegen(context, builder, signature, args):
        ctpop = builder.module.declare_intrinsic('llvm.ctpop', [ir.IntType(64)])
        return builder.call(ctpop, args)

    return types.int64(types.uint64), codegen


def n_words(n_bits):
    return (n_bits + WORD - 1) // WORD


@njit(cache=True)
def next_uniform(state):
    """xoshiro256+ step on a uint64[4] state: float64 uniform in [0, 1)."""
    s0, s1, s2, s3 = state[0], state[1], state[2], state[3]
    result = s0 + s3
    t = s1 << np.uint64(17)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = (s3 << np.uint64(45)) | (s3 >> np.uint64(19))
    state[0], state[1], state[2], state[3] = s0, s1, s2, s3
    return np.float64(result >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def uniforms(state, n):
    out = np.empty(n)
    for i in range(n):
        out[i] = next_uniform(state)
    return out


@njit(cache=True)
def pack_bits(x):
    """bool (rows, n) -> uint64 (rows, n_words(n)), bit i of a row in word i // 64."""
    rows, n = x.shape
    out = np.zeros((rows, (n + 63) >> 6), np.uint64)
    x8 = x.view(np.uint8)
    groups = n >> 3
    for r in range(rows):
        for g in range(groups):
            v = np.uint64(0)
            for k in range(8):
                v |= np.uint64(x8[r, 8 * g + k]) << np.uint64(8 * k)
            # moves byte k's low bit to bit 56 + k, without collisions
            byte = (v * np.uint64(0x0102040810204080)) >> np.uint64(56)
            out[r, g >> 3] |= byte << np.uint64((g & 7) * 8)
        for i in range(groups * 8, n):
            out[r, i >> 6] |= np.uint64(x8[r, i]) << np.uint64(i & 63)
    return out


@njit(cache=True)
def margins(x_bits, exc, inh, thresholds):
    """Input sum minus threshold for every (sample, output); fires iff >= 0."""
    batch, words = x_bits.shape
    n_out = exc.shape[0]
    out = np.empty((batch, n_out), np.int32)
    if words == 1:  # narrow inputs: a loop over outputs alone vectorizes
        for b in range(batch):
            xb = x_bits[b, 0]
            for o in range(n_out):
                out[b, o] = _popcount(xb & exc[o, 0]) - _popcount(xb & inh[o, 0]) - thresholds[o]
        return out
    for b in range(batch):
        for o in range(n_out):
            s = 0
            for w in range(words):
                s += _popcount(x_bits[b, w] & exc[o, w]) - _popcount(x_bits[b, w] & inh[o, w])
            out[b, o] = s - thresholds[o]
    return out


@njit(cache=True)
def output_pressure(margins, targets, hinge):
    """
    Pressure on the output layer: +1 where the target is on, -1 where it is off,
    0 where the output is right with at least `hinge` to spare (hinge loss).
    """
    batch, n_out = margins.shape
    q = np.empty((batch, n_out), np.float32)
    for b in range(batch):
        for o in range(n_out):
            m = margins[b, o]
            distance = m if m >= 0 else -m - 1
            right = (m >= 0) == targets[b, o]
            q[b, o] = 0.0 if right and distance >= hinge else (1.0 if targets[b, o] else -1.0)
    return q


@njit(cache=True)
def backward(
    x, margins, pressure, weights,
    window, accept_distance, stabilize_margin, surrogate, normalize_reach, blame_inputs,
    counter_delta, threshold_delta, touched, rng_state,
):
    """
    Accumulate counter and threshold votes; return the pressure on the inputs (no rows unless blame_inputs).
    pressure > 0 asks for more excitation. See ReferenceCounterDense.backward for the rule;
    float operations follow it term by term so both give the same numbers.
    """
    batch, n_in = x.shape
    n_out = margins.shape[1]
    q_in = np.zeros((batch if blame_inputs else 0, n_in), np.float32)
    reach = np.zeros(n_in, np.int64)
    for b in range(batch):
        if blame_inputs:
            reach[:] = 0
        for o in range(n_out):
            q = pressure[b, o]
            if q == 0:
                continue
            m = np.int64(margins[b, o])
            out = m >= 0
            distance = m if out else -m - 1
            flip = (q > 0) != out
            may_vote = distance <= accept_distance if flip else distance < stabilize_margin
            # one draw per output that may vote, in row-major order
            if may_vote and next_uniform(rng_state) < abs(np.float64(q)):
                vote = np.int32(1) if q > 0 else np.int32(-1)
                for i in range(n_in):
                    counter_delta[o, i] += vote * (2 * np.int32(x[b, i]) - 1)
                threshold_delta[o] -= vote
                touched[o] = True
            if blame_inputs and (surrogate == NONE or distance < window):
                c = np.float32(np.float64(q) / (distance + 1)) if surrogate == TRIANGLE else q
                for i in range(n_in):  # absent synapses add 0, as in the reference
                    w = weights[o, i]
                    q_in[b, i] += c * np.float32(w)
                    reach[i] += w != 0
        if blame_inputs and normalize_reach:
            for i in range(n_in):
                if reach[i] > 1:
                    q_in[b, i] /= np.float32(np.sqrt(np.float64(reach[i])))
    return q_in


@njit(cache=True)
def homeostasis(margins, thresholds, threshold_counters, run_length, run, last, mean):
    """
    Revive units that stopped varying: a unit whose output was the same on `run_length` samples
    in a row moves its threshold to the running mean of its input sum (HOMEOSTASIS_RATE per sample).
    `margins` were computed with the thresholds as they are on entry.
    """
    base = thresholds.copy()
    for b in range(margins.shape[0]):
        for o in range(margins.shape[1]):
            m = margins[b, o]
            out = m >= 0
            mean[o] += (np.float64(m) + base[o] - mean[o]) * HOMEOSTASIS_RATE
            if out == last[o]:
                run[o] += 1
            else:
                run[o] = 1
                last[o] = out
            if run[o] >= run_length:
                thresholds[o] = np.int32(np.floor(mean[o]))
                threshold_counters[o] = 0
                run[o] = 0


@njit(cache=True)
def update(
    counters, counter_delta, touched, counter_max, connect_level, weights, exc, inh,
    thresholds, threshold_counters, threshold_delta, threshold_level,
    tally_weights, tally_since, tally_count, tallying,
):
    """
    Apply and clear the votes accumulated in touched rows,
    flip the synapses whose counter crossed connect_level.
    When tallying, a touched row first adds its state over the updates since
    `tally_since[o]` up to `tally_count`, the number of updates tallied so far.
    """
    n_out, n_in = counters.shape
    for o in range(n_out):
        if touched[o]:
            touched[o] = False
            if tallying:
                unchanged = tally_count - tally_since[o]
                for i in range(n_in):
                    tally_weights[o, i] += weights[o, i] * unchanged
                tally_since[o] = tally_count
            flipped = False
            for i in range(n_in):
                c = min(max(np.int32(counters[o, i]) + counter_delta[o, i], -counter_max), counter_max)
                counter_delta[o, i] = 0
                counters[o, i] = c
                weight = np.int8(c >= connect_level) - np.int8(c <= -connect_level)
                flipped |= weight != weights[o, i]
                weights[o, i] = weight
            if flipped:
                exc[o] = pack_bits(weights[o:o + 1] > 0)[0]
                inh[o] = pack_bits(weights[o:o + 1] < 0)[0]

        t = threshold_counters[o] + threshold_delta[o]
        threshold_delta[o] = 0
        step = np.int32(t / threshold_level)  # truncates towards zero
        thresholds[o] += step
        threshold_counters[o] = t - step * threshold_level


@njit(cache=True)
def train_epoch(
    x, targets, order, batch_size, params,
    counters, counter_delta, touched, weights, exc, inh,
    thresholds, threshold_counters, threshold_delta, rng_state,
    homeo_run, homeo_last, homeo_mean,
    tally_weights, tally_thresholds, tally_since, tally_count, tallying,
):
    """
    Forward, backward and update over every batch of `order` for a stack of layers,
    the output layer under output_pressure(targets). Per-layer state comes in lists
    (see LAYER_STATE), settings as one tuple per layer (CounterDense._params).
    Returns the number of updates.
    """
    n_layers = len(counters)
    updates = 0
    for start in range(0, order.shape[0], batch_size):
        batch = order[start:start + batch_size]
        inputs = [x[batch]]
        sums = []
        for k in range(n_layers):
            sums.append(margins(pack_bits(inputs[k]), exc[k], inh[k], thresholds[k]))
            inputs.append(sums[k] >= 0)

        q = output_pressure(sums[n_layers - 1], targets[batch], params[n_layers - 1][7])
        for k in range(n_layers - 1, -1, -1):
            _, _, _, window, accept_distance, stabilize_margin, run_length, _, surrogate, normalize_reach = params[k]
            q = backward(
                inputs[k], sums[k], q, weights[k], window, accept_distance, stabilize_margin,
                surrogate, normalize_reach != 0, k > 0,
                counter_delta[k], threshold_delta[k], touched[k], rng_state[k],
            )
            if run_length > 0:
                homeostasis(sums[k], thresholds[k], threshold_counters[k], run_length,
                            homeo_run[k], homeo_last[k], homeo_mean[k])

        for k in range(n_layers):
            connect_level, counter_max, threshold_level = params[k][0], params[k][1], params[k][2]
            update(
                counters[k], counter_delta[k], touched[k], counter_max, connect_level,
                weights[k], exc[k], inh[k],
                thresholds[k], threshold_counters[k], threshold_delta[k], threshold_level,
                tally_weights[k], tally_since[k], tally_count + updates, tallying,
            )
            if tallying:
                tally_thresholds[k] += thresholds[k]
        updates += 1
    return updates
