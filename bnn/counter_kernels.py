"""
Compiled (numba) kernels for CounterDense.

Binary inputs and ternary weights are packed into 64-bit words, so a neuron's
input sum is popcount(x & exc) - popcount(x & inh) over I/64 words.
The backward pass looks only at outputs with an error signal (all of them when
stabilizing), decides each without branches, and touches a whole input row
only where an output votes or blames, which after the first epochs is rare.
Random numbers come from a per-layer xoshiro256+ state, one per nonzero error
signal. `train_epoch` runs a stack of layers for a whole epoch: per batch, the
numpy glue would cost more than the kernels.

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
)


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
def backward(
    x, x_bits, margins, wish,
    weights, exc, inh,
    backprop_margin, accept_distance, stabilize_margin, blame_inputs,
    counter_delta, threshold_delta, touched, rng_state,
):
    """
    Accumulate counter and threshold votes; return (too_much, not_enough) for the inputs
    (no rows unless blame_inputs). `wish` is not_enough - too_much for the outputs.
    See ReferenceCounterDense.backward for the rule; float operations follow it
    term by term so both give the same numbers.
    """
    batch, n_in = x.shape
    n_out = margins.shape[1]
    words = x_bits.shape[1]
    input_too_much = np.zeros((batch if blame_inputs else 0, n_in), np.float32)
    input_not_enough = np.zeros((batch if blame_inputs else 0, n_in), np.float32)
    # what each output of the current sample does to the blame;
    # FINE outputs are classified into DROP / RISE / nothing only for samples that blame
    FINE, FIX_ON, FIX_OFF, UNFIXABLE = 0, 1, 2, 3
    kind = np.zeros(n_out, np.int8)
    share = np.zeros(n_out, np.float32)
    # share-weighted helper counts from outputs wanting on / off,
    # and counts of outputs a flip would break (fine and at margin 0 or -1)
    act_on = np.empty(n_in, np.float32)
    act_off = np.empty(n_in, np.float32)
    deact_on = np.empty(n_in, np.float32)
    deact_off = np.empty(n_in, np.float32)
    drop_exc = np.empty(n_in, np.float32)
    drop_inh = np.empty(n_in, np.float32)
    rise_exc = np.empty(n_in, np.float32)
    rise_inh = np.empty(n_in, np.float32)

    active = np.arange(n_out)
    for b in range(batch):
        blamed = False
        if blame_inputs:
            kind[:] = FINE
        # Without a request or anything to stabilize an output neither votes nor blames.
        # Hidden layers get few requests; collecting them first keeps this scan tight.
        n_active = n_out
        if stabilize_margin <= 0:
            n_active = 0
            for o in range(n_out):
                active[n_active] = o
                n_active += wish[b, o] != 0
        for j in range(n_active):
            o = active[j]
            w = wish[b, o]
            m = np.int64(margins[b, o])
            out = m >= 0
            # one draw per nonzero wish, in row-major order; p = 0 asks for nothing
            p = next_uniform(rng_state) if w != 0 else 0.0
            # branch-free: output signs and requests are close to random, so branches mispredict
            distance = m ^ (m >> 63)  # m if out else -m - 1
            accepted = distance <= accept_distance
            want_on = (not out) & (p < w) & accepted
            want_off = out & (p < -w) & accepted
            fragile = (distance < stabilize_margin) & (not want_on) & (not want_off)
            vote = np.int32(want_on | (fragile & out)) - np.int32(want_off | (fragile & (not out)))
            if vote != 0:
                for i in range(n_in):
                    counter_delta[o, i] += vote * (2 * np.int32(x[b, i]) - 1)
                threshold_delta[o] -= vote
                touched[o] = True

            if not (blame_inputs and (want_on | want_off)):
                continue
            kind[o] = UNFIXABLE
            needed = -m if want_on else m + 1
            if needed > backprop_margin:
                continue
            active_exc = 0
            active_inh = 0
            all_exc = 0
            all_inh = 0
            for word in range(words):
                active_exc += _popcount(x_bits[b, word] & exc[o, word])
                active_inh += _popcount(x_bits[b, word] & inh[o, word])
                all_exc += _popcount(exc[o, word])
                all_inh += _popcount(inh[o, word])
            if want_on:
                helpers = all_exc - active_exc + active_inh
            else:
                helpers = active_exc + all_inh - active_inh
            if needed <= helpers:
                kind[o] = FIX_ON if want_on else FIX_OFF
                share[o] = np.float32(np.float64(needed) / np.float64(helpers))
                blamed = True

        # without a fixable output the blame is max(-collateral, 0) = 0
        if not blamed:
            continue
        act_on[:] = 0
        act_off[:] = 0
        deact_on[:] = 0
        deact_off[:] = 0
        drop_exc[:] = 0
        drop_inh[:] = 0
        rise_exc[:] = 0
        rise_inh[:] = 0
        for o in range(n_out):
            k = kind[o]
            s = share[o]
            if k == FIX_ON:
                for i in range(n_in):
                    act_on[i] += s * np.float32(weights[o, i] > 0)
                    deact_on[i] += s * np.float32(weights[o, i] < 0)
            elif k == FIX_OFF:
                for i in range(n_in):
                    act_off[i] += s * np.float32(weights[o, i] < 0)
                    deact_off[i] += s * np.float32(weights[o, i] > 0)
            elif k == FINE and margins[b, o] == 0:
                for i in range(n_in):
                    drop_exc[i] += np.float32(weights[o, i] > 0)
                    drop_inh[i] += np.float32(weights[o, i] < 0)
            elif k == FINE and margins[b, o] == -1:
                for i in range(n_in):
                    rise_exc[i] += np.float32(weights[o, i] > 0)
                    rise_inh[i] += np.float32(weights[o, i] < 0)
        for i in range(n_in):
            if x[b, i]:
                v = (deact_on[i] + deact_off[i]) - drop_exc[i] - rise_inh[i]
                input_too_much[b, i] = max(v, np.float32(0))
            else:
                v = (act_on[i] + act_off[i]) - drop_inh[i] - rise_exc[i]
                input_not_enough[b, i] = max(v, np.float32(0))
    return input_too_much, input_not_enough


@njit(cache=True)
def update(
    counters, counter_delta, touched, counter_max, connect_level, weights, exc, inh,
    thresholds, threshold_counters, threshold_delta, threshold_level,
):
    """
    Apply and clear the votes accumulated in touched rows,
    flip the synapses whose counter crossed connect_level.
    """
    n_out, n_in = counters.shape
    for o in range(n_out):
        if touched[o]:
            touched[o] = False
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
    tally_weights, tally_thresholds, tallying,
):
    """
    Forward, backward and update over every batch of `order` for a stack of layers,
    with output errors "too much" = out & ~targets, "not enough" = ~out & targets.
    Per-layer state comes in lists (see LAYER_STATE), settings as one tuple per layer
    (CounterDense._params). Returns the number of updates.
    """
    n_layers = len(counters)
    updates = 0
    for start in range(0, order.shape[0], batch_size):
        batch = order[start:start + batch_size]
        inputs = [x[batch]]
        bits = []
        sums = []
        for k in range(n_layers):
            bits.append(pack_bits(inputs[k]))
            sums.append(margins(bits[k], exc[k], inh[k], thresholds[k]))
            inputs.append(sums[k] >= 0)

        wish = targets[batch].astype(np.float32) - inputs[n_layers].astype(np.float32)
        for k in range(n_layers - 1, -1, -1):
            _, _, _, backprop_margin, accept_distance, stabilize_margin = params[k]
            too_much, not_enough = backward(
                inputs[k], bits[k], sums[k], wish, weights[k], exc[k], inh[k],
                backprop_margin, accept_distance, stabilize_margin, k > 0,
                counter_delta[k], threshold_delta[k], touched[k], rng_state[k],
            )
            wish = not_enough - too_much

        for k in range(n_layers):
            connect_level, counter_max, threshold_level, _, _, _ = params[k]
            update(
                counters[k], counter_delta[k], touched[k], counter_max, connect_level,
                weights[k], exc[k], inh[k],
                thresholds[k], threshold_counters[k], threshold_delta[k], threshold_level,
            )
            if tallying:
                tally_weights[k] += weights[k]
                tally_thresholds[k] += thresholds[k]
        updates += 1
    return updates
