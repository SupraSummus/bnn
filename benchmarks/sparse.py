"""
Fashion-MNIST: a wide SparseCounter hidden layer (N pointers per unit) and a readout (README, "Sparse").

usage: python -m benchmarks.sparse width n_slots hidden_mode readout seed [epochs] [hidden overrides json]
  e.g. python -m benchmarks.sparse 4096 8 fixed dense 0 3
  hidden_mode: frozen (thresholds learn) | fixed | fixed<t> (thresholds stay at t) | learn | rewire<p>
  readout: dense | sparse<N> | sparse<N>r<p> (rewiring with chance p per vote)
"""
import json
import re
import sys
import time

import numpy as np

from benchmarks.bnn_model import predict
from benchmarks.datasets import load_fashion_mnist
from bnn.abstract import Sequence
from bnn.counter import CounterDense, readout_inputs, refit_readout
from bnn.exact import polish_readout
from bnn.sparse import SparseCounter

width, k, mode, readout_kind, seed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 5
overrides = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
X, y, Xt, yt = load_fashion_mnist()
rng = np.random.default_rng(seed)
targets = np.arange(10) == y[:, None]

rewire = float(mode[6:]) if mode.startswith("rewire") else 0.0
hidden_kw = dict(accept_distance=0, homeostasis=256, threshold_level=1, rewire=rewire, rng=rng)
frozen = mode.startswith("frozen") or mode.startswith("fixed")
if frozen:
    hidden_kw.update(connect_level=256, counter_max=2048)
if mode.startswith("fixed"):  # thresholds stay put too
    hidden_kw.update(threshold_level=10**9, homeostasis=0)
hidden_kw.update(overrides)
hidden = SparseCounter(X.shape[1], width, k, **hidden_kw)
if frozen:
    hidden.counters[:] = rng.choice([-2048, 2048], hidden.counters.shape)
    hidden.sync_weights()
if mode.startswith("fixed") and len(mode) > 5:
    hidden.thresholds[:] = int(mode[5:])
lk = {"connect_level": 16, "counter_max": 128, "init_range": 24, "threshold_level": 1, "stabilize_margin": 2}
if readout_kind == "dense":
    readout = CounterDense(width, 10, rng=rng, **lk)
else:
    n, p = re.fullmatch(r"sparse(\d+)(?:r([\d.]+))?", readout_kind).groups()
    readout = SparseCounter(width, 10, int(n), stabilize_margin=2, threshold_level=1, rewire=float(p or 0), rng=rng)
net = Sequence([hidden, readout])


def epoch_of(layers, inputs):
    order = rng.permutation(len(inputs))
    for s in range(0, len(inputs), 32):
        idx = order[s:s + 32]
        _, state = layers.forward(inputs[idx])
        layers.backward(state, *layers.layers[-1].output_pressure(state[-1], targets[idx]))
        layers.update()


t0 = time.time()
for epoch in range(epochs):
    if epoch == epochs - 1:
        readout.start_tally()
    epoch_of(net, X)
    acc = np.mean(predict(net, Xt) == yt)
    print(f"  epoch {epoch + 1}: test {acc:.4f}  synapses/unit {np.mean(hidden.weights != 0) * k:.1f}  "
          f"moved {hidden.moved} / readout {getattr(readout, 'moved', 0)}", flush=True)
readout.consolidate()
if readout_kind == "dense":
    refit_readout(net.layers, X, targets, rng.permutation(len(X)), 32)
    a0 = np.mean(predict(net, Xt) == yt)
    polish_readout(readout, readout_inputs(net.layers, X), y, max_steps=32)
else:  # the same refit, through the per-batch path
    readout.start_tally()
    epoch_of(Sequence([readout]), readout_inputs(net.layers, X))
    readout.consolidate()
    a0 = np.mean(predict(net, Xt) == yt)
a1 = np.mean(predict(net, Xt) == yt)
hs = int((hidden.weights != 0).sum())
rs = int((readout.weights != 0).sum())
active = np.mean(readout_inputs(net.layers, Xt[:2000]))
print(f"RESULT width={width} k={k} hidden={mode} readout={readout_kind} seed={seed} epochs={epochs} {overrides}: "
      f"acc {a0:.4f} final {a1:.4f}  hidden synapses {hs}  readout synapses {rs}  hidden active {active:.2f}  "
      f"{time.time() - t0:.0f}s", flush=True)
