"""
Fashion-MNIST: ConvCounter layers and a CounterDense readout (README, "Convolution").

usage: python -m benchmarks.conv spec mode seed [epochs] [conv overrides json]
  spec: layers, comma separated: filters x kernel / stride, or p<size> for OR pooling; e.g. 64x5/2 or 64x3/1,p2
  overrides: CounterDense settings for the filters; "threshold": fixed filters' threshold (default 1),
    a number or one per conv layer
  mode: fixed (random filters, only the readout learns) | learn
  e.g. python -m benchmarks.conv 64x5/2 fixed 0 3
"""
import json
import re
import sys
import time

import numpy as np

from benchmarks.datasets import load_fashion_mnist
from bnn.abstract import Sequence
from bnn.conv import ConvCounter, OrPool
from bnn.counter import CounterDense, readout_inputs, train_epoch
from bnn.exact import polish_readout

spec, mode, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
overrides = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
threshold = overrides.pop("threshold", 1)  # a blank patch sums to 0: at 0, the background fires
n_conv = sum(not part.startswith("p") for part in spec.split(","))
thresholds = iter(threshold if isinstance(threshold, list) else [threshold] * n_conv)
X, y, Xt, yt = load_fashion_mnist()
rng = np.random.default_rng(seed)
targets = np.arange(10) == y[:, None]

layers = []
channels, size = 2, 28
for part in spec.split(","):
    if part.startswith("p"):
        layers.append(OrPool(channels, size, size, int(part[1:])))
        size = layers[-1].out_height
        continue
    filters, kernel, stride = map(int, re.fullmatch(r"(\d+)x(\d+)/(\d+)", part).groups())
    kwargs = dict(accept_distance=0, homeostasis=256, threshold_level=1, rng=rng)
    if mode == "fixed":
        kwargs.update(connect_level=256, counter_max=2048, threshold_level=10**9, homeostasis=0)
    kwargs.update(overrides)
    conv = ConvCounter(channels, size, size, filters, kernel, stride, **kwargs)
    if mode == "fixed":  # random filters, about half the synapses present, far from changing
        conv.dense.counters[:] = rng.choice([-2048, 0, 0, 2048], conv.dense.counters.shape)
    conv.dense.thresholds[:] = next(thresholds)  # also where learned filters start
    conv.dense._sync_weights()
    layers.append(conv)
    channels, size = filters, conv.out_height
lk = {"connect_level": 16, "counter_max": 128, "init_range": 24, "threshold_level": 1, "stabilize_margin": 2}
readout = CounterDense(layers[-1].output_features, 10, rng=rng, **lk)
net = Sequence([*layers, readout])
print(f"{spec} {mode}: {readout.input_features} features", flush=True)


def features(inputs, chunk=2048):
    """What the readout reads, in chunks (patches of a second conv layer over 60k images don't fit in memory)."""
    return np.concatenate([readout_inputs(net.layers, inputs[s:s + chunk]) for s in range(0, len(inputs), chunk)])


def fit_readout(F, epochs):
    """Train the readout alone on fixed features, tallying the last epoch, and consolidate."""
    for epoch in range(epochs):
        if epoch == epochs - 1:
            readout.start_tally()
        train_epoch([readout], F, targets, rng.permutation(len(F)), 32)
    readout.consolidate()


def accuracy(Ft):
    return np.mean(np.argmax(readout.margins(Ft), 1) == yt)


t0 = time.time()
if mode == "fixed":
    F, Ft = features(X), features(Xt)
    print(f"  features: {time.time() - t0:.0f}s, active {F.mean():.2f}", flush=True)
    for i in range(1, len(layers) + 1):
        print(f"  layer {i} active {np.mean(readout_inputs(net.layers[:i + 1], Xt[:500])):.3f}", flush=True)
    fit_readout(F, epochs)
else:
    for epoch in range(epochs):
        if epoch == epochs - 1:
            for layer in net.layers:
                layer.start_tally()
        order = rng.permutation(len(X))
        for s in range(0, len(X), 32):
            idx = order[s:s + 32]
            _, state = net.forward(X[idx])
            net.backward(state, *readout.output_pressure(state[-1], targets[idx]))
            net.update()
        print(f"  epoch {epoch + 1}: test {accuracy(features(Xt)):.4f}  {time.time() - t0:.0f}s", flush=True)
    for layer in net.layers:
        layer.consolidate()
    # each layer was averaged on its own: refit the readout to the code the consolidated filters give
    F, Ft = features(X), features(Xt)
    fit_readout(F, 1)
a0 = accuracy(Ft)
polish_readout(readout, F, y, max_steps=32)
a1 = accuracy(Ft)
synapses = [int((layer.dense.weights != 0).sum()) for layer in layers if hasattr(layer, "dense")]
print(f"RESULT {spec} {mode} seed={seed} epochs={epochs} threshold={threshold} {overrides}: acc {a0:.4f} final {a1:.4f}  "
      f"filter synapses {synapses}  readout synapses {int((readout.weights != 0).sum())}  {time.time() - t0:.0f}s", flush=True)
