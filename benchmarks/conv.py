"""
Fashion-MNIST: ConvCounter layers and a CounterDense readout (README, "Convolution").

usage: python -m benchmarks.conv spec mode seed [epochs] [conv overrides json] [max_train]
  spec: layers, comma separated: filters x kernel / stride, or p<size> for OR pooling; e.g. 64x5/2 or 64x3/1,p2
  overrides: CounterDense settings for the filters, each a value or a list with one per conv layer;
    "threshold": the filters' starting threshold (default 1; fixed filters keep it)
  mode: fixed (random filters, only the readout learns) | learn
  max_train: train on the first n samples
  e.g. python -m benchmarks.conv 64x5/2 fixed 0 3
       python -m benchmarks.conv 32x5/1,p2,64x3/1,p2,64x2/1,p2 learn 0 4 '{"threshold": [1, 4, 6]}' 20000
"""
import json
import re
import sys
import time

import numpy as np

from benchmarks.datasets import load_fashion_mnist
from bnn.abstract import Sequence
from bnn.conv import ConvCounter, OrPool
from bnn.conv import train_epoch as conv_train_epoch
from bnn.counter import CounterDense, readout_inputs, train_epoch
from bnn.exact import polish_readout

spec, mode, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
overrides = json.loads(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else {}
overrides.setdefault("threshold", 1)  # a blank patch sums to 0: at 0, the background fires
max_train = int(sys.argv[6]) if len(sys.argv) > 6 else None
X, y, Xt, yt = load_fashion_mnist()
X, y = X[:max_train], y[:max_train]
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
    # homeostasis counts patch rows: at 256 a filter off over part of one image's background jumped its threshold
    kwargs = dict(accept_distance=0, homeostasis=4096, threshold_level=1, rng=rng)
    if mode == "fixed":
        kwargs.update(connect_level=256, counter_max=2048, threshold_level=10**9, homeostasis=0)
    k = sum(isinstance(layer, ConvCounter) for layer in layers)
    settings = {name: value[k] if isinstance(value, list) else value for name, value in overrides.items()}
    threshold = settings.pop("threshold")
    kwargs.update(settings)
    conv = ConvCounter(channels, size, size, filters, kernel, stride, **kwargs)
    if mode == "fixed":  # random filters, about half the synapses present, far from changing
        conv.dense.counters[:] = rng.choice([-2048, 0, 0, 2048], conv.dense.counters.shape)
    conv.dense.thresholds[:] = threshold  # also where learned filters start
    conv.dense._sync_weights()
    layers.append(conv)
    channels, size = filters, conv.out_height
# the readout as build_counter_network's output layer: it passes its pressure down whole
lk = {"connect_level": 16, "counter_max": 128, "init_range": 24, "threshold_level": 1, "stabilize_margin": 2,
      "surrogate": "none", "normalize_reach": False}
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


def activity(n=1000):
    """Fraction on after each conv and pool layer, on the first n test images (large OR pools saturate)."""
    a, out = Xt[:n], []
    for layer in layers:
        a = layer.margins(a) >= 0
        out.append(f"{a.mean():.2f}")
    return " ".join(out)


t0 = time.time()
if mode == "fixed":
    F, Ft = features(X), features(Xt)
    print(f"  features: {time.time() - t0:.0f}s", flush=True)
    fit_readout(F, epochs)
else:
    for epoch in range(epochs):
        if epoch == epochs - 1:
            for layer in net.layers:
                layer.start_tally()
        conv_train_epoch(net.layers, X, targets, rng.permutation(len(X)), 32)
        print(f"  epoch {epoch + 1}: test {accuracy(features(Xt)):.4f}  active {activity()}  {time.time() - t0:.0f}s",
              flush=True)
    for layer in net.layers:
        layer.consolidate()
    # each layer was averaged on its own: refit the readout to the code the consolidated filters give
    F, Ft = features(X), features(Xt)
    fit_readout(F, 1)
a0 = accuracy(Ft)
polish_readout(readout, F, y, max_steps=32)
a1 = accuracy(Ft)
synapses = [int((layer.dense.weights != 0).sum()) for layer in layers if hasattr(layer, "dense")]
print(f"RESULT {spec} {mode} seed={seed} epochs={epochs} n={len(X)} {overrides}: acc {a0:.4f} final {a1:.4f}  "
      f"active {activity()}  filter synapses {synapses}  readout synapses {int((readout.weights != 0).sum())}  "
      f"{time.time() - t0:.0f}s", flush=True)
