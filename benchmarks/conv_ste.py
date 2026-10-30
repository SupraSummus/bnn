"""
The discrete conv nets of benchmarks.conv trained by gradient descent instead of the rule (README, "Convolution"):
ternary weights from counters (a weight is on while |counter| >= 16), binary units (margin >= 0), OR (max) pooling,
a ternary readout; straight-through gradients (the step's derivative is 1 within sqrt(fan_in) of the threshold),
Adam on the counters. `float` is the same net as a plain ReLU CNN, for reference. Needs torch (not a project dependency).

usage: python -m benchmarks.conv_ste spec binary|float seed [epochs] [max_train] [batch] [name=value ...]
  spec: comma separated, filters x kernel (stride 1, valid) or p<size> for pooling; e.g. 32x5,p2,64x3,p2
  options (binary): w=float (float weights, counter / 16), act=relu (ReLU units), win=<STE window multiplier>
  e.g. python -m benchmarks.conv_ste 32x5,p2 binary 0 3 10000 128
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.datasets import load_fashion_mnist

spec, kind, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
max_train = int(sys.argv[5]) if len(sys.argv) > 5 else None
batch = int(sys.argv[6]) if len(sys.argv) > 6 else 64
opts = dict(a.split("=") for a in sys.argv[7:])
torch.manual_seed(seed)
torch.set_num_threads(1)
CONNECT, COUNTER_MAX, INIT = 16.0, 128.0, 24.0

X, y, Xt, yt = load_fashion_mnist()
X, y = torch.tensor(X[:max_train].reshape(-1, 2, 28, 28), dtype=torch.float32), torch.tensor(y[:max_train], dtype=torch.long)
Xt, yt = torch.tensor(Xt.reshape(-1, 2, 28, 28), dtype=torch.float32), torch.tensor(yt, dtype=torch.long)


def ternary(c):
    """Weights from counters, with the counters' gradient passed straight through."""
    if opts.get("w") == "float":
        return c / CONNECT
    return c + (torch.sign(c) * (c.abs() >= CONNECT) - c).detach()


layers, params = [], []
channels, size = 2, 28
for part in spec.split(","):
    if part.startswith("p"):
        layers.append(int(part[1:]))
        size //= int(part[1:])
        continue
    filters, kernel = map(int, part.split("x"))
    fan_in = channels * kernel * kernel
    if kind == "binary":
        c = (torch.rand(filters, channels, kernel, kernel) * 2 - 1) * INIT
    else:
        c = torch.randn(filters, channels, kernel, kernel) * np.sqrt(2 / fan_in)
    t = torch.zeros(filters)
    params += [c.requires_grad_(), t.requires_grad_()]
    layers.append((c, t, np.sqrt(fan_in) * float(opts.get("win", 1))))
    channels, size = filters, size - kernel + 1
n_features = channels * size * size
if kind == "binary":
    rc = (torch.rand(10, n_features) * 2 - 1) * INIT
else:
    rc = torch.randn(10, n_features) * np.sqrt(1 / n_features)
rt = torch.zeros(10)
params += [rc.requires_grad_(), rt.requires_grad_()]
optimizer = torch.optim.Adam(params, lr=0.5 if kind == "binary" else 1e-3)


def net(x):
    for layer in layers:
        if isinstance(layer, int):
            x = F.max_pool2d(x, layer)
            continue
        c, t, window = layer
        if kind == "float":
            x = F.relu(F.conv2d(x, c) - t[None, :, None, None])
            continue
        m = F.conv2d(x, ternary(c)) - (t + (torch.round(t) - t).detach())[None, :, None, None]
        if opts.get("act") == "relu":
            x = F.relu(m) / window
        else:
            x = (m >= 0).float().detach() + (m - m.detach()) * (m.detach().abs() <= window)
    x = x.flatten(1)
    if kind == "float":
        return x @ rc.T - rt
    return (x @ ternary(rc).T - rt) / (np.sqrt(n_features) / 3)  # at the hinge's scale, as in the ladder


def accuracy(inputs, labels):
    with torch.no_grad():
        return np.mean([(net(inputs[s:s + 1000]).argmax(1) == labels[s:s + 1000]).float().mean().item()
                        for s in range(0, len(inputs), 1000)])


t0 = time.time()
test, train = [], []
for epoch in range(epochs):
    order = torch.randperm(len(X))
    for s in range(0, len(X), batch):
        idx = order[s:s + batch]
        loss = F.cross_entropy(net(X[idx]), y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if kind == "binary":
            with torch.no_grad():
                for counters in params[::2]:
                    counters.clamp_(-COUNTER_MAX, COUNTER_MAX)
    test.append(accuracy(Xt, yt))
    train.append(accuracy(X[:10000], y[:10000]))
    print(f"  epoch {epoch + 1}: test {test[-1]:.4f}  train {train[-1]:.4f}  {time.time() - t0:.0f}s", flush=True)
print(f"RESULT {spec} {kind} {' '.join(sys.argv[7:])} seed={seed} epochs={epochs} n={len(X)} batch={batch}: "
      f"final {test[-1]:.4f}  best {max(test):.4f}  train {train[-1]:.4f}  {time.time() - t0:.0f}s", flush=True)
