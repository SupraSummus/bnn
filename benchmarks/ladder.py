"""
From gradient descent to the CounterDense rule, one part at a time (README, "The ladder").

A chain of hidden layers on Fashion-MNIST or parity, the same discrete net throughout (ternary synapses from counters,
binary units, thresholds), trained by a rule assembled from parts. Each part has a gradient (STE)
version and a rule version; all rule parts are CounterDense, all STE parts are straight-through
gradient descent with Adam. Everything is in counter units: a synapse exists while |counter| >= connect_level.

Parts (STE / rule):
  opt          adam: Adam on the counters / sgd: the signal itself is the step (expected size of the rule's votes)
  loss         ce: softmax cross-entropy on the output margins / hinge: CounterDense.output_pressure
  down         ste: pressure on a unit = sum of pressure through its synapses, times the step's straight-through
               derivative (1 within the window) below hidden layers /
               rule: the output passes it whole; hidden layers only within the window, 1 / (distance + 1) each,
               divided by sqrt(reach); also triangle: the output too passes it as hidden layers do (before the port)
  gate_hidden  ste: every unit within the window of its threshold learns /
               rule: minimal disturbance, only flips at distance 0;
               window: flips within the window and stabilizing closer than it (CounterDense's "window");
               also flip<K>: flips at distance <= K, near<K>: any vote at distance <= K
  gate_output  ste: every output with pressure learns / rule: flips, and stabilizing below distance 2
  (gate=... sets both)
  discrete     float: counters move by the signal / counters: binary votes with probability |signal|
  inactive     x: only active inputs move (the gradient of x @ w) / pm: inactive inputs move the other way (2x - 1)
  homeostasis  off / on (hidden units stuck for 256 samples move their threshold to their mean input sum)
  consolidate  off / on (average the last epoch, refit the readout for one epoch)
  hidden       learn (both) / frozen: the hidden layers stay at their random start (a reference, not a part)

--probe prints, after each epoch, how each hidden layer's signal (before gating) agrees with what flipping
a unit actually does to the loss (cross-entropy, or the hinge's violations), and with the STE gradient.

usage: python -m benchmarks.ladder [--data fashion|parity<N>] [--hidden 1024|64,64,64] [--seed S] [--epochs E]
                                   [--lr LR] [--batch B] [--probe] name=value ...
  e.g. python -m benchmarks.ladder              (the rule)
       python -m benchmarks.ladder --ste        (gradient descent)
       python -m benchmarks.ladder --ste opt=sgd loss=hinge
       python -m benchmarks.ladder --data parity16 --hidden 64,64,64 --epochs 60 gate_hidden=window
"""
import argparse
import time

import numpy as np

from benchmarks.datasets import load_fashion_mnist, load_parity
from bnn.counter import resolve_settings

RULE = dict(opt="sgd", loss="hinge", down="rule", gate_hidden="rule", gate_output="rule", discrete="counters",
            inactive="pm", homeostasis="on", consolidate="on", hidden="learn")
STE = dict(opt="adam", loss="ce", down="ste", gate_hidden="ste", gate_output="ste", discrete="float",
           inactive="x", homeostasis="off", consolidate="off", hidden="learn")

HOMEOSTASIS_RUN, HOMEOSTASIS_RATE = 256, 1.0 / 64


class Layer:
    def __init__(self, n_in, n_out, rng, connect_level=None, counter_max=None, init_range=None, threshold_level=4,
                 stabilize_margin=0):
        self.c, self.cmax, init_range, self.window, _, _, self.hinge = resolve_settings(
            n_in, connect_level, counter_max, init_range, None, None, stabilize_margin, None)
        self.threshold_level = threshold_level
        self.counters = rng.integers(-init_range, init_range + 1, (n_out, n_in)).astype(np.float32)
        self.thresholds = np.zeros(n_out, np.float32)  # float under float/adam, whole numbers under counters
        self.threshold_counters = np.zeros(n_out, np.float32)
        self.adam = [np.zeros_like(self.counters), np.zeros_like(self.counters),
                     np.zeros_like(self.thresholds), np.zeros_like(self.thresholds)]
        self.steps = 0
        self.homeo = [np.zeros(n_out, np.int64), np.zeros(n_out, bool), np.zeros(n_out)]
        self.tally = None
        self.sync()

    def sync(self):
        self.weights = (np.sign(self.counters) * (np.abs(self.counters) >= self.c)).astype(np.float32)

    def visible_thresholds(self):
        return np.round(self.thresholds)

    def margins(self, x):
        return x.astype(np.float32) @ self.weights.T - self.visible_thresholds()

    def update(self, x, signal, cfg, lr, rng):
        """signal (batch, outputs) > 0 asks for more excitation."""
        x_code = 2 * x.astype(np.float32) - 1 if cfg["inactive"] == "pm" else x.astype(np.float32)
        if cfg["discrete"] == "counters":
            votes = np.sign(signal) * (rng.random(signal.shape) < np.abs(signal))
            delta, t_delta = votes.T @ x_code, -votes.sum(0)
            self.counters = np.clip(self.counters + delta, -self.cmax, self.cmax)
            self.threshold_counters += t_delta
            step = np.trunc(self.threshold_counters / self.threshold_level)
            self.thresholds += step
            self.threshold_counters -= step * self.threshold_level
        else:
            delta, t_delta = signal.T @ x_code, -signal.sum(0)
            if cfg["opt"] == "adam":
                self.steps += 1
                delta, t_delta = self._adam(delta, 0), self._adam(t_delta, 2)
            self.counters = np.clip(self.counters + lr * delta, -self.cmax, self.cmax)
            self.thresholds += lr * t_delta / self.threshold_level
        self.sync()
        if self.tally is not None:
            self.tally[0] += self.weights
            self.tally[1] += self.visible_thresholds()
            self.tally[2] += 1

    def _adam(self, step, k):
        m, v = self.adam[k], self.adam[k + 1]
        m *= 0.9
        m += 0.1 * step
        v *= 0.999
        v += 0.001 * step * step
        return (m / (1 - 0.9 ** self.steps)) / (np.sqrt(v / (1 - 0.999 ** self.steps)) + 1e-8)

    def homeostasis(self, margins):
        run, last, mean = self.homeo
        base = self.visible_thresholds()
        for m in margins:
            out = m >= 0
            mean += (m + base - mean) * HOMEOSTASIS_RATE
            run[:] = np.where(out == last, run + 1, 1)
            last[:] = out
            revive = run >= HOMEOSTASIS_RUN
            self.thresholds[revive] = np.floor(mean[revive])
            self.threshold_counters[revive] = 0
            run[revive] = 0

    def start_tally(self):
        self.tally = [np.zeros_like(self.weights), np.zeros_like(self.thresholds), 0]

    def consolidate(self):
        weights, thresholds, n = self.tally
        self.tally = None
        if n == 0:  # not trained while tallying (a frozen layer): keep the state
            return
        self.counters = np.sign(weights) * (2 * np.abs(weights) >= n) * self.c
        self.thresholds = np.round(thresholds / n)
        self.threshold_counters[:] = 0
        self.sync()


def distance(margins):
    return np.where(margins >= 0, margins, -margins - 1)


def output_signal(margins, targets, cfg, layer):
    if cfg["loss"] == "ce":
        logits = margins / (layer.window / 3)  # the hinge's scale
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return targets - p
    right = (margins >= 0) == targets
    return np.where(right & (distance(margins) >= layer.hinge), 0, np.where(targets, 1, -1)).astype(np.float32)


def pressure_down(q, margins, layer, cfg, output):
    if cfg["down"] == "ste" or (output and cfg["down"] == "rule"):
        return (q if output else q * (np.abs(margins) <= layer.window)) @ layer.weights
    d = distance(margins)
    near = (q != 0) & (d < layer.window)
    q_in = np.where(near, q / (d + 1), 0).astype(np.float32) @ layer.weights
    reach = near.astype(np.float32) @ (layer.weights != 0).astype(np.float32)
    return np.where(reach > 1, q_in / np.sqrt(np.maximum(reach, 1)), q_in)


def gate(q, margins, layer, cfg, hidden):
    """The part of the signal a layer learns from."""
    kind = cfg["gate_hidden" if hidden else "gate_output"]
    if kind == "ste":
        return q * (np.abs(margins) <= layer.window) if hidden else q
    d = distance(margins)
    flip = (q > 0) != (margins >= 0)
    if not hidden:
        may = np.where(flip, True, d < 2)
    elif kind == "rule":
        may = flip & (d <= 0)
    elif kind == "window":
        may = np.where(flip, d <= layer.window, d < layer.window)
    elif kind.startswith("flip"):
        may = flip & (d <= int(kind[4:]))
    elif kind.startswith("near"):
        may = d <= int(kind[4:])
    else:
        raise ValueError(kind)
    return np.where((q != 0) & may, q, 0)


def forward(layers, x):
    """Margins of every layer."""
    margins = []
    for layer in layers:
        margins.append(layer.margins(x))
        x = margins[-1] >= 0
    return margins


def signals(layers, x, margins, targets, cfg):
    """Each layer's signal before gating, top down."""
    q = [output_signal(margins[-1], targets, cfg, layers[-1])]
    for k in range(len(layers) - 1, 0, -1):
        q.insert(0, pressure_down(q[0], margins[k], layers[k], cfg, k == len(layers) - 1))
    return q


def train(cfg, X, y, Xt, yt, hidden=(1024,), seed=0, epochs=5, lr=None, batch=32, layer_kwargs=None, probe=False,
          log=print):
    rng = np.random.default_rng(seed)
    if lr is None:
        lr = 0.5 if cfg["opt"] == "adam" else 1.0
    sizes = [X.shape[1], *hidden]
    n_classes = int(y.max()) + 1
    layers = [Layer(n_in, n_out, rng, **(layer_kwargs or {})) for n_in, n_out in zip(sizes, sizes[1:])]
    layers.append(Layer(sizes[-1], n_classes, rng, stabilize_margin=2, **(layer_kwargs or {})))
    targets = (np.arange(n_classes) == y[:, None]).astype(np.float32)
    probe_rng = np.random.default_rng(seed + 1)  # its own generator: probing doesn't change the run
    probe_idx = probe_rng.choice(len(X), min(512, len(X)), replace=False)

    def accuracy():
        return np.mean(np.argmax(forward(layers, Xt)[-1], 1) == yt)

    def epoch(train_hidden=True):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            x = X[idx]
            margins = forward(layers, x)
            q = signals(layers, x, margins, targets[idx], cfg)
            inputs = [x] + [m >= 0 for m in margins[:-1]]
            for k, layer in enumerate(layers):
                hidden_layer = k < len(layers) - 1
                if hidden_layer and not (train_hidden and cfg["hidden"] == "learn"):
                    continue
                if hidden_layer and cfg["homeostasis"] == "on":  # on the margins as computed, before this update's votes
                    layer.homeostasis(margins[k])
                layer.update(inputs[k], gate(q[k], margins[k], layer, cfg, hidden_layer), cfg, lr, rng)

    t0 = time.time()
    raw = []
    for e in range(epochs):
        if e == epochs - 1 and cfg["consolidate"] == "on":
            for layer in layers:
                layer.start_tally()
        epoch()
        raw.append(accuracy())
        log(f"    epoch {e + 1}: test {raw[-1]:.4f}  {time.time() - t0:.0f}s")
        if probe:
            log(probe_report(layers, X[probe_idx], targets[probe_idx], cfg, probe_rng))
    final = raw[-1]
    if cfg["consolidate"] == "on":
        for layer in layers:
            layer.consolidate()
        layers[-1].start_tally()
        epoch(train_hidden=False)  # refit the readout to the consolidated hidden code
        layers[-1].consolidate()
        final = accuracy()
    return final, raw


def losses(layers, margins_out, targets):
    """Per sample: cross-entropy (at the hinge's scale) and the number of hinge violations."""
    out = layers[-1]
    logits = margins_out / (out.window / 3)
    logits = logits - logits.max(1, keepdims=True)
    ce = -(targets * (logits - np.log(np.exp(logits).sum(1, keepdims=True)))).sum(1)
    right = (margins_out >= 0) == targets
    violations = (~(right & (distance(margins_out) >= out.hinge))).sum(1)
    return ce, violations


def probe_report(layers, x, targets, cfg, rng, units=128):
    """
    For each hidden layer: flip `units` random units on every probe sample, propagate the flip up,
    and compare the change in the loss with the layer's signal (rule signal: what cfg passes down;
    STE: the gradient of cross-entropy with the straight-through step).
    """
    margins = forward(layers, x)
    q = signals(layers, x, margins, targets, cfg)
    g = signals(layers, x, margins, targets, dict(cfg, loss="ce", down="ste"))
    ce0, v0 = losses(layers, margins[-1], targets)
    lines = []
    for k in range(len(layers) - 1):
        code = margins[k] >= 0
        picked = rng.choice(code.shape[1], min(units, code.shape[1]), replace=False)
        gain_ce, gain_v, rule, ste = [], [], [], []
        for j in picked:
            flipped = code.copy()
            flipped[:, j] = ~flipped[:, j]
            up = flipped
            for layer in layers[k + 1:]:
                m = layer.margins(up)
                up = m >= 0
            ce1, v1 = losses(layers, m, targets)
            on = np.where(code[:, j], 1, -1)  # the flip turns the unit off where it was on
            # > 0: turning the unit on is better
            gain_ce.append(-(ce1 - ce0) * -on)
            gain_v.append(-(v1 - v0) * -on)
            rule.append(q[k][:, j])
            ste.append(g[k][:, j])
        gain_ce, gain_v, rule, ste = (np.concatenate(a) for a in (gain_ce, gain_v, rule, ste))

        def agree(signal, gain):
            both = (signal != 0) & (gain != 0)
            return np.mean(np.sign(signal[both]) == np.sign(gain[both])) if both.any() else np.nan

        def corr(a, b):
            return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan

        moved = np.mean(gain_v != 0)
        lines.append(
            f"      layer {k + 1}: flips that change violations {moved:.3f}, CE {np.mean(gain_ce != 0):.3f} | "
            f"rule signal nonzero {np.mean(rule != 0):.3f}, sign agrees with violations {agree(rule, gain_v):.3f}, "
            f"CE {agree(rule, gain_ce):.3f}, corr CE {corr(rule, gain_ce):.3f} | "
            f"STE agrees CE {agree(ste, gain_ce):.3f}, corr CE {corr(ste, gain_ce):.3f} | "
            f"rule·STE corr {corr(rule, ste):.3f} | code active {code.mean():.2f}")
    return "\n".join(lines)


def load(name):
    """Data and the layer settings it uses: Fashion-MNIST's from benchmarks/run.py, parity's CounterDense defaults."""
    if name == "fashion":
        return load_fashion_mnist(), dict(connect_level=16, counter_max=128, init_range=24, threshold_level=1)
    assert name.startswith("parity"), name
    return load_parity(int(name[6:])), {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ste", action="store_true", help="start from gradient descent instead of the rule")
    parser.add_argument("--data", default="fashion")
    parser.add_argument("--hidden", default="1024", help="hidden layer sizes, comma separated")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("parts", nargs="*", help="name=value")
    args = parser.parse_args()
    cfg = dict(STE if args.ste else RULE)
    for part in args.parts:
        name, value = part.split("=")
        names = ["gate_hidden", "gate_output"] if name == "gate" else [name]
        for name in names:
            assert name in cfg, f"unknown part {name}"
            cfg[name] = value
    (X, y, Xt, yt), layer_kwargs = load(args.data)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    final, raw = train(cfg, X, y, Xt, yt, hidden=hidden, seed=args.seed, epochs=args.epochs, lr=args.lr,
                       batch=args.batch, layer_kwargs=layer_kwargs, probe=args.probe)
    print(f"RESULT {args.data} {args.hidden} {' '.join(f'{k}={v}' for k, v in cfg.items())} seed={args.seed} "
          f"lr={args.lr}: final {final:.4f}  raw {' '.join(f'{a:.3f}' for a in raw[-10:])}  best raw {max(raw):.3f}", flush=True)


if __name__ == "__main__":
    main()
