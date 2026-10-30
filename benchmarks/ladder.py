"""
From gradient descent to the CounterDense rule, one part at a time (README, "The ladder").

One hidden layer on Fashion-MNIST, the same discrete net throughout (ternary synapses from counters,
binary units, thresholds), trained by a rule assembled from parts. Each part has a gradient (STE)
version and a rule version; all rule parts are CounterDense, all STE parts are straight-through
gradient descent with Adam. Everything is in counter units: a synapse exists while |counter| >= connect_level.

Parts (STE / rule):
  opt          adam: Adam on the counters / sgd: the signal itself is the step (expected size of the rule's votes)
  loss         ce: softmax cross-entropy on the output margins / hinge: CounterDense.output_pressure
  down         ste: pressure on a hidden unit = sum of output pressure over its synapses /
               rule: only outputs within the window pass it, 1 / (distance + 1) each, divided by sqrt(reach)
  gate_hidden  ste: every unit within the window of its threshold learns /
               rule: minimal disturbance, only flips at distance 0;
               also flip<K>: flips at distance <= K, near<K>: any vote at distance <= K
  gate_output  ste: every output with pressure learns / rule: flips, and stabilizing below distance 2
  (gate=... sets both)
  discrete     float: counters move by the signal / counters: binary votes with probability |signal|
  inactive     x: only active inputs move (the gradient of x @ w) / pm: inactive inputs move the other way (2x - 1)
  homeostasis  off / on (hidden units stuck for 256 samples move their threshold to their mean input sum)
  consolidate  off / on (average the last epoch, refit the readout for one epoch)
  hidden       learn (both) / frozen: the hidden layer stays at its random start (a reference, not a part)

usage: python -m benchmarks.ladder [--seed S] [--epochs E] [--lr LR] name=value ...
  e.g. python -m benchmarks.ladder              (the rule)
       python -m benchmarks.ladder --ste        (gradient descent)
       python -m benchmarks.ladder --ste opt=sgd loss=hinge
"""
import argparse
import time

import numpy as np

from benchmarks.datasets import load_fashion_mnist

RULE = dict(opt="sgd", loss="hinge", down="rule", gate_hidden="rule", gate_output="rule", discrete="counters",
            inactive="pm", homeostasis="on", consolidate="on", hidden="learn")
STE = dict(opt="adam", loss="ce", down="ste", gate_hidden="ste", gate_output="ste", discrete="float",
           inactive="x", homeostasis="off", consolidate="off", hidden="learn")

HOMEOSTASIS_RUN, HOMEOSTASIS_RATE = 256, 1.0 / 64


class Layer:
    def __init__(self, n_in, n_out, connect_level, counter_max, init_range, window, rng):
        self.c, self.cmax, self.window = connect_level, counter_max, window
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
            step = np.trunc(self.threshold_counters)  # threshold_level 1, as on Fashion-MNIST
            self.thresholds += step
            self.threshold_counters -= step
        else:
            delta, t_delta = signal.T @ x_code, -signal.sum(0)
            if cfg["opt"] == "adam":
                self.steps += 1
                delta, t_delta = self._adam(delta, 0), self._adam(t_delta, 2)
            self.counters = np.clip(self.counters + lr * delta, -self.cmax, self.cmax)
            self.thresholds += lr * t_delta
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


def output_signal(margins, targets, cfg, hinge, window):
    if cfg["loss"] == "ce":
        logits = margins / (window / 3)  # the hinge's scale
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return targets - p
    right = (margins >= 0) == targets
    return np.where(right & (distance(margins) >= hinge), 0, np.where(targets, 1, -1)).astype(np.float32)


def pressure_down(q, margins, layer, cfg):
    if cfg["down"] == "ste":
        return q @ layer.weights
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
    elif kind.startswith("flip"):
        may = flip & (d <= int(kind[4:]))
    elif kind.startswith("near"):
        may = d <= int(kind[4:])
    else:
        raise ValueError(kind)
    return np.where((q != 0) & may, q, 0)


def train(cfg, X, y, Xt, yt, seed=0, epochs=5, hidden=1024, lr=None, batch=32, log=print):
    rng = np.random.default_rng(seed)
    if lr is None:
        lr = 0.5 if cfg["opt"] == "adam" else 1.0
    n_in = X.shape[1]
    # the Fashion-MNIST layer settings (benchmarks/run.py) without the slow hidden layer; windows from fan-in
    h = Layer(n_in, hidden, 16, 128, 24, 40, rng)
    o = Layer(hidden, 10, 16, 128, 24, 32, rng)
    hinge = 11
    targets = (np.arange(10) == y[:, None]).astype(np.float32)

    def accuracy():
        return np.mean(np.argmax(o.margins(h.margins(Xt) >= 0), 1) == yt)

    def epoch(train_hidden=True):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            x = X[idx]
            mh = h.margins(x)
            code = mh >= 0
            mo = o.margins(code)
            q = output_signal(mo, targets[idx], cfg, hinge, o.window)
            if train_hidden and cfg["hidden"] == "learn":
                qh = pressure_down(q, mo, o, cfg)
                if cfg["homeostasis"] == "on":  # on the margins as computed, before this update's threshold votes
                    h.homeostasis(mh)
                h.update(x, gate(qh, mh, h, cfg, True), cfg, lr, rng)
            o.update(code, gate(q, mo, o, cfg, False), cfg, lr, rng)

    t0 = time.time()
    raw = []
    for e in range(epochs):
        if e == epochs - 1 and cfg["consolidate"] == "on":
            h.start_tally()
            o.start_tally()
        epoch()
        raw.append(accuracy())
        log(f"    epoch {e + 1}: test {raw[-1]:.4f}  {time.time() - t0:.0f}s")
    final = raw[-1]
    if cfg["consolidate"] == "on":
        h.consolidate()
        o.consolidate()
        o.start_tally()
        epoch(train_hidden=False)  # refit the readout to the consolidated hidden code
        o.consolidate()
        final = accuracy()
    return final, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ste", action="store_true", help="start from gradient descent instead of the rule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("parts", nargs="*", help="name=value")
    args = parser.parse_args()
    cfg = dict(STE if args.ste else RULE)
    for part in args.parts:
        name, value = part.split("=")
        names = ["gate_hidden", "gate_output"] if name == "gate" else [name]
        for name in names:
            assert name in cfg, f"unknown part {name}"
            cfg[name] = value
    X, y, Xt, yt = load_fashion_mnist()
    final, raw = train(cfg, X, y, Xt, yt, seed=args.seed, epochs=args.epochs, lr=args.lr)
    print(f"RESULT {' '.join(f'{k}={v}' for k, v in cfg.items())} seed={args.seed} lr={args.lr}: "
          f"final {final:.4f}  raw {' '.join(f'{a:.3f}' for a in raw)}", flush=True)


if __name__ == "__main__":
    main()
