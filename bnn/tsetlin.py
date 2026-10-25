"""
Multiclass Tsetlin Machine (Granmo 2018, https://arxiv.org/abs/1804.01508).

A fully discrete learner, kept here as a reference point for the BNN layers.
Each class owns a set of conjunctive clauses over literals (inputs and their
negations). Every literal in every clause has a Tsetlin automaton -- an integer
state in [1, 2 * n_states] -- deciding whether the literal is included (state >
n_states). Half the clauses vote for the class, half against. Learning only
increments or decrements automaton states, with probabilities set by how far the
class vote is from the target `T`.
"""

from typing import Optional

import numpy as np


class TsetlinMachine:
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_clauses: int = 100,
        T: int = 15,
        s: float = 3.9,
        n_states: int = 100,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            n_clauses: clauses per class, half positive and half negative.
            T: vote target -- a class with |vote| >= T stops receiving feedback.
            s: specificity -- higher s makes clauses include more literals.
            n_states: automaton states per action.
        """
        assert n_clauses % 2 == 0
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_clauses = n_clauses
        self.T = T
        self.s = s
        self.n_states = n_states
        self.rng = rng if rng is not None else np.random.default_rng()

        self.states = self.rng.integers(
            n_states, n_states + 2,
            size=(n_classes, n_clauses, 2 * n_features),
        ).astype(np.int16)
        self.polarity = np.where(np.arange(n_clauses) < n_clauses // 2, 1, -1).astype(np.int32)

    @staticmethod
    def _literals(X):
        return np.concatenate([X, ~X], axis=-1)

    def clause_outputs(self, X, training=False):
        """Shape (batch, n_classes, n_clauses)."""
        included = (self.states > self.n_states).reshape(-1, 2 * self.n_features)
        # a clause fires when none of its included literals is false
        false_literals = (~self._literals(X)).astype(np.float32)
        violations = false_literals @ included.T.astype(np.float32)
        out = (violations == 0).reshape(len(X), self.n_classes, self.n_clauses)
        if not training:
            # empty clauses would fire on everything -- they are only allowed to while learning
            out &= included.reshape(self.n_classes, self.n_clauses, -1).any(axis=-1)
        return out

    def votes(self, X):
        return self.clause_outputs(X) @ self.polarity

    def predict(self, X):
        return np.argmax(self.votes(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def fit_epoch(self, X, y):
        for i in self.rng.permutation(len(X)):
            self._learn(X[i], y[i])

    def _learn(self, x, y):
        literals = self._literals(x)
        other = self.rng.integers(self.n_classes - 1)
        other += other >= y
        for c, is_target in ((y, True), (other, False)):
            states = self.states[c]
            included = states > self.n_states
            out = ~(included & ~literals).any(axis=1)
            vote = np.clip(out @ self.polarity, -self.T, self.T)
            if is_target:
                p = (self.T - vote) / (2 * self.T)
            else:
                p = (self.T + vote) / (2 * self.T)
            selected = self.rng.random(self.n_clauses) < p
            towards_class = (self.polarity > 0) == is_target
            type_1 = selected & towards_class
            type_2 = selected & ~towards_class

            # Type I (recognize): firing clauses memorize the true literals, everything else slowly forgets.
            r = self.rng.random(states.shape)
            fired_true = out[:, None] & literals[None, :]
            reward = type_1[:, None] & fired_true & (r < (self.s - 1) / self.s)
            forget = type_1[:, None] & ~fired_true & (r < 1 / self.s)
            # Type II (reject): firing clauses include a false literal, so they stop firing here.
            reject = type_2[:, None] & out[:, None] & ~literals[None, :] & ~included

            delta = reward.astype(np.int16) + reject - forget
            self.states[c] = np.clip(states + delta, 1, 2 * self.n_states)
