# Predictive Coding and Binary Neural Networks: Analysis, Plans, and Integration

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Predictive Coding Theory In Depth](#predictive-coding-theory-in-depth)
   - [Historical Context](#historical-context)
   - [Core Principles](#core-principles)
   - [Neural Architecture](#neural-architecture)
   - [Mathematical Framework](#mathematical-framework)
   - [Relationship to Backpropagation](#relationship-to-backpropagation)
3. [This Project: Binary Sparse Neural Networks](#this-project-binary-sparse-neural-networks)
   - [Architecture Overview](#architecture-overview)
   - [Learning Mechanism](#learning-mechanism)
   - [Error Signal Propagation](#error-signal-propagation)
4. [Deep Comparison: Predictive Coding vs BNN](#deep-comparison-predictive-coding-vs-bnn)
   - [Structural Parallels](#structural-parallels)
   - [Key Divergences](#key-divergences)
   - [Convergence Points](#convergence-points)
5. [Proposed Hybrid: Predictive Binary Network (PBN)](#proposed-hybrid-predictive-binary-network-pbn)
   - [Architecture Design](#architecture-design)
   - [Prediction and Error Computation](#prediction-and-error-computation)
   - [Learning Rules](#learning-rules)
   - [Advantages Over Each Approach Alone](#advantages-over-each-approach-alone)
6. [Implementation Plan](#implementation-plan)
   - [Phase 1: Predictive Coding Layer (POC)](#phase-1-predictive-coding-layer-poc)
   - [Phase 2: Hierarchical Predictive Binary Network](#phase-2-hierarchical-predictive-binary-network)
   - [Phase 3: Convolutional Predictive Binary Network](#phase-3-convolutional-predictive-binary-network)
   - [Phase 4: Benchmarking and Evaluation](#phase-4-benchmarking-and-evaluation)
7. [POC Code Overview](#poc-code-overview)
8. [References](#references)

---

## Executive Summary

This document explores the deep connections between **predictive coding** — a leading
neuroscience theory of brain function — and the **Binary Neural Network (BNN)** approach
implemented in this repository. Both share a core commitment to biologically plausible,
local learning rules that avoid the problems of backpropagation. We argue that combining
predictive coding's hierarchical generative model with BNN's binary/sparse architecture
produces a powerful hybrid we call the **Predictive Binary Network (PBN)**.

The key insight is that the BNN project's dual error signals (`too_much` and `not_enough`)
are conceptually equivalent to the signed prediction errors of predictive coding, and the
BNN's local synapse modification rules are a binary/sparse analog of Hebbian learning at
prediction error sites. By explicitly adding a top-down prediction pathway and framing
learning as prediction error minimization, we can give the BNN architecture a principled
theoretical foundation while retaining its computational advantages.

---

## Predictive Coding Theory In Depth

### Historical Context

Predictive coding has roots in Helmholtz's 1860s concept of **unconscious inference** — the
idea that perception is not passive reception of stimuli but an active process of
constructing hypotheses about the world. The modern computational theory was crystallized
by **Rao and Ballard (1999)**, who proposed a hierarchical generative model of visual
processing where higher cortical areas send predictions downward and receive prediction
errors from below.

The theory was significantly extended by **Karl Friston** through the **Free Energy
Principle** (2005-2010), which provides a unified mathematical framework casting perception,
action, and learning as minimization of variational free energy — an upper bound on
surprise. More recently, **Millidge, Seth, and Buckley (2021-2022)** have shown that
predictive coding networks can approximate backpropagation and serve as general-purpose
machine learning algorithms.

Key papers:
- Rao & Ballard (1999): "Predictive coding in the visual cortex"
- Friston (2005): "A theory of cortical responses" (free energy principle)
- Whittington & Bogacz (2017): "Approximation of the error backpropagation algorithm in a
  predictive coding network"
- Millidge et al. (2022): "Predictive Coding Approximates Backprop Along Arbitrary
  Computation Graphs" (arXiv:2202.09467)
- Millidge et al. (2021): "Predictive Coding: a Theoretical and Experimental Review"
  (arXiv:2107.12979)

### Core Principles

1. **Hierarchical Generative Model**: The brain is organized as a hierarchy of levels.
   Each level maintains an internal representation (or "belief") about the causes of
   its inputs. Higher levels represent more abstract, slowly-changing features; lower
   levels represent concrete, rapidly-changing sensory details.

2. **Top-Down Predictions**: Each level generates a prediction of the expected activity
   at the level below, using a learned generative model. These predictions flow top-down
   through feedback connections.

3. **Bottom-Up Prediction Errors**: The actual bottom-up input is compared to the
   top-down prediction. The difference — the **prediction error** — is what propagates
   upward, not the raw sensory data.

4. **Prediction Error Minimization**: The system's goal is to minimize prediction errors.
   This is achieved through two complementary mechanisms:
   - **Perceptual inference**: Updating internal representations to better explain the
     current input (fast, within-trial dynamics).
   - **Learning**: Updating the generative model parameters (weights) to make better
     predictions in the future (slow, across-trial plasticity).

5. **Local Learning**: All weight updates are local — they depend only on pre-synaptic
   and post-synaptic activity (or prediction errors) at the same site. No global error
   signal needs to be propagated backward through the network.

6. **Precision Weighting**: Prediction errors are weighted by their estimated precision
   (inverse variance / reliability). This implements a form of attention — more reliable
   signals get more weight in updating beliefs.

### Neural Architecture

In the brain, predictive coding is thought to be implemented through two distinct
populations of neurons at each level of the cortical hierarchy:

```
Level L+1:  [Representation/Value Neurons]
                     |
                     | (top-down prediction via feedback connections)
                     v
Level L:    [Error Neurons] = (bottom-up input) - (top-down prediction)
                     |
                     | (prediction error sent upward via feedforward connections)
                     v
Level L+1:  [Representation/Value Neurons] (updated by prediction error)
```

- **Representation neurons** (deep pyramidal cells, layer 5/6): Encode the current
  best estimate of the causes at this level. Their activity is updated to minimize the
  prediction errors they receive.

- **Error neurons** (superficial pyramidal cells, layer 2/3): Compute the difference
  between the bottom-up input and the top-down prediction. They have sparse activity
  (most of the time predictions are accurate, so errors are small).

This maps onto known cortical anatomy:
- Feedback connections (top-down) project from deep layers to both superficial and deep
  layers of the lower area.
- Feedforward connections (bottom-up, carrying prediction errors) project from superficial
  layers to layer 4 (the granular layer) of the higher area.

### Mathematical Framework

For a hierarchical network with L levels, let:
- `x_l` = representation (value) at level l
- `e_l` = prediction error at level l
- `f_l()` = generative function (prediction function) from level l+1 to level l
- `θ_l` = parameters (weights) of the generative function at level l

The prediction error at each level is:
```
e_l = x_l - f_l(x_{l+1})
```

The total energy (variational free energy) to minimize:
```
F = Σ_l  ||e_l||² / (2 * σ_l²)
  = Σ_l  ||x_l - f_l(x_{l+1})||² / (2 * σ_l²)
```

**Inference** (fast dynamics): Update representations to minimize F:
```
dx_l/dt = -∂F/∂x_l = e_l - (∂f_{l-1}/∂x_l)^T * e_{l-1}
```
This means each representation node integrates:
1. Its own prediction error (bottom-up, from the level below)
2. A top-down signal derived from the error at the level above

**Learning** (slow dynamics): Update weights to minimize F:
```
Δθ_l ∝ -∂F/∂θ_l = e_l * (∂f_l/∂θ_l)
```
This is a local Hebbian-like rule: the weight change depends on the prediction error
at this level and the derivative of the prediction function w.r.t. the weight — both
locally available quantities.

### Relationship to Backpropagation

A landmark result (Whittington & Bogacz, 2017; Millidge et al., 2022) is that **predictive
coding at convergence approximates backpropagation**. Specifically:

1. If inference is run to equilibrium (prediction errors settle), the weight gradients
   computed by predictive coding equal the gradients computed by backpropagation.
2. Before convergence, predictive coding computes biased but still useful gradients.
3. Unlike backpropagation, predictive coding can work on **arbitrary computation graphs**
   (not just feedforward DAGs), can function as both a classifier and a generator, and
   requires only local computations.

This makes predictive coding a compelling biologically plausible alternative to
backpropagation — achieving similar performance while using only local learning rules.

---

## This Project: Binary Sparse Neural Networks

### Architecture Overview

The BNN project implements neural networks with three key characteristics:

1. **Binary activations**: Neurons output 0 or 1 (fire or don't fire).
2. **Binary/ternary weights**: Each synapse is either excitatory (+1), inhibitory (-1),
   or absent (0). Weights are stored as sparse index lists.
3. **Integer thresholds**: Each neuron has an integer activation threshold. If
   `(excitation_sum - inhibition_sum) >= threshold`, the neuron fires.

The architecture supports both dense layers (`bnn/dense.py`) and convolutional layers
(`bnn/__init__.py` with `ConvBNN`).

### Learning Mechanism

Learning is driven by two error signals at each neuron:
- **`too_much`**: The neuron fired but shouldn't have (false positive)
- **`not_enough`**: The neuron didn't fire but should have (false negative)

Corrections available:

| Error Type | Correction Options |
|---|---|
| `not_enough` | Add excitatory synapse to active input |
| `not_enough` | Remove inhibitory synapse from active input |
| `not_enough` | Decrease threshold |
| `too_much` | Add inhibitory synapse to active input |
| `too_much` | Remove excitatory synapse from active input |
| `too_much` | Increase threshold |

### Error Signal Propagation

Error signals propagate backward through existing synapses:

- If neuron has `too_much` error and has an active excitatory input, tell that input
  neuron it also has `too_much` error.
- If neuron has `too_much` error and has an inactive inhibitory input, tell that input
  neuron it has `not_enough` error.
- Symmetric rules for `not_enough` error.

This is a form of **credit assignment** — distributing blame/credit through the network
using only local information (existing synapse structure + input activity + error signals).

---

## Deep Comparison: Predictive Coding vs BNN

### Structural Parallels

| Aspect | Predictive Coding | BNN Project |
|---|---|---|
| **Error signals** | Prediction error (continuous, signed) | `too_much` / `not_enough` (continuous, unsigned pair) |
| **Signal direction** | Errors go bottom-up, predictions go top-down | Errors propagate top-down (backward), activations go bottom-up (forward) |
| **Learning locality** | Local: depends on pre/post-synaptic activity and local error | Local: depends on input activity and error signals |
| **Weight updates** | Hebbian-like: `Δw ∝ error * input` | Correlation-based: add excitatory synapse when input active AND `not_enough` error |
| **Biological plausibility** | High (two neuron types, cortical layers) | High (binary activations, sparse connectivity, local learning) |
| **Activation function** | Typically continuous (sigmoid, ReLU) | Binary step function |
| **Representation** | Dense, continuous | Sparse, binary |

### Key Divergences

1. **Prediction mechanism**: Predictive coding explicitly generates top-down predictions
   and computes their discrepancy with bottom-up input. The BNN project does NOT have an
   explicit prediction/generative pathway — errors are computed only from the supervised
   target at the output layer.

2. **Error semantics**: In predictive coding, the prediction error is `actual - predicted`,
   a signed quantity. In BNN, errors are split into two unsigned channels (`too_much` and
   `not_enough`), which together encode a signed error. This is actually closer to
   biological reality, where neurons cannot have negative firing rates — the brain likely
   uses separate populations for positive and negative errors.

3. **Inference dynamics**: Predictive coding performs iterative inference — representations
   are updated over multiple timesteps until prediction errors are minimized. BNN uses a
   single feedforward pass.

4. **Generative capability**: Predictive coding networks are inherently generative — they
   can generate data by running the model top-down. BNN networks are purely discriminative.

5. **Weight representation**: Predictive coding uses continuous weights. BNN uses
   binary/ternary weights stored as sparse index lists.

### Convergence Points

Despite these divergences, there are deep convergences:

1. **Both use local error-driven learning**: The fundamental learning signal is local —
   both approaches update weights based on local activity and local error signals.

2. **Both propagate errors through the network**: Both have mechanisms for distributing
   error information across layers, enabling credit assignment without global
   backpropagation.

3. **BNN's dual error channels match predictive coding's error decomposition**: The
   `too_much`/`not_enough` split in BNN mirrors the fact that prediction errors in
   biological predictive coding must be encoded by two populations (since neurons can't
   have negative rates). This is a feature, not a limitation.

4. **Both emphasize sparse representations**: Predictive coding predicts that representations
   should be sparse (only prediction errors are propagated; well-predicted inputs are
   suppressed). BNN explicitly uses sparse binary representations.

5. **Both are biologically inspired**: Both draw on neuroscience principles and aim for
   biological plausibility in their learning rules.

---

## Proposed Hybrid: Predictive Binary Network (PBN)

The proposed hybrid architecture — the **Predictive Binary Network (PBN)** — combines
the strengths of both approaches:

### Architecture Design

```
┌─────────────────────────────────────────────────────────┐
│                    PBN Architecture                     │
│                                                         │
│  Level L+1: [Binary Representation Neurons]             │
│                  │              ▲                        │
│    (top-down     │              │ (bottom-up             │
│     prediction)  │              │  prediction errors     │
│                  ▼              │  as too_much /         │
│  Level L:   [Prediction] ──► [Error    ] ──► [too_much] │
│             [Computed   ]    [Computed  ]     [not_enough│
│             [from L+1   ]    [= actual ]               ]│
│                              [- predicted]              │
│                  ▲                                       │
│                  │ (bottom-up actual activation)         │
│  Level L-1: [Binary Representation Neurons]             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

Each level of the PBN has:
1. **Representation neurons**: Binary neurons (as in current BNN) that encode the
   current best estimate of features at this level.
2. **A generative/prediction pathway**: A set of (sparse, binary) connections from the
   level above that predict what the activity at this level should be.
3. **Error computation**: Compares actual bottom-up activation with top-down prediction,
   producing `too_much` (predicted 0 but got 1) and `not_enough` (predicted 1 but got 0)
   error signals.

### Prediction and Error Computation

In a standard BNN forward pass:
```python
output = (excitation_sum - inhibition_sum) >= threshold  # Binary activation
```

In a PBN, we add a prediction step:
```python
# Forward pass (bottom-up): compute actual activations
actual = (excitation_sum - inhibition_sum) >= threshold

# Top-down prediction: higher layer predicts what this layer should look like
predicted = predict_from_above(higher_layer_activity)

# Prediction error (using binary error channels)
too_much  = actual & ~predicted   # Active but wasn't predicted → over-activation
not_enough = ~actual & predicted  # Inactive but was predicted → under-activation
```

The critical insight is that in a PBN, the error signals arise **at every layer** from
the discrepancy between bottom-up activation and top-down prediction — not just from
the supervised target at the output. This enables:
- **Unsupervised feature learning** at intermediate layers
- **Generative capability** (run predictions top-down without input)
- **More efficient credit assignment** (errors are local, not propagated from output)

### Learning Rules

The PBN uses modified BNN learning rules, now driven by prediction errors:

**Feedforward weights** (bottom-up, for computing activations):
Updated using the current BNN rule, but where the error signal at each layer comes
from the prediction error (not just propagated from the output layer).

**Feedback weights** (top-down, for generating predictions):
Updated using a Hebbian-like rule:
- If `not_enough` at this level and higher-level neuron is active: strengthen the
  feedback excitatory connection (so the prediction becomes more likely).
- If `too_much` at this level and higher-level neuron is active: strengthen the
  feedback inhibitory connection (so the prediction is suppressed).
- This is exactly the BNN learning rule applied to the feedback pathway.

**Thresholds**: Adjusted as in current BNN, but driven by prediction errors.

### Advantages Over Each Approach Alone

| Feature | BNN Only | PC Only | PBN (Hybrid) |
|---|---|---|---|
| Binary/sparse computation | ✓ | ✗ | ✓ |
| Biologically plausible learning | ✓ | ✓ | ✓ |
| Unsupervised feature learning | ✗ | ✓ | ✓ |
| Generative capability | ✗ | ✓ | ✓ |
| Layer-local error signals | Partial | ✓ | ✓ |
| Hardware efficiency | ✓ | ✗ | ✓ |
| Principled theoretical foundation | Partial | ✓ | ✓ |
| Works with sparse binary weights | ✓ | ✗ | ✓ |

---

## Implementation Plan

### Phase 1: Predictive Coding Layer (POC)

**Status: Implemented** (see `bnn/predictive.py`)

A `PredictiveCodingLayer` that extends the BNN `Layer` abstraction with:
- A feedback (generative) pathway: sparse binary connections from output to input
- Prediction computation: given output activity, predict expected input activity
- Prediction error computation: compare actual input with predicted input
- Combined learning: update both feedforward and feedback weights using local errors

This POC demonstrates the core idea on dense layers and validates that the combined
architecture trains correctly.

### Phase 2: Hierarchical Predictive Binary Network

Build a `PredictiveSequence` that chains PBN layers and:
- Runs iterative inference (multiple forward-backward passes) to settle prediction
  errors before weight updates
- Supports unsupervised pre-training (minimize prediction errors without labels)
- Supports supervised fine-tuning (add label-driven errors at output)

### Phase 3: Convolutional Predictive Binary Network

Extend the convolutional BNN (`ConvBNN`) with predictive coding:
- Top-down convolutional predictions (transpose convolutions for the feedback path)
- Layer-local prediction errors in the spatial domain
- Multi-scale feature learning

### Phase 4: Benchmarking and Evaluation

- Compare PBN vs BNN on MNIST classification accuracy
- Evaluate unsupervised feature learning quality
- Test generative capability (produce digits from learned representations)
- Measure convergence speed and computational efficiency
- Compare with standard predictive coding networks (continuous weights)

---

## POC Code Overview

The proof-of-concept is implemented in `bnn/predictive.py` and tested in
`bnn/predictive_tests.py`.

### `PredictiveCodingLayer`

A layer that maintains both:
- **Feedforward connections**: Sparse binary (excitatory/inhibitory) connections from
  input to output, exactly as in the existing `Dense` layer.
- **Feedback connections**: Sparse binary connections from output to input, used to
  generate top-down predictions.

The forward pass computes activations as usual. The backward pass now computes prediction
errors **locally** by comparing actual inputs with what the feedback pathway predicts.
These prediction errors augment the propagated error signals.

### `PredictiveSequence`

A sequence of `PredictiveCodingLayer`s that supports iterative inference: running
multiple forward-backward passes to allow prediction errors to settle before updating
weights. This implements the iterative message-passing dynamics of predictive coding.

See `bnn/predictive.py` for implementation and `bnn/predictive_tests.py` for tests.

---

## References

1. Rao, R.P.N. & Ballard, D.H. (1999). "Predictive coding in the visual cortex: a
   functional interpretation of some extra-classical receptive-field effects." *Nature
   Neuroscience*, 2(1), 79-87.

2. Friston, K. (2005). "A theory of cortical responses." *Philosophical Transactions
   of the Royal Society B*, 360(1456), 815-836.

3. Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature
   Reviews Neuroscience*, 11(2), 127-138.

4. Whittington, J.C.R. & Bogacz, R. (2017). "An Approximation of the Error
   Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic
   Plasticity." *Neural Computation*, 29(5), 1229-1262.

5. Millidge, B., Seth, A., & Buckley, C.L. (2022). "Predictive Coding Approximates
   Backprop Along Arbitrary Computation Graphs." arXiv:2202.09467.

6. Millidge, B., Seth, A., & Buckley, C.L. (2021). "Predictive Coding: a Theoretical
   and Experimental Review." arXiv:2107.12979.

7. Bastos, A.M., et al. (2012). "Canonical Microcircuits for Predictive Coding."
   *Neuron*, 76(4), 695-711.

8. Seth, A.K. (2013). "Interoceptive inference, emotion, and the embodied self."
   *Trends in Cognitive Sciences*, 17(11), 565-573.

9. Helmholtz, H. von (1867). *Handbuch der physiologischen Optik*. Leipzig: Voss.

10. McClelland, J.L. & Rumelhart, D.E. (1981). "An interactive activation model of
    context effects in letter perception." *Psychological Review*, 88, 375-407.
