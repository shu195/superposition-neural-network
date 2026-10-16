# Superposition Neural Network (FSNN)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)

**Author**: Shubham Barmase  
*Independent Researcher* (`barmase32@gmail.com`)  
**Research Paper**: [`paper/paper.md`](paper/paper.md)

---

## Overview

Classical connectionist artificial neural networks model synaptic transmission through real scalar affine projections followed by monotonic activation functions:

$$
z = \sum_{j=1}^D w_j x_j + b, \quad y = \sigma(z)
$$

This formulation lacks phase coherence, wave dispersion, and destructive interference dynamics, imposing foundational limitations such as the linear separability constraint of single-layer perceptrons.

This repository provides the reference implementation and empirical benchmarks for the **Fused Superposition Neural Network (FSNN)**. FSNN integrates the linear wave superposition principle ($\Psi = \sum W s$) with saturable Kerr non-linear absorption ($f(\Psi) = \tanh(\lvert\Psi\rvert)\frac{\Psi}{\lvert\Psi\rvert}$) and non-linear intensity detection ($y = \sigma(\alpha \lvert\Psi\rvert^2 + \dots)$).

---

## Mathematical Formulation

### 1. Fused Superposition Neuron

Each synapse possesses two physical properties:
* **Synaptic Conductance / Gain**: $w_{kj} \in \mathbb{R}$
* **Synaptic Phase Retardation**: $\phi_{kj} \in [-\pi, \pi]$

Represented as a complex transmission coefficient:

$$
W_{kj} = w_{kj} e^{i \phi_{kj}} \in \mathbb{C}
$$

Input wave states are represented as complex phasors:

$$
s_j = x_j e^{i \theta_j} \in \mathbb{C}
$$

The linear wave superposition at the soma is:

$$
\Psi_k = \sum_{j=1}^D W_{kj} s_j + B_k = \sum_{j=1}^D w_{kj} x_j e^{i(\theta_j + \phi_{kj})} + B_k
$$

Quadrature decomposition:

$$
\text{Re}(\Psi_k) = \sum_{j=1}^D w_{kj} x_j \cos(\theta_j + \phi_{kj}) + b_k \cos\phi_{b, k}
$$

$$
\text{Im}(\Psi_k) = \sum_{j=1}^D w_{kj} x_j \sin(\theta_j + \phi_{kj}) + b_k \sin\phi_{b, k}
$$

### 2. Multi-Layer Kerr Saturation & Intensity Detection

For hidden layers, wave amplitudes are constrained using phase-preserving saturable absorption (optical Kerr effect):

$$
f(\Psi_k) = \tanh(\lvert\Psi_k\rvert) \frac{\Psi_k}{\lvert\Psi_k\rvert} = \tanh(R_k) e^{i \Phi_k}
$$

For classification heads, energy intensity detection yields:

$$
I_k = \lvert\Psi_k\rvert^2 = \text{Re}(\Psi_k)^2 + \text{Im}(\Psi_k)^2
$$

$$
y_k = \sigma(\alpha I_k + \beta \text{Re}(\Psi_k) + c_k)
$$

Because $I_k$ contains harmonic interference cross-terms ($\cos(\Delta \theta_{jl})$), a single Fused Superposition Neuron can separate non-linear and closed manifolds without hidden layers.

---

## Empirical Benchmark Results

### Summary Table

| Benchmark Task | Input Domain | Classical Real MLP | Fused Superposition Net | Relative SNN Advantage |
| :--- | :--- | :---: | :---: | :--- |
| **XOR Problem** | Non-linear logic | $50.0\%$ (Fails) | **$100.0\%$ (0 hidden layers)** | **Solves with single neuron** |
| **Concentric Rings** | Radially closed | $55.3\%$ (Fails) | **$100.0\%$ (0 hidden layers)** | **$+44.7\%$ accuracy gain** |
| **Two-Spirals Manifold** | Phase-rotating | $100\%$ ($1,185$ params) | **$100\%$ ($675$ params)** | **$43.0\%$ fewer parameters, $5.5\times$ lower loss** |
| **Phased Wave Array** | Coherent physical wave | $88.2\%$ ($3,169$ params) | **$93.8\%$ ($1,091$ params)** | **$+5.60\%$ test accuracy, $65.6\%$ fewer parameters** |
| **Full MNIST Digits** | Spatial pixel intensity | **$97.31\%$** ($101\text{k}$ params) | $96.55\%$ ($203\text{k}$ params) | **Real MLP marginally leads by $+0.76\%$** |

---

### Detailed Findings

#### 1. Single-Neuron XOR & Concentric Circles
A single classical neuron is bounded by the linear hyperplane constraint ($50\text{--}75\%$). A single Fused Superposition Neuron uses destructive phase cancellation to achieve **$100.00\%$ accuracy** with zero hidden layers.

#### 2. Two-Spirals Non-Linear Manifold
The Fused Superposition Network untangles the interleaved spirals with **$675$ parameters** compared to $1,185$ parameters for the real MLP ($43.0\%$ parameter reduction), while converging to a $5.5\times$ lower loss ($0.0002$ vs $0.0011$).

#### 3. 4-Channel Phased Array Detection
On RF/acoustic phased array wave detection ($64$ features across 4 receiving channels), the Superposition Net uses coherent phase beamforming to achieve **$93.80\%$ test accuracy** ($+5.60\%$ higher than standard MLP) with **$65.6\%$ fewer parameters**.

#### 4. Critical Phase Decoherence Threshold
When subjected to stochastic synaptic phase noise $\phi \to \phi + \mathcal{N}(0, \sigma^2)$:
* $\sigma \le 2.9^\circ$ ($0.05\text{ rad}$): $93.5\%$ accuracy.
* $\sigma \approx 20.1^\circ$ ($0.35\text{ rad}$): **Critical breakdown threshold ($\sigma_{\text{crit}}$)** where interference cancellation breaks down and accuracy drops to random chance ($50.78\%$).

#### 5. Blind Continuous Attractor Identification
Feeding blind time-series telemetry ($t, x$) from the continuous-time superposition neuron into `math_solver.exe` yielded:
* Time-domain closed-form $x(t) = f(t)$: Unsolved (Non-elementary elliptic flow).
* Discovered Phase-Space ODE: $\ddot{x} = -4.0000 x - 0.2000 \dot{x} - 1.5000 x^3$ ($R^2 = 0.999996$, Extrapolation NRMSE $= 7.10 \times 10^{-6}$, Latency $= 17.4\text{ ms}$).

---

## Repository Structure

```
superposition-neural-network/
├── README.md                          # Repository documentation & benchmarks
├── LICENSE                            # MIT License
├── .gitignore                         # Build and cache ignore patterns
├── paper/
│   └── paper.md                       # Full academic research paper
├── src/
│   └── superposition_neuron.py        # Core PyTorch module (FSN, Layers, Networks)
├── benchmarks/
│   ├── run_xor.py                     # Single-neuron XOR benchmark
│   ├── run_concentric_and_spirals.py  # Concentric circles, two-spirals & ODE discovery
│   ├── run_phased_array.py            # Phased-array wave classification benchmark
│   ├── run_ablation_and_noise.py      # Synaptic ablation & phase decoherence sweep
│   └── run_mnist.py                   # Full MNIST digit comparison
└── data/
    └── superposition_neuron_telemetry.csv # 1,000-point continuous ODE trajectory
```

---

## Quick Start & Reproducibility

### Prerequisites
* Python 3.10+
* PyTorch 2.0+
* NumPy

```bash
pip install torch numpy
```

### Running the Benchmarks

```bash
# 1. Test single-neuron XOR resolution
python benchmarks/run_xor.py

# 2. Run concentric circles and two-spirals benchmarks
python benchmarks/run_concentric_and_spirals.py

# 3. Run phased array coherent wave classification
python benchmarks/run_phased_array.py

# 4. Run synaptic mechanism ablation and phase jitter sweep
python benchmarks/run_ablation_and_noise.py

# 5. Run full MNIST digit benchmark
python benchmarks/run_mnist.py
```

---

## Citation

```bibtex
@article{barmase2026superposition,
  title={Empirical Investigation of Neural Networks Based on the Physical Wave Superposition Principle},
  author={Barmase, Shubham},
  journal={Independent Research Archive},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
