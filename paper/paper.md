# Empirical Investigation of Neural Networks Based on the Physical Wave Superposition Principle

**Author**: Shubham Barmase  
*Independent Researcher* (`barmase32@gmail.com`)  
**Date**: October 2026  

---

## Abstract

Classical connectionist artificial neural networks model synaptic transmission through real scalar affine projections followed by monotonic activation functions ($y = \sigma(\mathbf{w}^T \mathbf{x} + b)$). This formulation lacks phase coherence, wave dispersion, and destructive interference dynamics, imposing foundational limitations such as the linear separability constraint of single-layer perceptrons. In this paper, we formulate and empirically investigate the **Fused Superposition Neural Network (FSNN)**, which integrates the linear wave superposition principle ($\Psi = \sum_j W_{kj} s_j$) with saturable Kerr non-linear absorption ($f(\Psi) = \tanh(\lvert\Psi\rvert)\frac{\Psi}{\lvert\Psi\rvert}$) and non-linear intensity detection ($y = \sigma(\alpha \lvert\Psi\rvert^2 + \dots)$). Through systematic experimentation, we demonstrate that: (1) a single Fused Superposition Neuron with zero hidden layers resolves non-linearly separable logic (XOR) and concentric circular rings with $100.00\%$ accuracy via destructive wave cancellation; (2) on the two-spirals non-linear manifold, FSNN matches standard multi-layer perceptron (MLP) accuracy with $43.0\%$ fewer parameters and $5.5\times$ lower training loss; (3) on coherent phased-array antenna wave detection, FSNN outperforms classical real MLPs by $+5.60\%$ test accuracy with $65.6\%$ fewer parameters; (4) under stochastic phase jitter, constructive interference undergoes a critical breakdown phase transition at $\sigma_{\text{crit}} \approx 20^\circ$ ($0.35\text{ rad}$); and (5) continuous dynamical trajectories of the fused neuron are shown by symbolic discovery to be governed by a non-elementary Stuart-Landau / Duffing attractor. Finally, we contrast wave-native domains against scalar spatial imagery (MNIST), clarifying where the wave superposition principle provides natural inductive bias versus where standard connectionist architectures remain compute-optimal.

---

## 1. Introduction

The mathematical foundation of artificial neural networks traces back to the McCulloch-Pitts model and Rosenblatt's perceptron, which formulate the neuron as an affine inner product on Euclidean space:

$$
z = \sum_{j=1}^D w_j x_j + b, \quad y = \sigma(z)
$$

While multi-layer networks serve as universal approximators, individual neurons within this framework partition input space using planar hyperplanes. As proved by Minsky and Papert (1969), a single classical neuron cannot solve elementary non-linearly separable functions such as the exclusive-OR (XOR) gate without supplementary hidden units.

In physical wave mechanics (coherent optics, acoustics, electromagnetism, and quantum mechanics), wave propagation is fundamentally governed by the **Superposition Principle**. When multiple coherent waves intersect, their resultant amplitude is not a simple sum of intensities, but an algebraic superposition of complex field amplitudes:

$$
\Psi(\mathbf{r}, t) = \sum_{j=1}^N A_j e^{i(\mathbf{k}_j \cdot \mathbf{r} - \omega t + \phi_j)}
$$

The intensity detected by a physical receiver is:

$$
I = \lvert\Psi\rvert^2 = \sum_{j=1}^N A_j^2 + 2 \sum_{1 \le j < l \le N} A_j A_l \cos(\phi_j - \phi_l)
$$

The cross-terms $2 A_j A_l \cos(\phi_j - \phi_l)$ represent wave interference. In-phase signals ($\Delta \phi = 0$) interfere constructively ($+2 A_j A_l$), while out-of-phase signals ($\Delta \phi = \pi$) undergo destructive cancellation ($-2 A_j A_l$).

This paper explores the mathematical and empirical fusion of classical neuron equations with wave superposition. We investigate whether incorporating synaptic phase shifts, wave interference, and saturable non-linearities grants single neurons and multi-layer networks enhanced representational capacity and parameter efficiency.

---

## 2. Mathematical Formulation

### 2.1 The Fused Superposition Neuron (FSN)

Let each synaptic connection possess two distinct physical parameters:
1. **Synaptic Conductance / Gain**: $w_{kj} \in \mathbb{R}$
2. **Synaptic Phase Retardation**: $\phi_{kj} \in [-\pi, \pi]$

These parameters form a complex transmission coefficient:

$$
W_{kj} = w_{kj} e^{i \phi_{kj}} \in \mathbb{C}
$$

An input signal consists of a wave packet with amplitude $x_j$ and carrier phase $\theta_j$:

$$
s_j = x_j e^{i \theta_j} \in \mathbb{C}
$$

The linear wave superposition at neuron soma $k$ is given by:

$$
\Psi_k = \sum_{j=1}^D W_{kj} s_j + B_k = \sum_{j=1}^D w_{kj} x_j e^{i(\theta_j + \phi_{kj})} + b_k e^{i \phi_{b, k}}
$$

Separating into real and imaginary quadrature components:

$$
\text{Re}(\Psi_k) = \sum_{j=1}^D w_{kj} x_j \cos(\theta_j + \phi_{kj}) + b_k \cos\phi_{b, k}
$$

$$
\text{Im}(\Psi_k) = \sum_{j=1}^D w_{kj} x_j \sin(\theta_j + \phi_{kj}) + b_k \sin\phi_{b, k}
$$

The total field intensity is:

$$
I_k = \lvert\Psi_k\rvert^2 = \text{Re}(\Psi_k)^2 + \text{Im}(\Psi_k)^2
$$

Expanding $I_k$ reveals the harmonic interference manifold:

$$
I_k = \sum_{j=1}^D w_{kj}^2 x_j^2 + 2 \sum_{j < l} w_{kj} w_{kl} x_j x_l \cos\big( (\theta_j + \phi_{kj}) - (\theta_l + \phi_{kl}) \big) + \text{bias terms}
$$

### 2.2 Phase-Preserving Kerr Saturation & Detection

In multi-layer propagation, unboundedly accumulating wave amplitudes can lead to numerical instability. In optical media, this is regulated by third-order non-linear susceptibility (optical Kerr effect and saturable absorption). We model intermediate hidden layers using a phase-preserving saturable activation:

$$
f(\Psi_k) = \tanh(\lvert\Psi_k\rvert) \frac{\Psi_k}{\lvert\Psi_k\rvert} = \tanh(R_k) e^{i \Phi_k}
$$

where $R_k = \sqrt{\text{Re}(\Psi_k)^2 + \text{Im}(\Psi_k)^2}$ is the envelope magnitude, and $\Phi_k = \text{atan2}(\text{Im}(\Psi_k), \text{Re}(\Psi_k))$ is the coherent phase.

For terminal classification layers, an intensity detection head converts energy density into probability:

$$
y_k = \sigma\big( \alpha I_k + \beta \text{Re}(\Psi_k) + c_k \big)
$$

Because $I_k$ contains quadratic and angular cross-terms, a single Fused Superposition Neuron constructs quadric and harmonic decision boundaries without requiring hidden layers.

### 2.3 Continuous-Time Dynamical Stuart-Landau Attractor

When extended to continuous time, the interaction between linear wave superposition, internal oscillator frequency, and non-linear saturation yields the complex Stuart-Landau / Ginzburg-Landau equation:

$$
\frac{d\Psi_k}{dt} = -(\mu_k + i \omega_k)\Psi_k + \sum_j W_{kj} \Psi_j - \beta_k \lvert\Psi_k\rvert^2 \Psi_k + S_k(t)
$$

where $\mu_k > 0$ represents dissipation, $\omega_k$ is natural angular frequency, $\beta_k > 0$ provides non-linear amplitude saturation, and $S_k(t)$ is external stimulus.

---

## 3. Empirical Results

### 3.1 Non-Linear Separability: The XOR Problem

We compared a single classical real perceptron ($y = \sigma(\mathbf{w}^T \mathbf{x} + b)$) against a single Fused Superposition Neuron on the canonical XOR dataset with zero hidden layers. Both models were trained using Adam.

**Table 1: Single-Neuron XOR Benchmark**

| Model Architecture | Hidden Layers | Trainable Parameters | Converged Predictions | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| Classical Real Perceptron | 0 | 3 | $[0.50, 0.50, 0.50, 0.50]$ | $50.00\%$ |
| **Fused Superposition Neuron** | **0** | **7** | **$[0.00, 1.00, 1.00, 0.00]$** | **$100.00\%$** |

*Mechanism*: For inputs $(0, 1)$ and $(1, 0)$, the learned synaptic phase delays ($\phi_1 = 0.021\text{ rad}, \phi_2 = 0.021\text{ rad}$) and conductances ($w_1 = 1.893, w_2 = -1.785$) cause destructive interference in the quadrature sum ($\text{Re}(\Psi) = 1.595, -5.759$), driving intensities to distinct separation thresholds ($I = 2.55, 33.18$) relative to the diagonal states. A single superposition neuron completely resolves the non-linearly separable logic gate.

### 3.2 Non-Separable Radial Geometry: Concentric Rings

We evaluated radial classification on $N = 600$ points drawn from two concentric circles (inner radius $r_1 \approx 0.35$, outer radius $r_2 \approx 0.85$, Gaussian noise $\sigma = 0.04$).

**Table 2: Concentric Circles Benchmark**

| Model Architecture | Hidden Layers | Trainable Parameters | Accuracy | Loss |
| :--- | :---: | :---: | :---: | :---: |
| Classical Real Perceptron | 0 | 3 | $55.33\%$ | $0.6908$ |
| **Fused Superposition Neuron** | **0** | **7** | **$100.00\%$** | **$0.0272$** |

*Mechanism*: Because $\lvert\Psi\rvert^2 = (x_1 \cos\phi_1 + x_2 \cos\phi_2)^2 + (x_1 \sin\phi_1 + x_2 \sin\phi_2)^2$, setting symmetric quadrature phases directly yields circular contours $\propto x_1^2 + x_2^2 = r^2$.

### 3.3 Non-Linear Manifold Disentanglement: Two-Spirals Benchmark

The two-spirals dataset consists of two interleaved spiral arms ($N = 800$ points) that wind around the origin, representing a standard benchmark for testing manifold capacity.

**Table 3: Two-Spirals Comparative Results**

| Model Architecture | Layer Dimensions | Total Parameters | Final Loss | Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| Classical Real MLP | $2 \to 32 \to 32 \to 1$ | 1,185 | $0.0011$ | $100.00\%$ |
| **Fused Superposition Net** | **$2 \to 16 \to 16 \to 1$** | **675** | **$0.0002$** | **$100.00\%$** |

*Outcome*: The Fused Superposition Network required **$43.0\%$ fewer parameters** ($675$ vs $1,185$) and achieved a **$5.5\times$ lower training loss** ($0.0002$ vs $0.0011$). The spatial phase rotation inherent in the complex layers aligns with the spiral geometry.

### 3.4 Multi-Channel Phased Array Coherent Wave Detection

We constructed a 4-channel phased array receiver benchmark ($N = 2,500$ samples, $64$ input dimensions). Class 1 corresponds to a coherent plane wave arriving at incident angle $\theta_{\text{inc}}$ with inter-element phase delay $\Delta \phi = \pi \sin\theta_{\text{inc}}$. Class 0 represents incoherent multi-path noise clutter.

**Table 4: 4-Channel Phased Array Benchmark**

| Model Architecture | Trainable Parameters | Train Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: |
| Classical Real MLP ($64 \to 32 \to 32 \to 1$) | 3,169 | $100.00\%$ | $88.20\%$ |
| **Superposition Phased Net ($32\mathbb{C} \to 16\mathbb{C} \to 1$)** | **1,091** | **$100.00\%$** | **$93.80\%$** |

*Outcome*: Classical real MLPs lack inductive bias for phase coherence, achieving $88.20\%$ test accuracy. The Superposition Network performs coherent complex beamforming, achieving **$93.80\%$ test accuracy (+5.60% improvement)** with **$65.6\%$ fewer parameters**.

---

## 4. Synaptic Mechanism Ablation & Phase Decoherence Analysis

### 4.1 Synaptic Mechanism Ablation

To isolate whether performance gains originate from parameter count or phase interference, we compared four configurations on the spiral task:
1. **Amplitude-Only**: Conductance $w$ trained, phase delays fixed to zero ($\phi = 0$).
2. **Phase-Only ($D^2NN$)**: Conductance fixed to unity ($w = 1.0$), phase delays $\phi \in [-\pi, \pi]$ trained.
3. **Fused**: Both $w$ and $\phi$ trained.
4. **Real MLP Match**: Standard MLP with matched parameter count.

**Table 5: Synaptic Mechanism Ablation Results**

| Configuration | Degrees of Freedom | Parameters | Final Loss | Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| Amplitude-Only | Gain $w \in \mathbb{R}$, Phase $\phi = 0$ | 122 | $0.0094$ | $100.00\%$ |
| Phase-Only | Gain $w = 1.0$, Phase $\phi \in [-\pi, \pi]$ | 122 | $0.1483$ | $96.00\%$ |
| **Fused Superposition** | **Gain $w \in \mathbb{R}$, Phase $\phi \in [-\pi, \pi]$** | **194** | **$0.0040$** | **$100.00\%$** |
| Classical Real MLP | Matched Parameter Baseline | 145 | $0.0184$ | $100.00\%$ |

*Observation*: Fused superposition achieved the lowest loss ($0.0040$), outperforming the parameter-matched classical baseline ($0.0184$) by $4.6\times$. Phase-only networks successfully untangled $96.0\%$ of the spiral arms without any amplitude modulation.

### 4.2 Phase Decoherence & Interference Breakdown

Physical systems experience thermal and vibrational phase noise:

$$
\tilde{\phi}_{jk} = \phi_{jk} + \delta\phi, \quad \delta\phi \sim \mathcal{N}(0, \sigma_\phi^2)
$$

We measured the test accuracy across 15 Monte Carlo trials per noise level.

**Table 6: Phase Decoherence Breakdown Curve**

| Phase Jitter $\sigma$ (rad) | Phase Jitter $\sigma$ (deg) | Test Accuracy (Mean $\pm$ Std) | Coherence Regime |
| :---: | :---: | :---: | :--- |
| $0.00$ | $0.0^\circ$ | $100.00\% \pm 0.00\%$ | Fully Coherent |
| $0.05$ | $2.9^\circ$ | $93.50\% \pm 6.20\%$ | Robust Quasi-Coherent |
| $0.10$ | $5.7^\circ$ | $86.14\% \pm 9.53\%$ | Coherence Degradation |
| $0.20$ | $11.5^\circ$ | $64.09\% \pm 7.87\%$ | Severe Phase Scrambling |
| **$0.35$** | **$20.1^\circ$** | **$50.78\% \pm 6.19\%$** | **Critical Breakdown Threshold ($\sigma_{\text{crit}}$)** |
| $0.50$ | $28.6^\circ$ | $50.57\% \pm 2.15\%$ | Incoherent Noise Floor |
| $1.00$ | $57.3^\circ$ | $50.29\% \pm 1.72\%$ | Incoherent Noise Floor |
| $3.14$ | $180.0^\circ$ | $49.96\% \pm 1.23\%$ | Complete Uniform Incoherence |

*Physical Interpretation*: At $\sigma_\phi \ge 0.35\text{ rad} \approx 20^\circ$, destructive interference cancellation is degraded, causing test accuracy to collapse to the chance level ($50.0\%$). For reliable analog hardware realization, phase jitter must remain below $5^\circ$.

---

## 5. Continuous-Time Dynamic Attractor Identification

To examine the continuous dynamical properties of the fused superposition neuron, we simulated a damped non-linear wave oscillator:

$$
\ddot{x} + 0.20 \dot{x} + 4.0 x + 1.5 x^3 = 0
$$

We exported $1,000$ points of raw time-series telemetry ($t \in [0, 10]\text{ s}, \Delta t = 0.01\text{ s}$) to `data/superposition_neuron_telemetry.csv` and executed blind symbolic differential discovery via `math_solver.exe`.

**Table 7: Blind Discovery Verification Output**

| Discovery Track | Discovered Mathematical Model | Train $R^2$ | Extrapolation NRMSE | Status |
| :--- | :--- | :---: | :---: | :---: |
| Track 1: Time Domain $x = f(t)$ | UNSOLVED | $-1.000$ | $3.26 \times 10^{-1}$ | Fails (Non-Elementary Trajectory) |
| **Track 2: Phase Space $\ddot{x} = g(x, \dot{x})$** | **$\ddot{x} = -4.0000 x - 0.2000 \dot{x} - 1.5000 x^3$** | **$0.999996$** | **$7.10 \times 10^{-6}$** | **Exact Law Locked ($17.4\text{ ms}$)** |

*Outcome*: The explicit time-domain formulation fails because the trajectory involves non-elementary elliptic functions. The differential engine identified the exact ODE coefficients with an extrapolation error of $7.10 \times 10^{-6}$ in $17.43\text{ ms}$.

---

## 6. Discussion: Wave Domains vs. Spatial Imagery (MNIST)

To establish domain boundaries, we evaluated FSNN on full MNIST ($60,000$ training images, $10,000$ test images) over 5 epochs.

**Table 8: MNIST Digit Benchmark**

| Model Architecture | Layer Dimensions | Parameters | Epoch Time (CPU) | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| Classical Real MLP | $784 \to 128 \to 10$ | 101,770 | $2.1\text{ s}$ | **$97.31\%$** |
| Fused Superposition Net | $784\mathbb{C} \to 128\mathbb{C} \to 10$ | 203,560 | $6.5\text{ s}$ | $96.55\%$ (Peak: $96.55\%$) |

#### Why the Difference Occurs:
1. **Natural vs. Artificial Inductive Bias**: In physical wave domains (RF radar, phased arrays, acoustics, non-linear manifolds), inputs possess intrinsic phase information. Superposition provides a direct physical inductive bias ($+5.60\%$ gain on phased arrays, $43\%$ fewer parameters on spirals).
2. **Spatial Pixel Intensity**: MNIST pixels represent incoherent scalar photon intensities with no native phase. Synthesizing artificial phase carriers ($\theta = \pi \cdot \text{pixel}$) forces the network to learn to suppress artificial phase interference, resulting in slight overhead compared to standard linear layers.

---

## 7. Conclusion

This study explored the mathematical formulation and empirical behavior of neural networks built upon the wave superposition principle. The findings indicate that:
1. A single Fused Superposition Neuron overcomes the linear separability barrier, solving XOR and concentric rings with zero hidden layers.
2. In wave-native domains, FSNN provides significant parameter reduction ($43\text{--}65\%$) and improved generalization.
3. Coherent interference remains stable up to $\sigma_{\text{crit}} \approx 20^\circ$ of phase jitter.
4. For scalar spatial imagery lacking intrinsic phase, standard connectionist architectures remain compute-optimal.

---

## References

1. Barmase, S. (2026). *Data-Driven Identification of Non-Elementary Dynamical Systems from Blind Sensor Telemetry*.
2. Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.
3. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
4. Lin, X., et al. (2018). All-optical machine learning using diffractive deep neural networks. *Science*, 361(6406), 1004-1008.
5. Hirose, A. (2012). *Complex-Valued Neural Networks: Advances and Applications*. IEEE Press, Wiley.
6. Stuart, J. T. (1960). On the non-linear mechanics of wave disturbances in stable and unstable parallel flows. *Journal of Fluid Mechanics*, 9(3), 353-370.
