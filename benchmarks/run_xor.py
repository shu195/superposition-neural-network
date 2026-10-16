"""
Empirical Verification of Fused Superposition Neuron (FSN)
Test 1: Single Superposition Neuron on Non-Linear Separability (XOR & Concentric Rings)
Comparing Single Classical Neuron (Perceptron) vs. Single Fused Superposition Neuron
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Classical Real Neuron (Single Perceptron)
# =====================================================================
class ClassicalNeuron(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# =====================================================================
# 2. Fused Superposition Neuron (Single FSN)
# Fuses linear wave superposition: Psi = sum_j (w_j * e^{i phi_j}) * (x_j * e^{i theta_j}) + B
# with nonlinear intensity & phase detection: y = sigmoid(alpha * |Psi|^2 + beta * Re(Psi) + c)
# =====================================================================
class FusedSuperpositionNeuron(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        # Synaptic wave parameters: gain w and phase delay phi
        self.w = nn.Parameter(torch.randn(in_features) * 0.5 + 1.0) # conductance
        self.phi = nn.Parameter(torch.randn(in_features) * 0.5)      # phase retardance
        
        # Complex bias: magnitude and phase
        self.b_mag = nn.Parameter(torch.zeros(1))
        self.b_phase = nn.Parameter(torch.zeros(1))
        
        # Detector fusion parameters:
        # alpha modulates pure interference intensity (|Psi|^2)
        # beta modulates coherent field amplitude (Re(Psi))
        # c is detection threshold
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_amp, x_phase=None):
        """
        x_amp: [batch, in_features] - wave amplitudes
        x_phase: [batch, in_features] - wave phases (if None, set to 0 or phase-encoded)
        """
        if x_phase is None:
            # Phase encoding: map real inputs into phase angle [0, pi]
            x_phase = x_amp * math.pi
            x_amp = torch.ones_like(x_amp)
            
        # Synaptic superposition:
        # Psi_j = (w_j * x_j) * e^{i (theta_j + phi_j)}
        total_phase = x_phase + self.phi.unsqueeze(0)
        eff_amp = x_amp * self.w.unsqueeze(0)
        
        # Superposition sum (Real and Imaginary quadrature)
        psi_re = torch.sum(eff_amp * torch.cos(total_phase), dim=-1) + self.b_mag * torch.cos(self.b_phase)
        psi_im = torch.sum(eff_amp * torch.sin(total_phase), dim=-1) + self.b_mag * torch.sin(self.b_phase)
        
        # Interference intensity: I = |Psi|^2 = psi_re^2 + psi_im^2
        intensity = psi_re**2 + psi_im**2
        
        # Fused neural activation
        activation_input = self.alpha * intensity + self.beta * psi_re - self.threshold
        y = torch.sigmoid(activation_input)
        return y, (psi_re, psi_im, intensity)

def test_xor():
    print("="*60)
    print("EXPERIMENT 1: XOR PROBLEM (NON-LINEAR SEPARABILITY)")
    print("Can a SINGLE neuron solve XOR without any hidden layers?")
    print("="*60)
    
    # XOR truth table:
    # (0, 0) -> 0
    # (0, 1) -> 1
    # (1, 0) -> 1
    # (1, 1) -> 0
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    
    # 1. Classical Neuron Training
    c_neuron = ClassicalNeuron(in_features=2)
    c_opt = optim.Adam(c_neuron.parameters(), lr=0.1)
    criterion = nn.BCELoss()
    
    for epoch in range(1000):
        c_opt.zero_grad()
        out = c_neuron(X)
        loss = criterion(out, Y)
        loss.backward()
        c_opt.step()
        
    c_preds = (c_neuron(X) > 0.5).float()
    c_acc = (c_preds == Y).float().mean().item() * 100.0
    print(f"\n[Classical Neuron - 0 Hidden Layers, 1 Real Output]")
    print(f"Predictions:\n{c_neuron(X).detach().numpy().flatten().round(decimals=3)}")
    print(f"Target:\n{Y.numpy().flatten()}")
    print(f"Classical Perceptron Accuracy: {c_acc:.1f}% (Theoretically bounded at 75%)")
    
    # 2. Fused Superposition Neuron Training
    fsn = FusedSuperpositionNeuron(in_features=2)
    f_opt = optim.Adam(fsn.parameters(), lr=0.05)
    
    for epoch in range(1000):
        f_opt.zero_grad()
        out, (psi_re, psi_im, intensity) = fsn(X)
        loss = criterion(out.unsqueeze(-1), Y)
        loss.backward()
        f_opt.step()
        
    f_out, (psi_re, psi_im, intensity) = fsn(X)
    f_preds = (f_out.unsqueeze(-1) > 0.5).float()
    f_acc = (f_preds == Y).float().mean().item() * 100.0
    
    print(f"\n[Fused Superposition Neuron - 0 Hidden Layers, 1 Fused Output]")
    print(f"Output Activations:\n{f_out.detach().numpy().round(decimals=3)}")
    print(f"Interference Intensities |Psi|^2:\n{intensity.detach().numpy().round(decimals=3)}")
    print(f"Re(Psi) Quadrature:\n{psi_re.detach().numpy().round(decimals=3)}")
    print(f"Im(Psi) Quadrature:\n{psi_im.detach().numpy().round(decimals=3)}")
    print(f"Superposition Neuron Accuracy: {f_acc:.1f}%")
    
    print(f"\nLearned Synaptic Parameters:")
    print(f"w (Conductances): {fsn.w.detach().numpy().round(decimals=3)}")
    print(f"phi (Phase Retardations, rad): {fsn.phi.detach().numpy().round(decimals=3)}")
    print(f"alpha (Intensity weight): {fsn.alpha.item():.3f}")
    print(f"threshold: {fsn.threshold.item():.3f}")

if __name__ == '__main__':
    test_xor()
