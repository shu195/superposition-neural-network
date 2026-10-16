"""
Fused Superposition Neural Network (FSNN)
Core Implementation Module

Fuses the linear wave superposition principle:
    Psi_k = sum_j W_{kj} s_j + B_k = sum_j (w_{kj} e^{i phi_{kj}}) * (a_j e^{i theta_j}) + B_k
with nonlinear neural activation and saturable absorption:
    Phase-preserving Kerr saturation: f(Psi) = tanh(|Psi|) * (Psi / |Psi|)
    Interference intensity readout:    y = sigmoid(alpha * |Psi|^2 + beta * Re(Psi) + bias)

Author: Shubham Barmase (Independent Researcher, barmase32@gmail.com)
"""

import torch
import torch.nn as nn
import math

class FusedSuperpositionNeuron(nn.Module):
    """
    A single artificial neuron operating on wave superposition.
    Capable of non-linearly separable classification (e.g. XOR, concentric circles)
    without any hidden layers via constructive and destructive wave interference.
    """
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        # Synaptic wave parameters: gain w and phase delay phi
        self.w = nn.Parameter(torch.randn(in_features) * 0.5 + 1.0)
        self.phi = nn.Parameter(torch.randn(in_features) * 0.5)
        
        # Complex bias: magnitude and phase
        self.b_mag = nn.Parameter(torch.zeros(1))
        self.b_phase = nn.Parameter(torch.zeros(1))
        
        # Detector fusion parameters
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_amp, x_phase=None):
        if x_phase is None:
            # Default phase encoding: map real scalar to phase [0, pi]
            x_phase = x_amp * math.pi
            x_amp = torch.ones_like(x_amp)
            
        total_phase = x_phase + self.phi.unsqueeze(0)
        eff_amp = x_amp * self.w.unsqueeze(0)
        
        # Real and Imaginary quadrature sums
        psi_re = torch.sum(eff_amp * torch.cos(total_phase), dim=-1) + self.b_mag * torch.cos(self.b_phase)
        psi_im = torch.sum(eff_amp * torch.sin(total_phase), dim=-1) + self.b_mag * torch.sin(self.b_phase)
        
        intensity = psi_re**2 + psi_im**2
        activation_input = self.alpha * intensity + self.beta * psi_re - self.threshold
        y = torch.sigmoid(activation_input)
        return y, (psi_re, psi_im, intensity)


class SuperpositionLayer(nn.Module):
    """
    Multi-channel Fused Superposition Layer:
    Propagates complex wave states through a complex synaptic transmission matrix
    and applies phase-preserving saturable Kerr absorption:
        Amp_out = tanh(|Psi_out|)
        Phase_out = angle(Psi_out)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.w = nn.Parameter(torch.randn(in_features, out_features) / math.sqrt(in_features))
        self.phi = nn.Parameter(torch.rand(in_features, out_features) * 2 * math.pi - math.pi)
        
        self.b_re = nn.Parameter(torch.zeros(out_features))
        self.b_im = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, amp_in, theta_in, phase_noise_std=0.0):
        x_re = amp_in * torch.cos(theta_in)
        x_im = amp_in * torch.sin(theta_in)
        
        phi = self.phi
        if phase_noise_std > 0.0:
            phi = phi + torch.randn_like(phi) * phase_noise_std
            
        W_re = self.w * torch.cos(phi)
        W_im = self.w * torch.sin(phi)
        
        psi_re = torch.matmul(x_re, W_re) - torch.matmul(x_im, W_im) + self.b_re
        psi_im = torch.matmul(x_re, W_im) + torch.matmul(x_im, W_re) + self.b_im
        
        r = torch.sqrt(psi_re**2 + psi_im**2 + 1e-8)
        amp_out = torch.tanh(r)
        theta_out = torch.atan2(psi_im, psi_re)
        
        return amp_out, theta_out, (psi_re, psi_im, r**2)


class FusedSuperpositionNetwork(nn.Module):
    """
    Deep Neural Network based on the Superposition Principle.
    Constructed of stacked SuperpositionLayers with Kerr saturation and a
    final intensity detection head.
    """
    def __init__(self, in_features, hidden_dims, out_features=1):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = in_features
        for h in hidden_dims:
            self.layers.append(SuperpositionLayer(prev_dim, h))
            prev_dim = h
            
        self.final_w = nn.Parameter(torch.randn(prev_dim, out_features) / math.sqrt(prev_dim))
        self.final_phi = nn.Parameter(torch.rand(prev_dim, out_features) * 2 * math.pi - math.pi)
        self.alpha = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x, phase_noise_std=0.0):
        amp = torch.ones_like(x)
        theta = x * math.pi
        
        for layer in self.layers:
            amp, theta, _ = layer(amp, theta, phase_noise_std=phase_noise_std)
            
        x_re = amp * torch.cos(theta)
        x_im = amp * torch.sin(theta)
        
        final_phi = self.final_phi
        if phase_noise_std > 0.0:
            final_phi = final_phi + torch.randn_like(final_phi) * phase_noise_std
            
        W_re = self.final_w * torch.cos(final_phi)
        W_im = self.final_w * torch.sin(final_phi)
        
        out_re = torch.matmul(x_re, W_re) - torch.matmul(x_im, W_im)
        out_im = torch.matmul(x_re, W_im) + torch.matmul(x_im, W_re)
        intensity = out_re**2 + out_im**2
        
        logit = self.alpha * intensity + self.beta * out_re + self.bias
        if self.final_w.shape[1] == 1:
            return torch.sigmoid(logit)
        return logit
