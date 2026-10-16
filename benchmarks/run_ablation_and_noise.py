"""
Deep Experiments on Fused Superposition Neural Networks:
1. Experiment A: Mechanism Ablation (Amplitude-Only vs Phase-Only vs Fused vs Equal-Param Real MLP)
2. Experiment B: Phase Decoherence & Interference Breakdown (Physical Noise Sweep)
3. Experiment C: Multi-Frequency Harmonic Demultiplexing in a Single Soma
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# EXPERIMENT A: MECHANISM ABLATION
# =====================================================================
def make_spirals(n_points=300, noise=0.04):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    d2x = np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d2y = -np.sin(n)*n + np.random.rand(n_points, 1) * noise
    X = np.vstack([np.hstack([d1x, d1y]), np.hstack([d2x, d2y])])
    Y = np.vstack([np.zeros((n_points, 1)), np.ones((n_points, 1))])
    X = X / np.max(np.abs(X))
    idx = np.random.permutation(len(X))
    return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(Y[idx], dtype=torch.float32)

class AblationSuperpositionNet(nn.Module):
    def __init__(self, in_features, hidden_dim, mode='fused'):
        """
        mode:
          - 'fused': both w and phi are trained
          - 'amplitude_only': w trained, phi = 0 fixed (no phase retardation)
          - 'phase_only': phi trained, w = 1.0 fixed (optical phase mask / D2NN)
        """
        super().__init__()
        self.mode = mode
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        
        # Layer 1
        if mode == 'phase_only':
            self.w1 = None # fixed to 1.0
        else:
            self.w1 = nn.Parameter(torch.randn(in_features, hidden_dim) / math.sqrt(in_features))
            
        if mode == 'amplitude_only':
            self.phi1 = None # fixed to 0.0
        else:
            self.phi1 = nn.Parameter(torch.rand(in_features, hidden_dim) * 2 * math.pi - math.pi)
            
        self.b1_re = nn.Parameter(torch.zeros(hidden_dim))
        self.b1_im = nn.Parameter(torch.zeros(hidden_dim))
        
        # Output Head
        if mode == 'phase_only':
            self.w2 = None
        else:
            self.w2 = nn.Parameter(torch.randn(hidden_dim, 1) / math.sqrt(hidden_dim))
            
        if mode == 'amplitude_only':
            self.phi2 = None
        else:
            self.phi2 = nn.Parameter(torch.rand(hidden_dim, 1) * 2 * math.pi - math.pi)
            
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, phase_noise_std=0.0):
        amp = torch.ones_like(x)
        theta = x * math.pi
        
        x_re = amp * torch.cos(theta)
        x_im = amp * torch.sin(theta)
        
        # W1 effective
        w1 = torch.ones(self.in_features, self.hidden_dim) if self.w1 is None else self.w1
        phi1 = torch.zeros(self.in_features, self.hidden_dim) if self.phi1 is None else self.phi1
        
        if phase_noise_std > 0.0:
            phi1 = phi1 + torch.randn_like(phi1) * phase_noise_std
            
        W1_re = w1 * torch.cos(phi1)
        W1_im = w1 * torch.sin(phi1)
        
        psi1_re = torch.matmul(x_re, W1_re) - torch.matmul(x_im, W1_im) + self.b1_re
        psi1_im = torch.matmul(x_re, W1_im) + torch.matmul(x_im, W1_re) + self.b1_im
        
        # Kerr saturation
        r1 = torch.sqrt(psi1_re**2 + psi1_im**2 + 1e-8)
        amp1 = torch.tanh(r1)
        theta1 = torch.atan2(psi1_im, psi1_re)
        
        # Layer 2
        h1_re = amp1 * torch.cos(theta1)
        h1_im = amp1 * torch.sin(theta1)
        
        w2 = torch.ones(self.hidden_dim, 1) if self.w2 is None else self.w2
        phi2 = torch.zeros(self.hidden_dim, 1) if self.phi2 is None else self.phi2
        
        if phase_noise_std > 0.0:
            phi2 = phi2 + torch.randn_like(phi2) * phase_noise_std
            
        W2_re = w2 * torch.cos(phi2)
        W2_im = w2 * torch.sin(phi2)
        
        out_re = torch.matmul(h1_re, W2_re) - torch.matmul(h1_im, W2_im)
        out_im = torch.matmul(h1_re, W2_im) + torch.matmul(h1_im, W2_re)
        
        intensity = out_re**2 + out_im**2
        return torch.sigmoid(self.alpha * intensity + self.bias)

def run_experiment_a():
    print("="*70)
    print("EXPERIMENT A: SYNAPTIC MECHANISM ABLATION ON NON-LINEAR MANIFOLD")
    print("Is the power from extra parameters or specifically from phase interference?")
    print("="*70)
    X, Y = make_spirals(n_points=350, noise=0.03)
    
    modes = ['amplitude_only', 'phase_only', 'fused']
    results = {}
    
    for mode in modes:
        model = AblationSuperpositionNet(in_features=2, hidden_dim=24, mode=mode)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = optim.Adam(model.parameters(), lr=0.01)
        crit = nn.BCELoss()
        
        for epoch in range(1200):
            opt.zero_grad()
            out = model(X)
            loss = crit(out, Y)
            loss.backward()
            opt.step()
            
        acc = ((model(X) > 0.5).float() == Y).float().mean().item() * 100.0
        results[mode] = (acc, loss.item(), n_params, model)
        print(f"Mode: {mode:<16} | Params: {n_params:3d} | Final Loss: {loss.item():.4f} | Accuracy: {acc:.2f}%")
        
    # Also compare with Real MLP matching params
    # Real MLP 2 -> 34 -> 1 has (2*34 + 34) + (34*1 + 1) = 102 + 35 = 137 params
    real_mlp = nn.Sequential(
        nn.Linear(2, 36),
        nn.Tanh(),
        nn.Linear(36, 1),
        nn.Sigmoid()
    )
    n_real = sum(p.numel() for p in real_mlp.parameters())
    opt_real = optim.Adam(real_mlp.parameters(), lr=0.01)
    crit = nn.BCELoss()
    for _ in range(1200):
        opt_real.zero_grad()
        loss_r = crit(real_mlp(X), Y)
        loss_r.backward()
        opt_real.step()
    acc_r = ((real_mlp(X) > 0.5).float() == Y).float().mean().item() * 100.0
    print(f"Mode: {'real_mlp_match':<16} | Params: {n_real:3d} | Final Loss: {loss_r.item():.4f} | Accuracy: {acc_r:.2f}%")
    return results['fused'][3], X, Y

# =====================================================================
# EXPERIMENT B: PHASE DECOHERENCE & INTERFERENCE BREAKDOWN
# =====================================================================
def run_experiment_b(fused_model, X, Y):
    print("\n" + "="*70)
    print("EXPERIMENT B: PHASE DECOHERENCE & INTERFERENCE BREAKDOWN SWEEP")
    print("Measuring accuracy degradation as phase jitter sigma increases")
    print("="*70)
    
    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 3.14]
    
    print(f"{'Phase Jitter (rad)':<20} | {'Phase Jitter (deg)':<20} | {'Test Accuracy (%)':<20}")
    print("-" * 65)
    
    for sigma in noise_levels:
        accs = []
        for trial in range(15): # 15 stochastic evaluations per noise level
            with torch.no_grad():
                preds = fused_model(X, phase_noise_std=sigma)
                acc = ((preds > 0.5).float() == Y).float().mean().item() * 100.0
                accs.append(acc)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        deg = sigma * (180.0 / math.pi)
        print(f"{sigma:<20.2f} | {deg:<20.1f} | {mean_acc:6.2f}% +/- {std_acc:4.2f}%")

# =====================================================================
# EXPERIMENT C: HARMONIC DEMULTIPLEXING IN A SINGLE SOMA
# =====================================================================
def run_experiment_c():
    print("\n" + "="*70)
    print("EXPERIMENT C: HARMONIC MULTI-FREQUENCY DEMULTIPLEXING IN 1 SOMA")
    print("Can 1 Superposition Soma demultiplex overlaid signals via wave orthogonality?")
    print("="*70)
    
    # 3 simultaneous signals modulated at distinct frequencies:
    # Channel 1: omega_1 = 1.0 (Bit A in {0, 1})
    # Channel 2: omega_2 = 2.0 (Bit B in {0, 1})
    # Channel 3: omega_3 = 3.0 (Bit C in {0, 1})
    # All 3 signals superpose in the SAME channel:
    # S(t) = Bit_A * cos(omega_1 * t) + Bit_B * cos(omega_2 * t) + Bit_C * cos(omega_3 * t)
    
    n_samples = 1200
    n_time = 32
    t = torch.linspace(0, 2*math.pi, n_time)
    
    bits = torch.randint(0, 2, (n_samples, 3)).float()
    
    # Composite superposed waveform
    S = (bits[:, 0:1] * torch.cos(1.0 * t).unsqueeze(0) +
         bits[:, 1:2] * torch.cos(2.0 * t).unsqueeze(0) +
         bits[:, 2:3] * torch.cos(3.0 * t).unsqueeze(0))
         
    # Add noise
    S = S + torch.randn_like(S) * 0.1
    
    print(f"Input: Overlaid 3-harmonic wave mixture S(t) across {n_time} time steps.")
    print("Task: Read out Bit 2 (omega = 2.0) despite presence of overlapping omega=1.0 and omega=3.0.")
    
    target_bit = bits[:, 1:2] # Target is Bit B
    
    # 1. Classical Single Perceptron
    p_classical = nn.Sequential(nn.Linear(n_time, 1), nn.Sigmoid())
    p_opt = optim.Adam(p_classical.parameters(), lr=0.01)
    crit = nn.BCELoss()
    for _ in range(500):
        p_opt.zero_grad()
        loss = crit(p_classical(S), target_bit)
        loss.backward()
        p_opt.step()
    acc_classical = ((p_classical(S) > 0.5).float() == target_bit).float().mean().item() * 100.0
    
    # 2. Resonant Superposition Neuron (Single Soma with phase-frequency tuning)
    class ResonantSuperpositionNeuron(nn.Module):
        def __init__(self, in_features):
            super().__init__()
            # Complex synaptic resonator
            self.w = nn.Parameter(torch.randn(in_features) * 0.1)
            self.phi = nn.Parameter(torch.rand(in_features) * 2 * math.pi - math.pi)
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.threshold = nn.Parameter(torch.tensor(0.5))
            
        def forward(self, x):
            # Complex phasor representation
            # x acts as amplitude; phase is uniform or time-encoded
            re = torch.sum(x * self.w.unsqueeze(0) * torch.cos(self.phi.unsqueeze(0)), dim=-1)
            im = torch.sum(x * self.w.unsqueeze(0) * torch.sin(self.phi.unsqueeze(0)), dim=-1)
            intensity = re**2 + im**2
            return torch.sigmoid(self.alpha * intensity - self.threshold)
            
    s_neuron = ResonantSuperpositionNeuron(in_features=n_time)
    s_opt = optim.Adam(s_neuron.parameters(), lr=0.02)
    for _ in range(500):
        s_opt.zero_grad()
        loss_s = crit(s_neuron(S).unsqueeze(-1), target_bit)
        loss_s.backward()
        s_opt.step()
    acc_snn = ((s_neuron(S).unsqueeze(-1) > 0.5).float() == target_bit).float().mean().item() * 100.0
    
    print(f"Classical Linear Perceptron: Accuracy = {acc_classical:.2f}%")
    print(f"Resonant Superposition Neuron: Accuracy = {acc_snn:.2f}%")

if __name__ == '__main__':
    fused_m, X, Y = run_experiment_a()
    run_experiment_b(fused_m, X, Y)
    run_experiment_c()
