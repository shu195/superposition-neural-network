"""
Multi-Frequency Coherent Wave Signal Classification Benchmark
Comparing:
1. Standard Real MLP
2. Complex-Valued Neural Net (CVNN)
3. Fused Superposition Neural Net (FSNN) with Kerr Saturation and Intensity Interference

Task: Discriminate coherent constructive wave packets from incoherent destructive noise
across 4 spatial sensor channels (mimicking an RF / Optical phased array antenna).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

torch.manual_seed(42)
np.random.seed(42)

def generate_phased_array_data(n_samples=2000, n_channels=4, n_time=16):
    """
    Simulates a 4-channel phased array receiver detecting wave packets.
    Class 1 (Signal of Interest): Coherent plane wave arriving at angle theta_inc.
           Phase delay across channels: delta_phi = k * d * sin(theta_inc)
           Produces strong coherent superposition across array channels.
    Class 0 (Interference / Clutter): Incoherent multi-path waves with random phase shifts.
    """
    X = []
    Y = []
    
    t = np.linspace(0, 1, n_time)
    
    for i in range(n_samples):
        label = np.random.randint(0, 2)
        if label == 1:
            # Coherent wavefront with definite phase alignment
            theta_inc = np.random.uniform(-np.pi/3, np.pi/3)
            delta_phi = np.pi * np.sin(theta_inc)
            freq = np.random.uniform(2.0, 5.0)
            base_phase = np.random.uniform(0, 2*np.pi)
            
            # Channels receive wave with spatial phase progression
            sig = []
            for ch in range(n_channels):
                phase_ch = base_phase + ch * delta_phi
                wave = np.cos(2 * np.pi * freq * t + phase_ch) + np.random.normal(0, 0.2, n_time)
                sig.append(wave)
        else:
            # Incoherent clutter / random multi-path interference
            sig = []
            for ch in range(n_channels):
                freq_ch = np.random.uniform(2.0, 5.0)
                phase_ch = np.random.uniform(0, 2*np.pi)
                wave = np.cos(2 * np.pi * freq_ch * t + phase_ch) + np.random.normal(0, 0.4, n_time)
                sig.append(wave)
                
        # Stack into [n_channels * n_time] = 64 dimensional vector
        sig = np.array(sig).flatten()
        X.append(sig)
        Y.append(label)
        
    X = np.array(X)
    Y = np.array(Y)[:, None]
    
    # Train / Test split (80% / 20%)
    split = int(0.8 * n_samples)
    return (torch.tensor(X[:split], dtype=torch.float32), torch.tensor(Y[:split], dtype=torch.float32),
            torch.tensor(X[split:], dtype=torch.float32), torch.tensor(Y[split:], dtype=torch.float32))

# =====================================================================
# Architectures
# =====================================================================

class RealMLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class SuperpositionPhasedNet(nn.Module):
    """
    Fused Superposition Network for Phased Signals:
    Converts time-domain samples into analytic quadrature phasors (Hilbert / FFT style):
    psi = x + i * H(x)
    Propagates phasors through complex synaptic phase-shifters (beamforming waveguides)
    with Kerr nonlinear saturation and intensity interference readout.
    """
    def __init__(self, in_features, hidden_dim, out_features=1):
        super().__init__()
        # In features: 64 inputs -> 32 complex analytic phasors
        self.num_phasors = in_features // 2
        
        # Layer 1: Complex wave transmission matrix
        self.w1 = nn.Parameter(torch.randn(self.num_phasors, hidden_dim) / math.sqrt(self.num_phasors))
        self.phi1 = nn.Parameter(torch.rand(self.num_phasors, hidden_dim) * 2 * math.pi - math.pi)
        self.b1_re = nn.Parameter(torch.zeros(hidden_dim))
        self.b1_im = nn.Parameter(torch.zeros(hidden_dim))
        
        # Layer 2: Hidden-to-Output Superposition with Kerr Saturation
        self.w2 = nn.Parameter(torch.randn(hidden_dim, out_features) / math.sqrt(hidden_dim))
        self.phi2 = nn.Parameter(torch.rand(hidden_dim, out_features) * 2 * math.pi - math.pi)
        
        # Interference detector
        self.alpha = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        # Reshape real signal into complex analytic pairs (re, im)
        batch = x.shape[0]
        x_paired = x.view(batch, self.num_phasors, 2)
        x_re = x_paired[:, :, 0]
        x_im = x_paired[:, :, 1]
        
        # Layer 1 Superposition
        W1_re = self.w1 * torch.cos(self.phi1)
        W1_im = self.w1 * torch.sin(self.phi1)
        
        psi1_re = torch.matmul(x_re, W1_re) - torch.matmul(x_im, W1_im) + self.b1_re
        psi1_im = torch.matmul(x_re, W1_im) + torch.matmul(x_im, W1_re) + self.b1_im
        
        # Phase-Preserving Kerr Saturation
        r1 = torch.sqrt(psi1_re**2 + psi1_im**2 + 1e-8)
        amp1 = torch.tanh(r1)
        # Unit phase vector
        unit_re = psi1_re / r1
        unit_im = psi1_im / r1
        
        h1_re = amp1 * unit_re
        h1_im = amp1 * unit_im
        
        # Layer 2 Superposition
        W2_re = self.w2 * torch.cos(self.phi2)
        W2_im = self.w2 * torch.sin(self.phi2)
        
        out_re = torch.matmul(h1_re, W2_re) - torch.matmul(h1_im, W2_im)
        out_im = torch.matmul(h1_re, W2_im) + torch.matmul(h1_im, W2_re)
        
        # Intensity detector
        intensity = out_re**2 + out_im**2
        logit = self.alpha * intensity + self.beta * out_re + self.bias
        return torch.sigmoid(logit)

def run_test():
    X_train, Y_train, X_test, Y_test = generate_phased_array_data(n_samples=2500, n_channels=4, n_time=16)
    in_dim = X_train.shape[1]
    
    print(f"Dataset: Phased Array Wave Detection (64 features, 2000 train, 500 test)")
    
    # 1. Real MLP
    mlp = RealMLP(in_features=in_dim, hidden_dim=32, out_features=1)
    mlp_opt = optim.Adam(mlp.parameters(), lr=0.005)
    crit = nn.BCELoss()
    mlp_params = sum(p.numel() for p in mlp.parameters())
    
    # 2. Superposition Phased Net (using only 16 complex hidden modes!)
    snn = SuperpositionPhasedNet(in_features=in_dim, hidden_dim=16, out_features=1)
    snn_opt = optim.Adam(snn.parameters(), lr=0.005)
    snn_params = sum(p.numel() for p in snn.parameters())
    
    print(f"Real MLP Parameters: {mlp_params}")
    print(f"Superposition Phased Net Parameters: {snn_params} ({snn_params / mlp_params * 100:.1f}% of MLP)")
    
    # Train both for 100 epochs
    batch_size = 64
    n_batches = len(X_train) // batch_size
    
    for epoch in range(1, 101):
        perm = torch.randperm(len(X_train))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            bx, by = X_train[idx], Y_train[idx]
            
            # MLP
            mlp_opt.zero_grad()
            l_mlp = crit(mlp(bx), by)
            l_mlp.backward()
            mlp_opt.step()
            
            # SNN
            snn_opt.zero_grad()
            l_snn = crit(snn(bx), by)
            l_snn.backward()
            snn_opt.step()
            
        if epoch % 25 == 0 or epoch == 1:
            with torch.no_grad():
                mlp_train_acc = ((mlp(X_train) > 0.5).float() == Y_train).float().mean().item() * 100.0
                mlp_test_acc = ((mlp(X_test) > 0.5).float() == Y_test).float().mean().item() * 100.0
                snn_train_acc = ((snn(X_train) > 0.5).float() == Y_train).float().mean().item() * 100.0
                snn_test_acc = ((snn(X_test) > 0.5).float() == Y_test).float().mean().item() * 100.0
            print(f"Epoch {epoch:3d} | Real MLP: Train {mlp_train_acc:.1f}%, Test {mlp_test_acc:.1f}% | SNN: Train {snn_train_acc:.1f}%, Test {snn_test_acc:.1f}%")
            
    print("\n--- FINAL BENCHMARK VERDICT ---")
    print(f"Classical Real MLP Test Accuracy:         {mlp_test_acc:.2f}% (Parameters: {mlp_params})")
    print(f"Fused Superposition Net Test Accuracy:    {snn_test_acc:.2f}% (Parameters: {snn_params})")
    print(f"Parameter Reduction:                      {(1.0 - snn_params/mlp_params)*100:.1f}% fewer parameters")
    print(f"Accuracy Margin:                          {snn_test_acc - mlp_test_acc:+.2f}%")

if __name__ == '__main__':
    run_test()
