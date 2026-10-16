"""
Comprehensive Benchmark Suite for Fused Superposition Neural Networks (FSNN)
Ablations & Tests:
1. Concentric Rings (Radially Non-separable): Single FSN vs Single Perceptron
2. Two-Spirals Benchmark: Deep FSNN vs Standard Real MLP (Capacity, Parameter Efficiency, Accuracy)
3. Continuous-Time Dynamic Attractor: Extract telemetry and verify with math_solver.exe
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import subprocess
import os

torch.manual_seed(123)
np.random.seed(123)

# =====================================================================
# 1. Concentric Circles Dataset
# =====================================================================
def make_concentric_circles(n_samples=400, noise=0.05):
    n_per_ring = n_samples // 2
    # Inner ring: radius ~ 0.4
    r_inner = np.random.normal(0.35, noise, n_per_ring)
    theta_inner = np.random.uniform(0, 2*np.pi, n_per_ring)
    x_inner = np.stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)], axis=1)
    y_inner = np.zeros((n_per_ring, 1))
    
    # Outer ring: radius ~ 0.85
    r_outer = np.random.normal(0.85, noise, n_per_ring)
    theta_outer = np.random.uniform(0, 2*np.pi, n_per_ring)
    x_outer = np.stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)], axis=1)
    y_outer = np.ones((n_per_ring, 1))
    
    X = np.vstack([x_inner, x_outer])
    Y = np.vstack([y_inner, y_outer])
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(Y[idx], dtype=torch.float32)

# =====================================================================
# 2. Two-Spirals Dataset
# =====================================================================
def make_two_spirals(n_points=300, noise=0.05):
    # Spiral 1
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    
    # Spiral 2
    d2x = np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d2y = -np.sin(n)*n + np.random.rand(n_points, 1) * noise
    
    X = np.vstack([np.hstack([d1x, d1y]), np.hstack([d2x, d2y])])
    Y = np.vstack([np.zeros((n_points, 1)), np.ones((n_points, 1))])
    
    # Normalize to [-1, 1]
    X = X / np.max(np.abs(X))
    
    idx = np.random.permutation(len(X))
    return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(Y[idx], dtype=torch.float32)

# =====================================================================
# 3. Layer Architectures
# =====================================================================
class SuperpositionLayer(nn.Module):
    """
    Multi-channel Fused Superposition Layer:
    Incoming wave phasors: Psi_in = A_in * e^{i Theta_in}  [batch, in_features]
    Synaptic transmission matrix: W = w_jk * e^{i phi_jk} [in_features, out_features]
    Superposition at soma:
      Psi_out_re = (A_in * cos(Theta)) @ (w * cos(phi)) - (A_in * sin(Theta)) @ (w * sin(phi)) + b_re
      Psi_out_im = (A_in * cos(Theta)) @ (w * sin(phi)) + (A_in * sin(Theta)) @ (w * cos(phi)) + b_im
    Phase-preserving Kerr nonlinear saturation:
      Amp_out = tanh(|Psi_out|)
      Theta_out = angle(Psi_out)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Gain and phase parameters
        self.w = nn.Parameter(torch.randn(in_features, out_features) / math.sqrt(in_features))
        self.phi = nn.Parameter(torch.rand(in_features, out_features) * 2 * math.pi - math.pi)
        
        # Bias phasors
        self.b_re = nn.Parameter(torch.zeros(out_features))
        self.b_im = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, amp_in, theta_in):
        # Quadrature decomposition of incoming waves
        x_re = amp_in * torch.cos(theta_in)
        x_im = amp_in * torch.sin(theta_in)
        
        # Synaptic transmission matrix components
        W_re = self.w * torch.cos(self.phi)
        W_im = self.w * torch.sin(self.phi)
        
        # Complex matrix multiplication (Wave Superposition)
        # (x_re + i x_im) @ (W_re + i W_im) = (x_re @ W_re - x_im @ W_im) + i (x_re @ W_im + x_im @ W_re)
        psi_re = torch.matmul(x_re, W_re) - torch.matmul(x_im, W_im) + self.b_re
        psi_im = torch.matmul(x_re, W_im) + torch.matmul(x_im, W_re) + self.b_im
        
        # Total amplitude (Envelope) and Phase
        r = torch.sqrt(psi_re**2 + psi_im**2 + 1e-8)
        
        # Phase-preserving saturable Kerr compression:
        amp_out = torch.tanh(r)
        theta_out = torch.atan2(psi_im, psi_re)
        
        return amp_out, theta_out, (psi_re, psi_im, r**2)

class FusedSuperpositionNetwork(nn.Module):
    """
    Deep Neural Net based entirely on Wave Superposition:
    Input: Real coordinates (x, y) encoded as amplitude and spatial phase
    Hidden Layers: Superposition Layers with Kerr Saturation
    Output: Fused Superposition Intensity Detector
    """
    def __init__(self, in_features, hidden_dims, out_features=1):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = in_features
        for h in hidden_dims:
            self.layers.append(SuperpositionLayer(prev_dim, h))
            prev_dim = h
            
        # Final intensity detection head
        self.final_w = nn.Parameter(torch.randn(prev_dim, out_features) / math.sqrt(prev_dim))
        self.final_phi = nn.Parameter(torch.rand(prev_dim, out_features) * 2 * math.pi - math.pi)
        self.alpha = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        # Wave encoding: coordinates map to spatial wavefront
        # amplitude = 1.0, phase = x * pi
        amp = torch.ones_like(x)
        theta = x * math.pi
        
        for layer in self.layers:
            amp, theta, _ = layer(amp, theta)
            
        # Final interference readout
        x_re = amp * torch.cos(theta)
        x_im = amp * torch.sin(theta)
        W_re = self.final_w * torch.cos(self.final_phi)
        W_im = self.final_w * torch.sin(self.final_phi)
        
        out_re = torch.matmul(x_re, W_re) - torch.matmul(x_im, W_im)
        out_im = torch.matmul(x_re, W_im) + torch.matmul(x_im, W_re)
        intensity = out_re**2 + out_im**2
        
        logit = self.alpha * intensity + self.beta * out_re + self.bias
        return torch.sigmoid(logit)

# =====================================================================
# 4. Standard Classical MLP for Direct Comparison
# =====================================================================
class ClassicalMLP(nn.Module):
    def __init__(self, in_features, hidden_dims, out_features=1):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, out_features))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =====================================================================
# RUN BENCHMARKS
# =====================================================================
def run_all_benchmarks():
    print("="*70)
    print("FUSED SUPERPOSITION NEURAL NETWORK: EMPIRICAL BENCHMARK SUITE")
    print("="*70)
    
    # -------------------------------------------------------------
    # Test 1: Concentric Rings (Radial Invariance & Superposition)
    # -------------------------------------------------------------
    print("\n--- TEST 1: CONCENTRIC NON-SEPARABLE CIRCLES ---")
    X_circ, Y_circ = make_concentric_circles(n_samples=600, noise=0.04)
    
    # Classical Single Perceptron (0 hidden layers)
    c_single = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    c_opt = optim.Adam(c_single.parameters(), lr=0.05)
    crit = nn.BCELoss()
    for _ in range(500):
        c_opt.zero_grad()
        loss = crit(c_single(X_circ), Y_circ)
        loss.backward()
        c_opt.step()
    c_acc = ((c_single(X_circ) > 0.5).float() == Y_circ).float().mean().item() * 100.0
    
    # Fused Superposition Network (0 hidden layers, 1 output neuron with interference)
    fsn_single = FusedSuperpositionNetwork(in_features=2, hidden_dims=[], out_features=1)
    f_opt = optim.Adam(fsn_single.parameters(), lr=0.05)
    for _ in range(500):
        f_opt.zero_grad()
        loss = crit(fsn_single(X_circ), Y_circ)
        loss.backward()
        f_opt.step()
    f_acc = ((fsn_single(X_circ) > 0.5).float() == Y_circ).float().mean().item() * 100.0
    
    print(f"Classical Perceptron (0 hidden layers, {count_params(c_single)} params): Accuracy = {c_acc:.2f}% (Chance level: 50%)")
    print(f"Fused Superposition Neuron (0 hidden layers, {count_params(fsn_single)} params): Accuracy = {f_acc:.2f}%")
    
    # -------------------------------------------------------------
    # Test 2: Two-Spirals Non-Linear Manifold Classification
    # -------------------------------------------------------------
    print("\n--- TEST 2: TWO-SPIRALS MANIFOLD BENCHMARK ---")
    X_sp, Y_sp = make_two_spirals(n_points=400, noise=0.03)
    
    # Standard MLP: 2 -> 32 -> 32 -> 1
    mlp = ClassicalMLP(in_features=2, hidden_dims=[32, 32], out_features=1)
    mlp_opt = optim.Adam(mlp.parameters(), lr=0.01)
    mlp_params = count_params(mlp)
    
    # Fused Superposition Net: 2 -> 16 -> 16 -> 1 (Substantially fewer neurons due to phase dimension!)
    fsnn = FusedSuperpositionNetwork(in_features=2, hidden_dims=[16, 16], out_features=1)
    fsnn_opt = optim.Adam(fsnn.parameters(), lr=0.01)
    fsnn_params = count_params(fsnn)
    
    # Train both for 1500 epochs
    epochs = 1500
    for epoch in range(epochs):
        # MLP step
        mlp_opt.zero_grad()
        l_mlp = crit(mlp(X_sp), Y_sp)
        l_mlp.backward()
        mlp_opt.step()
        
        # FSNN step
        fsnn_opt.zero_grad()
        l_fsnn = crit(fsnn(X_sp), Y_sp)
        l_fsnn.backward()
        fsnn_opt.step()
        
    mlp_acc = ((mlp(X_sp) > 0.5).float() == Y_sp).float().mean().item() * 100.0
    fsnn_acc = ((fsnn(X_sp) > 0.5).float() == Y_sp).float().mean().item() * 100.0
    
    print(f"Classical MLP [2->32->32->1]: Parameters = {mlp_params} | Final Loss = {l_mlp.item():.4f} | Accuracy = {mlp_acc:.2f}%")
    print(f"Fused Superposition Net [2->16->16->1]: Parameters = {fsnn_params} | Final Loss = {l_fsnn.item():.4f} | Accuracy = {fsnn_acc:.2f}%")
    
    # -------------------------------------------------------------
    # Test 3: Continuous Dynamical Superposition Neuron & Math Solver ODE Discovery
    # -------------------------------------------------------------
    print("\n--- TEST 3: DYNAMICAL STUART-LANDAU/KERR SUPERPOSITION ATTRACTOR ---")
    print("Simulating continuous-time superposition oscillator:")
    print("d^2x/dt^2 + gamma*dx/dt + omega0^2*x + beta*x^3 = 0")
    
    # Parameters of physical wave oscillator
    gamma = 0.20
    omega0 = 2.0  # omega0^2 = 4.0
    beta = 1.50   # cubic nonlinear Kerr saturation
    
    # RK4 Simulation
    dt = 0.01
    steps = 1000
    t = np.linspace(0, (steps-1)*dt, steps)
    x = np.zeros(steps)
    v = np.zeros(steps)
    x[0] = 1.5
    v[0] = 0.0
    
    for i in range(steps - 1):
        def f(cur_x, cur_v):
            return -gamma * cur_v - (omega0**2) * cur_x - beta * (cur_x**3)
            
        k1_x = v[i]
        k1_v = f(x[i], v[i])
        
        k2_x = v[i] + 0.5 * dt * k1_v
        k2_v = f(x[i] + 0.5 * dt * k1_x, v[i] + 0.5 * dt * k1_v)
        
        k3_x = v[i] + 0.5 * dt * k2_v
        k3_v = f(x[i] + 0.5 * dt * k2_x, v[i] + 0.5 * dt * k2_v)
        
        k4_x = v[i] + dt * k3_v
        k4_v = f(x[i] + dt * k3_x, v[i] + dt * k3_v)
        
        x[i+1] = x[i] + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v[i+1] = v[i] + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, "data", "superposition_neuron_telemetry.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as fp:
        fp.write("t,x\n")
        for ti, xi in zip(t, x):
            fp.write(f"{ti:.6f},{xi:.6f}\n")
            
    print(f"Exported {steps} telemetry points of fused neuron dynamics to {csv_path}")
    
    # Check for math_solver binary
    solver_path = os.path.join(os.path.dirname(repo_root), "code-neuron-discovery", "math_solver.exe")
    if os.path.exists(solver_path):
        print("Invoking math_solver.exe --telemetry on blind telemetry to discover governing law...")
        cmd = [solver_path, "--telemetry", csv_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("\n--- MATH_SOLVER CLI VERBATIM OUTPUT ---")
        print(res.stdout)
        if res.stderr:
            print("Stderr:", res.stderr)
    else:
        print(f"math_solver.exe not found at {solver_path}. Telemetry CSV saved.")

if __name__ == '__main__':
    run_all_benchmarks()
