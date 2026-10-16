"""
Full Real-World Benchmark: Fused Superposition Network vs Real MLP on Full MNIST (60,000 Train / 10,000 Test)
Measures:
- Parameter count
- Epoch wall-clock training time
- Cross-entropy loss curve
- Final Test Accuracy on 10,000 unseen test digits
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import struct

torch.manual_seed(42)
np.random.seed(42)

def load_mnist_raw():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "data", "mnist_raw"),
        r"C:\Users\Admin\.gemini\antigravity\scratch\mnist_raw"
    ]
    path = None
    for c in candidates:
        if os.path.exists(os.path.join(c, "train-images-idx3-ubyte")):
            path = c
            break
    if path is None:
        raise FileNotFoundError("Could not locate raw MNIST files.")
    with open(os.path.join(path, "train-images-idx3-ubyte"), "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        train_x = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows * cols).astype(np.float32) / 255.0
    with open(os.path.join(path, "train-labels-idx1-ubyte"), "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    with open(os.path.join(path, "t10k-images-idx3-ubyte"), "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        test_x = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows * cols).astype(np.float32) / 255.0
    with open(os.path.join(path, "t10k-labels-idx1-ubyte"), "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
        
    return (torch.tensor(train_x), torch.tensor(train_y),
            torch.tensor(test_x), torch.tensor(test_y))

# =====================================================================
# 1. Classical Real MLP Baseline: 784 -> 128 -> 10
# =====================================================================
class ClassicalMNISTMLP(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)

# =====================================================================
# 2. Fused Superposition Network: 784 -> 128 -> 10
# Uses Wave Superposition + Kerr Saturation + Intensity Detectors
# =====================================================================
class SuperpositionMNISTNet(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Layer 1 Synaptic transmission: gain w and phase delay phi
        self.w1 = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.05)
        self.phi1 = nn.Parameter(torch.rand(in_dim, hidden_dim) * 2 * np.pi - np.pi)
        self.b1_re = nn.Parameter(torch.zeros(hidden_dim))
        self.b1_im = nn.Parameter(torch.zeros(hidden_dim))
        
        # Layer 2 Synaptic transmission: hidden -> 10 class beams
        self.w2 = nn.Parameter(torch.randn(hidden_dim, num_classes) * 0.1)
        self.phi2 = nn.Parameter(torch.rand(hidden_dim, num_classes) * 2 * np.pi - np.pi)
        self.b2_re = nn.Parameter(torch.zeros(num_classes))
        self.b2_im = nn.Parameter(torch.zeros(num_classes))
        
        # Class detector intensity gain and bias
        self.alpha = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        
    def forward(self, x):
        # Wavefront encoding:
        # pixel amplitude A = x
        # spatial phase theta = x * pi
        x_re = x * torch.cos(x * np.pi)
        x_im = x * torch.sin(x * np.pi)
        
        # Layer 1: Superposition Sum
        W1_re = self.w1 * torch.cos(self.phi1)
        W1_im = self.w1 * torch.sin(self.phi1)
        
        psi1_re = torch.matmul(x_re, W1_re) - torch.matmul(x_im, W1_im) + self.b1_re
        psi1_im = torch.matmul(x_re, W1_im) + torch.matmul(x_im, W1_re) + self.b1_im
        
        # Kerr nonlinear saturation: tanh(|Psi|) * (Psi / |Psi|)
        r1 = torch.sqrt(psi1_re**2 + psi1_im**2 + 1e-8)
        amp1 = torch.tanh(r1)
        unit1_re = psi1_re / r1
        unit1_im = psi1_im / r1
        
        h1_re = amp1 * unit1_re
        h1_im = amp1 * unit1_im
        
        # Layer 2: Output Class Superposition
        W2_re = self.w2 * torch.cos(self.phi2)
        W2_im = self.w2 * torch.sin(self.phi2)
        
        out_re = torch.matmul(h1_re, W2_re) - torch.matmul(h1_im, W2_im) + self.b2_re
        out_im = torch.matmul(h1_re, W2_im) + torch.matmul(h1_im, W2_re) + self.b2_im
        
        # Superposition Intensity Detection (Energy density per class channel)
        intensity = out_re**2 + out_im**2
        logits = self.alpha * intensity + self.bias
        return logits

def run_mnist_benchmark():
    print("="*75)
    print("FULL MNIST BENCHMARK: FUSED SUPERPOSITION NETWORK VS REAL MLP")
    print("60,000 Train Images | 10,000 Test Images | 5 Training Epochs")
    print("="*75)
    
    train_x, train_y, test_x, test_y = load_mnist_raw()
    print(f"Data Loaded: Train {train_x.shape}, Test {test_x.shape}")
    
    # 1. Models Setup
    mlp = ClassicalMNISTMLP(784, 128, 10)
    snn = SuperpositionMNISTNet(784, 128, 10)
    
    mlp_params = sum(p.numel() for p in mlp.parameters())
    snn_params = sum(p.numel() for p in snn.parameters())
    print(f"Classical MLP Parameters:       {mlp_params}")
    print(f"Superposition Net Parameters:   {snn_params}")
    
    mlp_opt = optim.Adam(mlp.parameters(), lr=0.002)
    snn_opt = optim.Adam(snn.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 128
    n_batches = len(train_x) // batch_size
    epochs = 5
    
    # Benchmark Loop
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(train_x))
        
        # Train MLP
        t0_mlp = time.time()
        mlp.train()
        loss_mlp_sum = 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            bx, by = train_x[idx], train_y[idx]
            mlp_opt.zero_grad()
            out_m = mlp(bx)
            loss_m = criterion(out_m, by)
            loss_m.backward()
            mlp_opt.step()
            loss_mlp_sum += loss_m.item()
        t_mlp = time.time() - t0_mlp
        avg_loss_mlp = loss_mlp_sum / n_batches
        
        # Train SNN
        t0_snn = time.time()
        snn.train()
        loss_snn_sum = 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            bx, by = train_x[idx], train_y[idx]
            snn_opt.zero_grad()
            out_s = snn(bx)
            loss_s = criterion(out_s, by)
            loss_s.backward()
            snn_opt.step()
            loss_snn_sum += loss_s.item()
        t_snn = time.time() - t0_snn
        avg_loss_snn = loss_snn_sum / n_batches
        
        # Evaluate on Test Set
        mlp.eval()
        snn.eval()
        with torch.no_grad():
            mlp_preds = mlp(test_x).argmax(dim=-1)
            mlp_test_acc = (mlp_preds == test_y).float().mean().item() * 100.0
            
            snn_preds = snn(test_x).argmax(dim=-1)
            snn_test_acc = (snn_preds == test_y).float().mean().item() * 100.0
            
        print(f"Epoch {epoch}/{epochs} | MLP: Loss {avg_loss_mlp:.4f}, Test Acc {mlp_test_acc:5.2f}% ({t_mlp:.1f}s) | SNN: Loss {avg_loss_snn:.4f}, Test Acc {snn_test_acc:5.2f}% ({t_snn:.1f}s)")
        
    print("\n" + "="*75)
    print("FINAL MNIST TEST RESULTS (10,000 DIGITS)")
    print("="*75)
    print(f"Classical Real MLP Test Accuracy:         {mlp_test_acc:.2f}%")
    print(f"Fused Superposition Net Test Accuracy:    {snn_test_acc:.2f}%")
    print(f"Delta:                                    {snn_test_acc - mlp_test_acc:+.2f}%")

if __name__ == '__main__':
    run_mnist_benchmark()
