# =============================================================================
# TEIXIDO RESILIENCE GAP BENCHMARK
# Part of the Topological Analytical Homeostasis (TAH) Paradigm
# Purpose: Side-by-side comparison of TBM vs. Standard Linear Models
# under high-magnitude impulse noise (Total Anarchy).
# License: AGPL-3.0
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from core.boreal_unit import BorealForest

def run_resilience_benchmark():
    print("--- [TAH BENCHMARK] Initializing Resilience Battle ---")
    
    # Configuration
    n_in, n_hidden, n_out = 784, 128, 10
    batch_size = 64
    
    # 1. Initialize Competitors
    # Teixido Model (Sparse, Tropical, Inhibited)
    teixido_model = BorealForest(n_in, n_hidden, n_out, epsilon=1.0)
    
    # Standard Model (Dense, Linear, ReLU) - The 'Fragile Giant'
    standard_model = nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_out)
    )
    
    # 2. Generate Synthetic Data (Clean)
    # We simulate a high-dimensional classification task
    X_train = torch.randn(500, n_in)
    y_train = torch.randint(0, n_out, (500,))
    
    X_test = torch.randn(100, n_in)
    y_test = torch.randint(0, n_out, (100,))

    # 3. Train both on CLEAN data
    print("Training models on clean environment...")
    criterion = nn.CrossEntropyLoss()
    t_opt = optim.Adam(teixido_model.parameters(), lr=0.01)
    s_opt = optim.Adam(standard_model.parameters(), lr=0.01)
    
    for epoch in range(10):
        # Teixido Step
        t_opt.zero_grad()
        t_loss = criterion(teixido_model(X_train), y_train)
        t_loss.backward()
        t_opt.step()
        
        # Standard Step
        s_opt.zero_grad()
        s_loss = criterion(standard_model(X_train), y_train)
        s_loss.backward()
        s_opt.step()

    # 4. COMMENCE ATTACK (30% 'Total Anarchy' Impulse Noise)
    print("\n[ATTACK] Injecting 30% Impulse Noise Spikes (Magnitude 10.0)...")
    X_noisy = X_test.clone()
    noise_mask = torch.rand(X_noisy.shape) < 0.3
    X_noisy[noise_mask] = 10.0 # Extreme outliers
    
    # 5. Evaluate Stability
    teixido_model.eval()
    standard_model.eval()
    
    with torch.no_grad():
        # Clean Accuracies
        t_clean_acc = (teixido_model(X_test).argmax(1) == y_test).float().mean().item()
        s_clean_acc = (standard_model(X_test).argmax(1) == y_test).float().mean().item()
        
        # Noisy Accuracies
        t_noisy_acc = (teixido_model(X_noisy).argmax(1) == y_test).float().mean().item()
        s_noisy_acc = (standard_model(X_noisy).argmax(1) == y_test).float().mean().item()

    # 6. CALCULATE THE RESILIENCE GAP
    # The gap is the difference in robustness between the two paradigms
    t_drop = t_clean_acc - t_noisy_acc
    s_drop = s_clean_acc - s_noisy_acc
    resilience_gap = s_drop - t_drop

    print("\n" + "="*45)
    print("       RESILIENCE GAP REPORT")
    print("="*45)
    print(f"Standard Model Noise Drop:  {s_drop*100:.1f}%")
    print(f"Teixido Model Noise Drop:   {t_drop*100:.1f}%")
    print("-" * 45)
    print(f"THE TEIXIDO RESILIENCE GAP: {resilience_gap*100:.1f}%")
    print("-" * 45)
    print("VERDICT: " + ("REVOLUTIONARY" if resilience_gap > 0.2 else "ITERATIVE"))
    print("="*45)

if __name__ == "__main__":
    run_resilience_benchmark()
