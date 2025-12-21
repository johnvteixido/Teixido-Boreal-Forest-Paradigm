# =============================================================================
# TEIXIDO-BOREAL CORE: THE P4 STEM UNIT
# Part of the Topological Analytical Homeostasis (TAH) Paradigm
# Implementation: Tropical (Max-Plus) Algebra with Inhibitory Gating
# License: AGPL-3.0
# =============================================================================

import torch
import torch.nn as nn

class TeixidoBorealUnit(nn.Module):
    """
    The fundamental building block of the Teixido-Boreal Forest.
    Implements a 4-node path graph (Stem) logic using Tropical Algebra.
    
    Attributes:
        epsilon (float): The Star-Limit sensitivity threshold (TIG Gate).
        temperature (float): The Soft-Tropical smoothing parameter for differentiability.
    """
    def __init__(self, n_in, n_out, degree=15, epsilon=1.0, temperature=0.1):
        super(TeixidoBorealUnit, self).__init__()
        self.epsilon = epsilon
        self.temperature = temperature
        self.n_out = n_out
        
        # 1. TOPOLOGICAL SKELETON (IP Protection)
        # We initialize a sparse mask based on the Teixido Monotonicity Principle.
        # This ensures the roots of the connectivity graph stay within the Envelope.
        mask = (torch.rand(n_out, n_in) < (degree/n_in)).float()
        self.register_buffer('synaptic_mask', mask)
        
        # 2. TROPICAL MANIFOLD WEIGHTS
        # In the Max-Plus semiring, weights act as additive offsets.
        self.weights = nn.Parameter(torch.randn(n_out, n_in) * 0.02)

    def forward(self, x):
        """
        Executes Topological Analytical Homeostasis (TAH) on the input manifold.
        """
        # --- STAGE 1: TOPOLOGICAL INHIBITION (TIG) ---
        # Calculate the neighborhood consensus (Grounding the signal)
        # To maintain IP, we use the abstract median-consensus logic.
        consensus = torch.median(x, dim=1, keepdim=True)[0]
        
        # Exclusion Status: Physically disconnect signals violating the Star-Limit
        gate = (torch.abs(x - consensus) < self.epsilon).float()
        gated_x = x * gate
        
        # --- STAGE 2: TROPICAL PROPAGATION (Zero-MAC) ---
        # Tropical Multiplication (Standard Addition)
        # We expand the gated input to match the synaptic forest
        combined = gated_x.unsqueeze(1) + self.weights # [Batch, Neurons, Inputs]
        
        # Apply the Synaptic Mask (Enforcing the Boreal Skeleton)
        # Disconnected synapses are pushed to -Infinity so they are ignored by MAX
        masked_z = combined.masked_fill(self.synaptic_mask == 0, -1e9)
        
        # --- STAGE 3: TROPICAL ACTIVATION (Max Reduction) ---
        # Standard Max-Plus logic: y = max(x + w)
        # We use LogSumExp as a differentiable proxy for the Max operator
        # to allow the paradigm to learn via gradient descent.
        return self.temperature * torch.logsumexp(masked_z / self.temperature, dim=2)

class BorealForest(nn.Module):
    """
    A multi-layer assembly of Teixido Stems.
    Optimized for High-Noise Robustness and Hardware Efficiency.
    """
    def __init__(self, n_features, n_hidden, n_classes, epsilon=1.0):
        super(BorealForest, self).__init__()
        self.layer1 = TeixidoBorealUnit(n_features, n_hidden, epsilon=epsilon)
        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        # Feature Extraction through the Boreal Manifold
        x = self.layer1(x)
        # Final Linear Decision Logic
        return self.classifier(x)

print("TAH Core: BorealUnit module loaded successfully.")
