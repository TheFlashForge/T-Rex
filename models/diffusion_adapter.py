"""
DiffusionAdapter - Core Innovation for RL-guided Diffusion

This module implements the LoRA-style adapter that is trained using RL
to guide the diffusion process towards generating adversarial samples.

Key features:
1. Low-rank decomposition (A @ B) for parameter efficiency
2. Sigma-aware conditioning (adapts to noise level)
3. Constrained delta to preserve sample quality
4. Support for REINFORCE probability distribution
"""

import torch
import torch.nn as nn
import numpy as np


class DiffusionAdapter(nn.Module):
    """
    LoRA-style adapter for diffusion model.

    The adapter learns a residual delta that adjusts the denoising direction
    to steer generation towards adversarial regions in latent space.

    Architecture:
        z_t → A (down-project) + sigma_embed → B (up-project) → delta
        denoised_adjusted = denoised + delta
    """

    def __init__(self, latent_dim=4096, rank=128, max_delta=0.15):
        super().__init__()

        self.latent_dim = latent_dim
        self.rank = rank
        self.max_delta = max_delta

        # Sigma encoder (condition on noise level)
        self.sigma_encoder = nn.Sequential(
            nn.Linear(1, rank),
            nn.SiLU(),
            nn.Linear(rank, rank)
        )

        # LoRA low-rank decomposition
        self.A = nn.Linear(latent_dim, rank, bias=False)
        self.B = nn.Linear(rank, latent_dim, bias=False)

        # Initialize B to zero (adapter starts as identity)
        nn.init.zeros_(self.B.weight)

        # Scaling factor
        self.scale = 0.01

        # Layer normalization
        self.layer_norm = nn.LayerNorm(latent_dim)

        # Learnable log_std for REINFORCE distribution
        # Initialized to -2.0, giving std ≈ 0.135
        self.log_std = nn.Parameter(torch.ones(latent_dim) * -2.0)

    def forward(self, z_t, t, return_dist=False):
        """
        Args:
            z_t: Current noisy latent [batch_size, latent_dim]
            t: Sigma value [batch_size] or scalar
            return_dist: If True, return probability distribution for REINFORCE

        Returns:
            delta: Adjustment [batch_size, latent_dim]
                   OR torch.distributions.Normal if return_dist=True
        """
        batch_size = z_t.shape[0]

        # Process sigma value
        if isinstance(t, (int, float)):
            t = torch.full((batch_size,), float(t), device=z_t.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        t = t.float().reshape(-1, 1)  # [batch_size, 1]

        # Encode sigma
        t_embed = self.sigma_encoder(t)  # [batch_size, rank]

        # Low-rank decomposition
        down = self.A(z_t)  # [batch_size, rank]

        # Add sigma embedding
        down = down + t_embed

        up = self.B(down)  # [batch_size, latent_dim]

        # Apply scaling
        delta = self.scale * up

        # Layer normalization
        delta = self.layer_norm(delta)

        # Clip delta to prevent excessive perturbation
        if self.max_delta > 0:
            delta = torch.clamp(delta, -self.max_delta, self.max_delta)

        # Return distribution for REINFORCE
        if return_dist:
            mu = delta
            std = torch.exp(self.log_std).clamp(min=0.01, max=1.0)
            dist = torch.distributions.Normal(mu, std)
            return dist
        else:
            return delta

    def get_num_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class MultiScaleAdapter(nn.Module):
    """
    Multi-scale adapter using multiple adapters with different ranks.
    Captures both fine-grained and coarse adjustments.
    """

    def __init__(self, latent_dim=4096, ranks=[32, 64, 128]):
        super().__init__()

        self.adapters = nn.ModuleList([
            DiffusionAdapter(latent_dim, rank)
            for rank in ranks
        ])

        # Fusion layer
        self.fusion = nn.Linear(len(ranks) * latent_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, z_t, t):
        deltas = [adapter(z_t, t) for adapter in self.adapters]
        fused = torch.cat(deltas, dim=-1)
        delta = self.fusion(fused)
        delta = self.layer_norm(delta)
        return delta

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def create_adapter(adapter_type="standard", latent_dim=4096, rank=128, max_delta=0.15):
    """Factory function to create adapter."""
    if adapter_type == "standard":
        return DiffusionAdapter(latent_dim, rank, max_delta=max_delta)
    elif adapter_type == "multi_scale":
        return MultiScaleAdapter(latent_dim, [rank // 2, rank, rank * 2])
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


if __name__ == "__main__":
    # Quick test
    adapter = DiffusionAdapter(latent_dim=1024, rank=64)
    z_t = torch.randn(4, 1024)
    t = torch.tensor([0.5, 1.0, 2.0, 4.0])

    # Test forward
    delta = adapter(z_t, t)
    print(f"Input: {z_t.shape}, Output: {delta.shape}")
    print(f"Delta range: [{delta.min():.4f}, {delta.max():.4f}]")
    print(f"Parameters: {adapter.get_num_params():,}")

    # Test distribution
    dist = adapter(z_t, t, return_dist=True)
    sample = dist.sample()
    log_prob = dist.log_prob(sample).sum(dim=-1)
    print(f"Sample: {sample.shape}, Log prob: {log_prob.shape}")
