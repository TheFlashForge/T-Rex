# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# Simplified diffusion model for DiffLM-RL demo

import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for time/sigma encoding."""

    def __init__(self, num_channels, max_positions=10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2)
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPBlock(nn.Module):
    """MLP block with residual connections."""

    def __init__(self, dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x


class MLPDiffusion(nn.Module):
    """MLP-based denoising network."""

    def __init__(self, d_in, dim_t=512, num_layers=1):
        super().__init__()
        self.proj = nn.Linear(d_in, dim_t)
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            MLPBlock(dim_t * 2, num_layers=num_layers),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

    def forward(self, x, noise_labels):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        x = self.proj(x) + emb
        return self.mlp(x)


class Precond(nn.Module):
    """Preconditioning wrapper for denoising function."""

    def __init__(self, denoise_fn, hid_dim, sigma_data=0.5):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):
        sigma = sigma.reshape(-1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        x_in = c_in * x
        F_x = self.denoise_fn_F(x_in, c_noise.flatten())
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class LatentDiffuser(nn.Module):
    """Latent diffusion model."""

    def __init__(self, latent_size, dim_noise=512, denoising_layers=5):
        super().__init__()
        self.denoise_fn = MLPDiffusion(latent_size, dim_t=dim_noise, num_layers=denoising_layers)
        self.precond_denoise_fn = Precond(self.denoise_fn, latent_size)

    def forward(self, x):
        # EDM loss
        rnd_normal = torch.randn(x.shape[0], device=x.device)
        sigma = (rnd_normal * 1.2 + (-1.2)).exp()
        weight = (sigma ** 2 + 0.5 ** 2) / (sigma * 0.5) ** 2
        n = torch.randn_like(x) * sigma.unsqueeze(1)
        D_yn = self.precond_denoise_fn(x + n, sigma)
        loss = weight.unsqueeze(1) * ((D_yn - x) ** 2)
        return loss.mean()

    @classmethod
    def load_from_pretrained(cls, model_path, device="cpu"):
        """Load pretrained diffusion model."""
        model = cls(latent_size=1024)
        # Implement loading based on your saved format
        return model.to(device)
