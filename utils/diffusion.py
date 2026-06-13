# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import torch
import numpy as np


# EDM sampling parameters
SIGMA_MIN = 0.002
SIGMA_MAX = 80
RHO = 7
S_CHURN = 1
S_MIN = 0
S_MAX = float("inf")
S_NOISE = 1


@torch.inference_mode()
def sample(net, latent_vector, num_steps=50, device="cuda:0"):
    """Standard diffusion sampling using Heun's method."""
    latents = torch.tensor(latent_vector, device=device, dtype=torch.float32)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (
        sigma_max ** (1 / RHO) + step_indices / (num_steps - 1) *
        (sigma_min ** (1 / RHO) - sigma_max ** (1 / RHO))
    ) ** RHO
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_next = _sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next


@torch.inference_mode()
def _sample_step(net, num_steps, i, t_cur, t_next, x_next):
    """Single sampling step with 2nd order correction."""
    x_cur = x_next

    gamma = min(S_CHURN / num_steps, np.sqrt(2) - 1) if S_MIN <= t_cur <= S_MAX else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_NOISE * torch.randn_like(x_cur)

    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def sample_with_adapter(diffusion_model, adapter, initial_noise, num_steps=50, training=False):
    """
    Diffusion sampling with adapter adjustment.

    The adapter adds a learned residual to the denoising direction,
    steering generation towards adversarial regions.

    Returns:
        latent_z: Generated latent vectors
        log_probs: Log probabilities for REINFORCE (if training)
    """
    device = initial_noise.device
    batch_size = initial_noise.shape[0]

    # EDM schedule
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min = max(SIGMA_MIN, diffusion_model.precond_denoise_fn.sigma_min)
    sigma_max = min(SIGMA_MAX, diffusion_model.precond_denoise_fn.sigma_max)

    t_steps = (
        sigma_max ** (1 / RHO) + step_indices / (num_steps - 1) *
        (sigma_min ** (1 / RHO) - sigma_max ** (1 / RHO))
    ) ** RHO
    t_steps = torch.cat([
        diffusion_model.precond_denoise_fn.round_sigma(t_steps),
        torch.zeros_like(t_steps[:1])
    ])

    # Initialize
    x_next = initial_noise.to(torch.float32) * t_steps[0]
    log_probs_batch = [[] for _ in range(batch_size)]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Add noise (Heun's method)
        gamma = min(S_CHURN / num_steps, np.sqrt(2) - 1) if S_MIN <= t_cur <= S_MAX else 0
        t_hat = diffusion_model.precond_denoise_fn.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_NOISE * torch.randn_like(x_cur)

        # Denoise with base model
        denoised = diffusion_model.precond_denoise_fn(x_hat, t_hat).to(torch.float32)

        # Apply adapter adjustment
        if training:
            dist = adapter(x_hat, t_hat, return_dist=True)
            delta = dist.sample()
            log_prob = dist.log_prob(delta).sum(dim=-1)
            for j in range(batch_size):
                log_probs_batch[j].append(log_prob[j])
        else:
            delta = adapter(x_hat, t_hat, return_dist=False)

        denoised_adjusted = denoised + delta

        # Euler step
        d_cur = (x_hat - denoised_adjusted) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd order correction
        if i < num_steps - 1 and t_next > 0:
            denoised = diffusion_model.precond_denoise_fn(x_next, t_next).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    log_probs = log_probs_batch if training else None
    return x_next, log_probs
