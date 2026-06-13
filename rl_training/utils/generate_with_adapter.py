"""
Generate adversarial samples using trained DiffusionAdapter.

This script loads a trained adapter and uses it to generate samples
that are both realistic and adversarial to target models.
"""

import os
import json
import argparse

import torch
import numpy as np


def generate_with_adapter(
    adapter,
    diffusion_model,
    vae_model,
    num_samples=1000,
    num_steps=50,
    max_new_tokens=256,
    batch_size=32,
    device="cuda:0",
    vae_device="cpu",
):
    """Generate samples using adapter-guided diffusion."""
    from utils.diffusion import sample_with_adapter
    from utils.generation import generate_from_latent_vector

    adapter.eval()
    diffusion_model.eval()
    vae_model.eval()

    # Load mean_z from training
    # (In practice, this should be saved during diffusion training)
    mean_z = None

    samples = []
    decoder_tokenizer = vae_model.decoder_tokenizer
    bos_id = decoder_tokenizer.bos_token_id
    context_tokens = [bos_id] if bos_id is not None else []

    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

        # Sample initial noise
        rnd_priors = torch.stack([
            vae_model.prior.sample() for _ in range(current_batch_size)
        ]).to(device)

        # Diffusion with adapter
        latent_z, _ = sample_with_adapter(
            diffusion_model, adapter, rnd_priors,
            num_steps=num_steps, training=False
        )

        # Post-processing
        if mean_z is not None:
            latent_z = latent_z * 2 + mean_z

        # Decode
        with torch.no_grad():
            for i in range(current_batch_size):
                z = latent_z[i:i + 1].to(vae_device)
                try:
                    text = generate_from_latent_vector(
                        model=vae_model.decoder,
                        tokenizer=decoder_tokenizer,
                        context_tokens=context_tokens,
                        latent_vector=z,
                        device=vae_device,
                        temperature=0.5,
                        max_new_tokens=max_new_tokens,
                    )
                    sample = json.loads(text)
                    samples.append(sample)
                except Exception as e:
                    print(f"Warning: Failed to decode sample {batch_idx * batch_size + i}: {e}")
                    samples.append({})

        print(f"Generated {(batch_idx + 1) * current_batch_size}/{num_samples} samples")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate samples with adapter")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained adapter")
    parser.add_argument("--diffusion_model_path", type=str, required=True)
    parser.add_argument("--vae_model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_file", type=str, default="output/risky_samples.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load models
    from models.diffusion_adapter import create_adapter

    # Load adapter
    checkpoint = torch.load(args.adapter_path, map_location=args.device)
    adapter = create_adapter()  # Create with default params
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    adapter.to(args.device)

    # Load diffusion model and VAE (implement based on your model format)
    # diffusion_model, mean_z = load_diffusion(args.diffusion_model_path, args.device)
    # vae_model = load_vae(args.vae_model_path, args.device)

    # Generate samples
    # samples = generate_with_adapter(
    #     adapter, diffusion_model, vae_model,
    #     num_samples=args.num_samples,
    #     batch_size=args.batch_size,
    #     device=args.device
    # )

    # Save
    # with open(args.output_file, "w") as f:
    #     for sample in samples:
    #         f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Generated {args.num_samples} samples to {args.output_file}")


if __name__ == "__main__":
    main()
