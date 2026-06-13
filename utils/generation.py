# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import torch
from tqdm import tqdm


@torch.inference_mode()
def generate_from_latent_vector(
    model,
    tokenizer,
    context_tokens,
    latent_vector,
    temperature=1.0,
    max_new_tokens=256,
    device="cpu",
):
    """Generate text from latent vector using autoregressive decoding."""
    generated_ids = list(context_tokens)

    for _ in tqdm(range(max_new_tokens), desc="Generating", leave=False):
        outputs = model(
            input_ids=torch.as_tensor([generated_ids], dtype=torch.long, device=device),
            latent_vector=latent_vector,
            use_cache=False,
        )
        logits = outputs.logits

        # Apply temperature
        last_token_logits = logits[0, -1, :] / temperature

        # Sample
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        generated_ids.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
