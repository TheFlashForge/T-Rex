# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# Simplified VAE model for DiffLM-RL demo

import os
import json
import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer


class BertEncoder(nn.Module):
    """BERT-based encoder for VAE."""

    def __init__(self, model_name="bert-base-uncased", latent_size=1024):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.latent_linear = nn.Linear(self.bert.config.hidden_size, latent_size * 2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_output

    def encode(self, input_ids, attention_mask=None):
        encoded = self.forward(input_ids, attention_mask)
        mu, logvar = self.latent_linear(encoded).chunk(2, -1)
        return mu, logvar


class GPT2Decoder(nn.Module):
    """GPT-2 based decoder with soft prompt injection."""

    def __init__(self, model_name="gpt2", latent_size=1024, adapter_size=32):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.latent_to_soft_prompt = nn.Sequential(
            nn.Linear(latent_size, self.gpt2.config.n_embd * adapter_size),
            nn.Tanh(),
            nn.Linear(self.gpt2.config.n_embd * adapter_size, self.gpt2.config.n_embd * adapter_size),
        )
        self.adapter_size = adapter_size

    def forward(self, input_ids, latent_vector, attention_mask=None, labels=None):
        inputs_embeds = self.gpt2.transformer.wte(input_ids)

        if latent_vector is not None:
            soft_prompt = self.latent_to_soft_prompt(latent_vector)
            soft_prompt = soft_prompt.view(-1, self.adapter_size, self.gpt2.config.n_embd)
            inputs_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)

            if attention_mask is not None:
                prefix_mask = torch.ones(soft_prompt.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            if labels is not None:
                prefix_labels = torch.full((soft_prompt.shape[0], soft_prompt.shape[1]), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"logits": logits, "loss": loss}


class VAELanguageModel(nn.Module):
    """VAE for text generation with latent diffusion support."""

    def __init__(self, encoder, decoder, latent_size=1024):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size

        # Standard normal prior
        self.prior = torch.distributions.normal.Normal(
            loc=torch.zeros(latent_size),
            scale=torch.ones(latent_size)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, text, tokenizer, device="cpu"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        mu, logvar = self.encoder.encode(inputs["input_ids"], inputs["attention_mask"])
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, tokenizer, max_new_tokens=256, temperature=0.5, device="cpu"):
        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        generated_ids = [bos_id]

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
            outputs = self.decoder(input_ids, z)
            logits = outputs["logits"]

            # Apply temperature
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            generated_ids.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    @classmethod
    def load_from_pretrained(cls, model_path, device="cpu"):
        """Load pretrained VAE model."""
        # Simplified loading - implement based on your saved format
        encoder = BertEncoder()
        decoder = GPT2Decoder()
        model = cls(encoder, decoder)
        return model.to(device)
