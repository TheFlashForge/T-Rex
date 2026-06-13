"""
REINFORCE V3 - Improved RL Algorithm for DiffusionAdapter Training

Key improvements over V2:
1. Top-K selection for high-quality samples
2. Replay buffer for successful adversarial samples
3. Per-sample credit assignment
4. Entropy regularization for exploration
5. Gradient clipping for stability

This is the recommended algorithm for training the DiffusionAdapter.
"""

import os
import json
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class REINFORCEV3Trainer:
    """
    REINFORCE V3 trainer for DiffusionAdapter.

    Implements:
    - Top-K sample selection
    - Replay buffer
    - Per-sample credit assignment
    - Entropy regularization
    """

    def __init__(
        self,
        diffusion_model,
        vae_model,
        adapter,
        reward_computer,
        mean_z=None,
        device='cuda:0',
        vae_device='cpu',
        lr=1e-5,
        gamma=0.99,
        entropy_coef=0.01,
        grad_clip_norm=0.5,
        replay_size=256,
        replay_ratio=0.25,
        candidate_multiplier=2,
        top_k=8,
    ):
        self.diffusion_model = diffusion_model
        self.vae_model = vae_model
        self.adapter = adapter
        self.reward_computer = reward_computer
        self.mean_z = mean_z
        self.device = device
        self.vae_device = vae_device

        # RL parameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip_norm = grad_clip_norm

        # Sampling parameters
        self.replay_size = replay_size
        self.replay_ratio = replay_ratio
        self.candidate_multiplier = candidate_multiplier
        self.top_k = top_k

        # Freeze base models
        self.diffusion_model.eval()
        self.vae_model.eval()
        for param in self.diffusion_model.parameters():
            param.requires_grad = False
        for param in self.vae_model.parameters():
            param.requires_grad = False

        # Optimizer for adapter only
        self.optimizer = optim.Adam(self.adapter.parameters(), lr=lr)

        # Baseline for variance reduction
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.95

        # Replay buffer
        self.replay_buffer = []

        # Training history
        self.training_history = []

    def generate_samples_batch(self, batch_size, num_steps=50, max_new_tokens=256, initial_latents=None):
        """
        Generate samples using diffusion model with adapter.

        The adapter modifies the denoising direction at each step.
        """
        from utils.diffusion import sample_with_adapter
        from utils.generation import generate_from_latent_vector

        # Sample initial noise
        if initial_latents is None:
            rnd_priors = torch.stack([
                self.vae_model.prior.sample() for _ in range(batch_size)
            ]).to(self.device)
            init_latents = [rnd_priors[i].detach().cpu().numpy() for i in range(batch_size)]
        else:
            rnd_priors = torch.tensor(
                np.stack(initial_latents), device=self.device, dtype=torch.float32
            )
            init_latents = initial_latents

        # Diffusion sampling with adapter
        latent_z, log_probs_batch = sample_with_adapter(
            self.diffusion_model,
            self.adapter,
            rnd_priors,
            num_steps=num_steps,
            training=self.adapter.training,
        )

        # Post-processing
        if self.mean_z is not None:
            latent_z = latent_z * 2 + self.mean_z

        # VAE decoding
        samples = []
        decoder_tokenizer = self.vae_model.decoder_tokenizer
        bos_id = decoder_tokenizer.bos_token_id
        context_tokens = [bos_id] if bos_id is not None else []

        with torch.no_grad():
            for i in range(batch_size):
                latent_z_vae = latent_z[i:i + 1].to(self.vae_device)
                try:
                    decoded_text = generate_from_latent_vector(
                        model=self.vae_model.decoder,
                        tokenizer=decoder_tokenizer,
                        context_tokens=context_tokens,
                        latent_vector=latent_z_vae,
                        device=self.vae_device,
                        temperature=0.5,
                        max_new_tokens=max_new_tokens,
                    )
                    sample = json.loads(decoded_text)
                except Exception:
                    sample = {}
                samples.append(sample)

        return samples, log_probs_batch, init_latents

    def train_episode(self, batch_size=8, num_steps=50, max_new_tokens=256):
        """Train one episode with Top-K selection and replay buffer."""
        self.adapter.train()

        candidate_size = max(batch_size * self.candidate_multiplier, batch_size)
        top_k = min(self.top_k, candidate_size)

        samples = []
        log_probs_batch = []
        rewards = []
        infos = []
        init_latents = []

        # Replay samples
        replay_count = 0
        if self.replay_buffer:
            replay_count = int(candidate_size * self.replay_ratio)
            replay_count = min(replay_count, len(self.replay_buffer))

        if replay_count > 0:
            replay_indices = np.random.choice(len(self.replay_buffer), size=replay_count, replace=False)
            replay_latents = [self.replay_buffer[idx] for idx in replay_indices]

            replay_samples, replay_log_probs, replay_init_latents = self.generate_samples_batch(
                batch_size=replay_count,
                num_steps=num_steps,
                max_new_tokens=max_new_tokens,
                initial_latents=replay_latents
            )
            samples.extend(replay_samples)
            log_probs_batch.extend(replay_log_probs)
            init_latents.extend(replay_init_latents)

            for sample in replay_samples:
                reward, info = self.reward_computer.compute_reward(sample)
                rewards.append(reward)
                infos.append(info)

        # New samples
        new_count = candidate_size - replay_count
        if new_count > 0:
            new_samples, new_log_probs, new_init_latents = self.generate_samples_batch(
                batch_size=new_count,
                num_steps=num_steps,
                max_new_tokens=max_new_tokens,
            )
            samples.extend(new_samples)
            log_probs_batch.extend(new_log_probs)
            init_latents.extend(new_init_latents)

            for sample in new_samples:
                reward, info = self.reward_computer.compute_reward(sample)
                rewards.append(reward)
                infos.append(info)

        # Top-K selection
        top_indices = np.argsort(rewards)[-top_k:]

        # Update replay buffer with successful samples
        for idx in top_indices:
            if (infos[idx]['is_valid'] and
                infos[idx]['is_error'] == 1 and
                infos[idx]['authenticity'] >= self.reward_computer.auth_min):
                self.replay_buffer.append(init_latents[idx])

        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer = self.replay_buffer[-self.replay_size:]

        # Per-sample credit assignment
        log_prob_sums = []
        selected_rewards = []
        selected_infos = []

        for idx in top_indices:
            probs = log_probs_batch[idx]
            if probs:
                log_prob_sums.append(torch.stack(probs).sum())
                selected_rewards.append(rewards[idx])
                selected_infos.append(infos[idx])

        if not log_prob_sums:
            return self._empty_metrics()

        # Compute advantages
        mean_reward = float(np.mean(selected_rewards))
        advantages = [r - self.reward_baseline for r in selected_rewards]

        # Normalize advantages
        adv_std = np.std(advantages)
        if adv_std > 1e-8:
            advantages = [(a - np.mean(advantages)) / (adv_std + 1e-8) for a in advantages]

        # Update baseline
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline +
            (1 - self.baseline_momentum) * mean_reward
        )

        # Compute loss
        log_prob_tensor = torch.stack(log_prob_sums)
        adv_tensor = torch.tensor(advantages, device=log_prob_tensor.device, dtype=log_prob_tensor.dtype)

        # Policy gradient loss
        policy_loss = -(log_prob_tensor * adv_tensor).mean()

        # Entropy regularization (encourages exploration)
        entropy_loss = -self.entropy_coef * log_prob_tensor.mean()

        loss = policy_loss + entropy_loss

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()

        grad_norm_before = self._get_grad_norm()
        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=self.grad_clip_norm)
        grad_norm_after = self._get_grad_norm()

        self.optimizer.step()

        # Compute statistics
        reward_stats = {
            'num_positive': sum(1 for r in selected_rewards if r > 0),
            'num_zero': sum(1 for r in selected_rewards if abs(r) < 1e-6),
            'num_negative': sum(1 for r in selected_rewards if r < 0),
            'authenticity_mean': np.mean([
                info['authenticity'] for info in selected_infos if info['is_valid']
            ]) if any(info['is_valid'] for info in selected_infos) else 0.0,
            'error_rate': np.mean([
                info['is_error'] for info in selected_infos if info['is_valid']
            ]) if any(info['is_valid'] for info in selected_infos) else 0.0,
            'candidate_error_rate': float(np.mean([
                info['is_error'] for info in infos if info['is_valid']
            ])) if any(info['is_valid'] for info in infos) else 0.0,
        }

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'mean_reward': mean_reward,
            'baseline': self.reward_baseline,
            'grad_norm_before': grad_norm_before,
            'grad_norm_after': grad_norm_after,
            'reward_stats': reward_stats,
            'num_valid_samples': sum(1 for info in selected_infos if info['is_valid']),
            'replay_buffer_size': len(self.replay_buffer)
        }

    def evaluate(self, num_samples=50, num_steps=50, max_new_tokens=256):
        """Evaluate current adapter."""
        self.adapter.eval()

        rewards = []
        authenticities = []
        errors = []

        for _ in range(num_samples):
            samples, _, _ = self.generate_samples_batch(
                batch_size=1,
                num_steps=num_steps,
                max_new_tokens=max_new_tokens
            )
            sample = samples[0]

            if sample:
                reward, info = self.reward_computer.compute_reward(sample)
                if info['is_valid']:
                    rewards.append(reward)
                    authenticities.append(info['authenticity'])
                    errors.append(info['is_error'])

        return {
            'num_valid_samples': len(rewards),
            'mean_reward': np.mean(rewards) if rewards else 0.0,
            'mean_authenticity': np.mean(authenticities) if authenticities else 0.0,
            'error_rate': np.mean(errors) if errors else 0.0
        }

    def train(self, num_episodes=500, eval_every=50, save_every=100,
              output_dir='output/adapter_rl', batch_size=8,
              num_steps=50, max_new_tokens=256, early_stop_patience=10):
        """Main training loop."""
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()

        print(f"\n{'=' * 80}")
        print(f"REINFORCE V3 Training")
        print(f"{'=' * 80}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Episodes: {num_episodes}, Batch: {batch_size}, Top-K: {self.top_k}")
        print(f"Output: {output_dir}")

        best_reward = -float('inf')
        best_error_rate = 0.0
        best_eval_error_rate = 0.0
        no_improve_count = 0

        for episode in range(num_episodes):
            episode_start = time.time()
            train_metrics = self.train_episode(
                batch_size=batch_size,
                num_steps=num_steps,
                max_new_tokens=max_new_tokens
            )
            episode_time = time.time() - episode_start
            elapsed_time = time.time() - start_time

            self.training_history.append(train_metrics)

            current_error_rate = train_metrics['reward_stats']['error_rate']

            # Save best adversarial model
            if current_error_rate > best_error_rate:
                best_error_rate = current_error_rate
                torch.save({
                    'episode': episode + 1,
                    'adapter_state_dict': self.adapter.state_dict(),
                    'error_rate': current_error_rate,
                    'mean_reward': train_metrics['mean_reward'],
                }, os.path.join(output_dir, 'best_adversarial.pt'))
                print(f"  ✓ New best adversarial (error_rate={current_error_rate:.2%})")

            # Print status
            print(f"\n[Episode {episode + 1}/{num_episodes}]")
            print(f"  Loss: {train_metrics['loss']:.4f} | Reward: {train_metrics['mean_reward']:.4f}")
            print(f"  Auth: {train_metrics['reward_stats']['authenticity_mean']:.4f} | "
                  f"Error(sel): {train_metrics['reward_stats']['error_rate']:.2%} | "
                  f"Error(cand): {train_metrics['reward_stats']['candidate_error_rate']:.2%}")
            print(f"  Time: {episode_time:.1f}s | Elapsed: {elapsed_time / 60:.1f}min")

            # Periodic evaluation
            if (episode + 1) % eval_every == 0:
                eval_metrics = self.evaluate(
                    num_samples=50,
                    num_steps=num_steps,
                    max_new_tokens=max_new_tokens
                )
                eval_error_rate = eval_metrics['error_rate']

                print(f"\n  [Eval] Reward: {eval_metrics['mean_reward']:.4f} | "
                      f"Auth: {eval_metrics['mean_authenticity']:.4f} | "
                      f"Error: {eval_error_rate:.2%}")

                if eval_error_rate > best_eval_error_rate:
                    best_eval_error_rate = eval_error_rate
                    no_improve_count = 0
                    torch.save({
                        'episode': episode,
                        'adapter_state_dict': self.adapter.state_dict(),
                        'eval_metrics': eval_metrics,
                    }, os.path.join(output_dir, 'best_adapter.pt'))
                    print(f"  ✓ New best eval error_rate: {best_eval_error_rate:.2%}")
                else:
                    no_improve_count += 1
                    print(f"  No improvement ({no_improve_count}/{early_stop_patience})")

                if no_improve_count >= early_stop_patience:
                    print(f"\n⚠️ Early stopping at episode {episode + 1}")
                    break

                if eval_metrics['mean_reward'] > best_reward:
                    best_reward = eval_metrics['mean_reward']

            # Periodic checkpoint
            if (episode + 1) % save_every == 0:
                torch.save({
                    'episode': episode,
                    'adapter_state_dict': self.adapter.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(output_dir, f'checkpoint_ep{episode + 1}.pt'))

        # Save training history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"Training completed!")
        print(f"Best reward: {best_reward:.4f}")
        print(f"Best error rate: {best_error_rate:.2%}")
        print(f"{'=' * 80}")

    def _get_grad_norm(self):
        """Compute gradient norm."""
        total_norm = 0.0
        for p in self.adapter.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _empty_metrics(self):
        """Return empty metrics when no valid samples."""
        return {
            'loss': 0.0,
            'policy_loss': 0.0,
            'mean_reward': 0.0,
            'baseline': self.reward_baseline,
            'grad_norm_before': 0.0,
            'grad_norm_after': 0.0,
            'reward_stats': {
                'num_positive': 0, 'num_zero': 0, 'num_negative': 0,
                'authenticity_mean': 0.0, 'error_rate': 0.0,
                'candidate_error_rate': 0.0
            },
            'num_valid_samples': 0,
            'replay_buffer_size': len(self.replay_buffer)
        }


def main():
    parser = argparse.ArgumentParser(description="REINFORCE V3 for DiffusionAdapter")

    # Model paths
    parser.add_argument("--diffusion_model_path", type=str, required=True)
    parser.add_argument("--vae_model_path", type=str, required=True)
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument("--authenticity_model_path", type=str, required=True)

    # Output
    parser.add_argument("--output_dir", type=str, default="output/adapter_rl_v3")

    # Adapter parameters
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--adapter_rank", type=int, default=128)
    parser.add_argument("--max_delta", type=float, default=0.15)

    # Training parameters
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)

    # RL parameters
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--grad_clip_norm", type=float, default=0.5)

    # Sampling parameters
    parser.add_argument("--replay_size", type=int, default=256)
    parser.add_argument("--replay_ratio", type=float, default=0.25)
    parser.add_argument("--candidate_multiplier", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=8)

    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=10)

    # Device
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vae_device", type=str, default="cpu")

    args = parser.parse_args()

    # Load models (simplified - actual loading depends on your model format)
    print("Loading models...")

    # TODO: Implement model loading based on your setup
    # diffusion_model, mean_z = load_diffusion_model(args.diffusion_model_path, args.device)
    # vae_model = load_vae_model(args.vae_model_path, args.vae_device)
    # target_model = load_target_model(args.target_model_path)
    # authenticity_scorer = load_authenticity_scorer(args.authenticity_model_path)

    # Create adapter
    from models.diffusion_adapter import create_adapter
    adapter = create_adapter(
        adapter_type="standard",
        latent_dim=args.latent_dim,
        rank=args.adapter_rank,
        max_delta=args.max_delta
    )
    adapter = adapter.to(args.device)
    print(f"Adapter parameters: {adapter.get_num_params():,}")

    # Create reward computer
    # reward_computer = RewardComputer(authenticity_scorer, target_model)

    # Create trainer
    # trainer = REINFORCEV3Trainer(
    #     diffusion_model=diffusion_model,
    #     vae_model=vae_model,
    #     adapter=adapter,
    #     reward_computer=reward_computer,
    #     mean_z=mean_z,
    #     device=args.device,
    #     vae_device=args.vae_device,
    #     lr=args.lr,
    #     entropy_coef=args.entropy_coef,
    #     grad_clip_norm=args.grad_clip_norm,
    # )

    # Train
    # trainer.train(
    #     num_episodes=args.num_episodes,
    #     eval_every=args.eval_every,
    #     save_every=args.save_every,
    #     output_dir=args.output_dir,
    #     batch_size=args.batch_size,
    #     num_steps=args.num_steps,
    #     max_new_tokens=args.max_new_tokens,
    #     early_stop_patience=args.early_stop_patience,
    # )

    print("Training complete!")


if __name__ == "__main__":
    main()
