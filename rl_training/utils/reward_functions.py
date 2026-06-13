"""
Dual-Signal Reward Functions for RL Training

This module implements the reward function that combines:
1. Authenticity: How realistic the generated sample looks
2. Adversariality: How well the sample fools the target model

The reward design is crucial for training the DiffusionAdapter to generate
samples that are both realistic and adversarial.
"""

import numpy as np


class RewardComputer:
    """
    Computes rewards for RL training.

    Reward structure:
    - Low authenticity (< threshold): High penalty
    - High authenticity + Target model fails: High reward (success!)
    - High authenticity + Target model correct: Small penalty
    """

    def __init__(
        self,
        authenticity_scorer,
        target_model,
        auth_min=0.5,
        success_base=0.9,
        success_gain=0.1,
        fail_penalty=0.2,
        success_conf_weight=0.5
    ):
        """
        Args:
            authenticity_scorer: Model that predicts P(real) for a sample
            target_model: The model we want to fool
            auth_min: Minimum authenticity threshold
            success_base: Base reward for successful adversarial attack
            success_gain: Additional reward scaled by authenticity
            fail_penalty: Penalty multiplier for failed attacks
            success_conf_weight: Weight for confidence in success reward
        """
        self.authenticity_scorer = authenticity_scorer
        self.target_model = target_model
        self.auth_min = auth_min
        self.success_base = success_base
        self.success_gain = success_gain
        self.fail_penalty = fail_penalty
        self.success_conf_weight = success_conf_weight

    def compute_reward(self, generated_sample):
        """
        Compute reward for a generated sample.

        Args:
            generated_sample: Dict containing the generated data sample

        Returns:
            reward: Scalar reward value
            info: Dict with detailed information
        """
        info = {
            'authenticity': 0.0,
            'is_error': 0,
            'is_valid': False,
            'error_msg': None,
            'prob_true': None
        }

        if not generated_sample or not isinstance(generated_sample, dict):
            info['error_msg'] = "Invalid sample format"
            return -1.0, info

        try:
            # 1. Compute authenticity score
            authenticity_proba = self.authenticity_scorer.predict_proba(generated_sample)
            authenticity = authenticity_proba[0, 1]  # P(real)
            info['authenticity'] = float(authenticity)

            # 2. Target model prediction
            proba = self.target_model.predict_proba(generated_sample)[0]
            prediction = self.target_model.predict(generated_sample)[0]

            # Get true label (dataset-specific)
            true_label = self._get_true_label(generated_sample)
            is_error = int(prediction != true_label)
            info['is_error'] = is_error
            info['is_valid'] = True

            prob_true = float(proba[true_label])
            info['prob_true'] = prob_true

            # 3. Compute reward
            if authenticity < self.auth_min:
                # Too fake, penalize heavily
                reward = -1.0
            elif is_error == 1:
                # Successful adversarial attack!
                pred_conf = float(proba[int(prediction)])
                reward = self.success_base + self.success_gain * float(authenticity)
                # Scale by prediction confidence
                reward = reward * (
                    self.success_conf_weight +
                    (1 - self.success_conf_weight) * pred_conf
                )
            else:
                # Failed attack, penalize based on confidence
                reward = -self.fail_penalty * prob_true

            return reward, info

        except Exception as e:
            info['error_msg'] = str(e)
            return -1.0, info

    def _get_true_label(self, sample):
        """Get true label for the sample (dataset-specific)."""
        # Default: for income dataset
        if 'income' in sample:
            return 1 if sample.get('income', '') == '>50K' else 0
        # Add other dataset-specific logic here
        return 0


def compute_batch_rewards(samples, reward_computer):
    """
    Compute rewards for a batch of samples.

    Args:
        samples: List of generated samples
        reward_computer: RewardComputer instance

    Returns:
        rewards: List of reward values
        infos: List of info dicts
    """
    rewards = []
    infos = []

    for sample in samples:
        reward, info = reward_computer.compute_reward(sample)
        rewards.append(reward)
        infos.append(info)

    return rewards, infos


def compute_advantages(rewards, baseline, normalize=True):
    """
    Compute advantages for policy gradient.

    Args:
        rewards: List of reward values
        baseline: Baseline value (usually running average)
        normalize: Whether to normalize advantages

    Returns:
        advantages: List of advantage values
    """
    advantages = [r - baseline for r in rewards]

    if normalize:
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        if adv_std > 1e-8:
            advantages = [(a - adv_mean) / (adv_std + 1e-8) for a in advantages]

    return advantages
