"""
Shared encoder/decoder components for reservoir-style neural control phases.

Provides reusable building blocks for the encode (game state -> stim rates) and
decode (spike rates -> game action) pipelines used by MsPacManPhase, AntPhaseV3,
and any future game-phase.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class PPORolloutBuffer:
    """Simple rollout storage for PPO-style updates."""

    def __init__(self):
        self.clear()

    def add(self, state, action, log_prob, value, reward, done):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.states)


# ---------------------------------------------------------------------------
# Encoding: game features -> sensory stimulation rates
# ---------------------------------------------------------------------------

class InputPolicyActorCritic(nn.Module):
    """Continuous action policy for learned sensory stimulation rates.

    Takes projected game features as input and outputs a vector of stimulation
    rates (one per sensory neuron).  Trained via PPO from game reward.
    """

    def __init__(self, input_size, action_size, hidden_size=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

    def latent_to_rates(self, latent_action, max_stim_hz):
        return max_stim_hz * torch.sigmoid(latent_action)

    def act(self, state_tensor, max_stim_hz, deterministic=False):
        with torch.no_grad():
            mean = self.actor(state_tensor)
            std = torch.exp(self.log_std.clamp(-5, 2))
            dist = Normal(mean, std)
            latent_action = mean if deterministic else dist.sample()
            log_prob = dist.log_prob(latent_action).sum(dim=-1)
            value = self.critic(state_tensor).squeeze(-1)
            rates = self.latent_to_rates(latent_action, max_stim_hz)

        return (
            latent_action.squeeze(0).cpu().numpy().astype(np.float32),
            rates.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def evaluate(self, states, actions):
        mean = self.actor(states)
        std = torch.exp(self.log_std.clamp(-5, 2))
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(states).squeeze(-1)
        return log_probs, entropy, values


# ---------------------------------------------------------------------------
# Decoding: spike rates -> game actions
# ---------------------------------------------------------------------------

class DiscreteOutputPPO(nn.Module):
    """Categorical PPO policy for discrete action spaces (e.g. Ms. Pac-Man)."""

    def __init__(self, input_size, action_size, hidden_size=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def act(self, state_tensor, deterministic=False):
        with torch.no_grad():
            logits = self.actor(state_tensor)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor).squeeze(-1)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate(self, states, actions):
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.long())
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)
        return log_probs, entropy, values


class ContinuousOutputPPO(nn.Module):
    """Gaussian PPO policy for continuous action spaces (e.g. Ant)."""

    def __init__(self, input_size, action_size, hidden_size=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

    def act(self, state_tensor, deterministic=False):
        with torch.no_grad():
            mean = self.actor(state_tensor)
            std = torch.exp(self.log_std.clamp(-5, 2))
            dist = Normal(mean, std)
            raw_action = mean if deterministic else dist.sample()
            action = torch.tanh(raw_action)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            # Tanh squashing correction
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
            value = self.critic(state_tensor).squeeze(-1)

        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def evaluate(self, states, actions):
        mean = self.actor(states)
        std = torch.exp(self.log_std.clamp(-5, 2))
        dist = Normal(mean, std)
        # Inverse tanh to recover raw actions for log_prob computation
        raw_actions = torch.atanh(actions.clamp(-0.999, 0.999))
        log_probs = dist.log_prob(raw_actions).sum(dim=-1)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(states).squeeze(-1)
        return log_probs, entropy, values


# ---------------------------------------------------------------------------
# GAE and PPO helpers
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation.

    All inputs should be array-like of the same length.
    Returns (advantages, returns) as np.float32 arrays.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    advantages = np.zeros_like(rewards)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        next_value = 0.0 if step == len(rewards) - 1 else values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae

    returns = advantages + values
    return advantages, returns


def compute_gradient_magnitude(module):
    """Compute L2 norm of gradients across all parameters."""
    total_norm = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return float(np.sqrt(total_norm))


def ppo_update(
    policy,
    optimizer,
    buffer,
    action_type,
    gamma,
    gae_lambda,
    clip_epsilon,
    ppo_epochs,
    ppo_minibatch_size,
    entropy_coef,
    value_coef,
    max_grad_norm,
):
    """Run a full PPO update on *policy* using data in *buffer*.

    Parameters
    ----------
    action_type : str
        ``"discrete"`` or ``"continuous"``.

    Returns the gradient magnitude from the last optimiser step.
    """
    if len(buffer) == 0:
        return 0.0

    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones, gamma, gae_lambda,
    )

    states = torch.as_tensor(np.asarray(buffer.states), dtype=torch.float32)
    old_log_probs = torch.as_tensor(np.asarray(buffer.log_probs), dtype=torch.float32)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    returns = torch.as_tensor(returns, dtype=torch.float32)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    if action_type == "discrete":
        actions = torch.as_tensor(np.asarray(buffer.actions), dtype=torch.long)
    else:
        actions = torch.as_tensor(np.asarray(buffer.actions), dtype=torch.float32)

    grad_norm = 0.0
    n_samples = len(buffer)
    batch_size = min(ppo_minibatch_size, n_samples)

    for _ in range(ppo_epochs):
        indices = np.random.permutation(n_samples)
        for batch_start in range(0, n_samples, batch_size):
            batch_inds = indices[batch_start: batch_start + batch_size]
            batch_states = states[batch_inds]
            batch_actions = actions[batch_inds]
            batch_old_log_probs = old_log_probs[batch_inds]
            batch_advantages = advantages[batch_inds]
            batch_returns = returns[batch_inds]

            new_log_probs, entropy, values = policy.evaluate(
                batch_states, batch_actions,
            )
            ratios = torch.exp(new_log_probs - batch_old_log_probs)
            clipped_ratios = torch.clamp(
                ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon,
            )
            actor_loss = -torch.min(
                ratios * batch_advantages, clipped_ratios * batch_advantages,
            ).mean()
            critic_loss = nn.functional.mse_loss(values, batch_returns)
            loss = (
                actor_loss
                + value_coef * critic_loss
                - entropy_coef * entropy.mean()
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            grad_norm = compute_gradient_magnitude(policy)
            optimizer.step()

    buffer.clear()
    return grad_norm
