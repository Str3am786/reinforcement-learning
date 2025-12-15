import copy
import collections
import random


import numpy as np
import torch
import torch.nn.functional as F

import utils.common_utils as cu
from .ddpg_agent import DDPGAgent


def to_numpy(tensor: torch.Tensor):
    """Helper: convert torch tensor to numpy array."""
    return tensor.cpu().numpy().flatten()


def safe_cfg_get(cfg, key, default_val):
    if cfg is None:
        return default_val
    if isinstance(cfg, dict):
        return cfg.get(key, default_val)
    return getattr(cfg, key, default_val)


class DDPGExtension(DDPGAgent):
    """
    DDPG + conservative Self-Imitation Learning (SIL).

    Design goals:
    - Baseline DDPG behaviour stays almost identical.
    - SIL is a *small, safe* extra term:
        * only on clearly good episodes,
        * starts late,
        * small loss weight.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.extension_implemented = "SIL"

        # ------------------------------------------------------------------
        # 1. Exploration noise (do NOT change baseline by default)
        # ------------------------------------------------------------------
        # Baseline DDPGAgent sets:
        #   self.initial_noise = 0.2 * self.max_action
        #   self.final_noise   = 0.05 * self.max_action
        #   self.noise_decay_steps = 20000
        #
        # Here we *keep those values* unless the config explicitly overrides
        # them. That means: same behaviour as baseline, unless you want to
        # experiment by passing these in the config.
        self.initial_noise = safe_cfg_get(
            self.cfg, "initial_noise", self.initial_noise
        )
        self.final_noise = safe_cfg_get(
            self.cfg, "final_noise", self.final_noise
        )
        self.noise_decay_steps = safe_cfg_get(
            self.cfg, "noise_decay_steps", self.noise_decay_steps
        )

        # ------------------------------------------------------------------
        # 2. SIL hyperparameters (very conservative defaults)
        # ------------------------------------------------------------------
        # How many *episodes* to sample when building a SIL minibatch
        self.sil_batch_size = safe_cfg_get(self.cfg, "sil_batch_size", 32)

        # Weight of SIL term in actor loss – keep small
        self.sil_loss_weight = safe_cfg_get(self.cfg, "sil_loss_weight", 0.02)

        # Buffer of GOOD episodes only
        self.trajectory_buffer_max_size = safe_cfg_get(
            self.cfg, "trajectory_buffer_max_size", 2000
        )
        self.good_buffer = collections.deque(
            maxlen=self.trajectory_buffer_max_size
        )

        # Only store episodes with total return > min_sil_reward as "good"
        # In your env, 1.0 means at least one good green hit, so 1.0 is a
        # "genuinely useful" threshold by default.
        self.min_sil_reward = safe_cfg_get(self.cfg, "min_sil_reward", 1.0)

        # Start SIL only after critic & actor have learned something reasonable
        self.enable_sil_after_steps = safe_cfg_get(
            self.cfg, "enable_sil_after_steps", 30000
        )

        # Optional margin for advantage:
        # require G_t - V(s) > margin to imitate (default 0.1)
        self.sil_advantage_margin = safe_cfg_get(
            self.cfg, "sil_advantage_margin", 0.1
        )

    # ------------------------------------------------------------------
    # SIL helper: Monte Carlo returns for one episode
    # ------------------------------------------------------------------
    def _calculate_sil_returns(self, episode):
        """
        Episode: list of (obs, action_np, next_obs, reward, done_bool).

        We assume done_bool already encodes truncation vs. true termination,
        same as in train_iteration (0.0 if time-limit, 1.0 if real terminal).
        """
        rewards = [step[3] for step in episode]
        dones = [step[4] for step in episode]

        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        current_return = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]  # = 0.0 if real terminal, else 1.0
            current_return = rewards[t] + self.gamma * current_return * mask
            returns[t] = current_return

        return returns

    # ------------------------------------------------------------------
    # Override update() to add SIL stats but keep warmup logic identical
    # ------------------------------------------------------------------
    
    def update(self):
        """
        Same warmup & per-transition update logic as DDPGAgent.update(),
        but calls our _update() which includes an extra SIL term.
        """
        if self.buffer_ptr < self.random_transition:
            return {
                "critic_loss": float("nan"),
                "actor_loss": float("nan"),
                "actor_loss_sil": float("nan"),
            }

        update_iter = self.buffer_ptr - self.buffer_head

        critic_losses = []
        actor_losses = []
        sil_losses = []

        for _ in range(update_iter):
            info = self._update()
            critic_losses.append(info["critic_loss"])
            actor_losses.append(info["actor_loss"])
            sil_losses.append(info["actor_loss_sil"])

        self.buffer_head = self.buffer_ptr

        if not critic_losses:
            return {
                "critic_loss": float("nan"),
                "actor_loss": float("nan"),
                "actor_loss_sil": float("nan"),
            }

        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "actor_loss_sil": float(np.mean(sil_losses)),
        }

    # ------------------------------------------------------------------
    # Core update: DDPG critic + DDPG actor + (optional) SIL actor term
    # ------------------------------------------------------------------
    
    def _update(self):
        """
        Single gradient step:

        1) Standard DDPG critic update (unchanged).
        2) Standard DDPG actor update (unchanged).
        3) Extra actor loss from SIL using good_buffer, if enabled.
        """
        device = self.device

        # --- 1. Standard DDPG batch ---
        batch = self.buffer.sample(self.batch_size, device=device)

        # Critic (same as in base agent)
        current_q = self.q(batch.state, batch.action)
        target_q = self.calculate_target(batch)
        critic_loss = self.calculate_critic_loss(current_q, target_q)

        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        # Actor (same as in base agent)
        actor_loss = self.calculate_actor_loss(batch=batch)

        # --- 2. SIL actor term (conservative) ---
        actor_loss_sil = torch.tensor(0.0, device=device)

        # Conditions to run SIL:
        can_use_sil = (
            (self.total_steps >= self.enable_sil_after_steps)
            and (len(self.good_buffer) > 0)
        )

        if can_use_sil:
            # We may have fewer episodes than sil_batch_size; be robust to that.
            n_episodes = min(self.sil_batch_size, len(self.good_buffer))
            episodes = random.sample(self.good_buffer, n_episodes)

            sil_states = []
            sil_actions = []
            sil_returns = []

            for ep in episodes:
                G = self._calculate_sil_returns(ep)
                for (obs, act_np, next_obs, r, done_bool), G_t in zip(ep, G):
                    sil_states.append(obs)
                    sil_actions.append(act_np)
                    sil_returns.append(G_t)

            sil_states = torch.tensor(
                np.array(sil_states), dtype=torch.float32, device=device
            )
            sil_actions = torch.tensor(
                np.array(sil_actions), dtype=torch.float32, device=device
            )
            sil_returns = torch.tensor(
                np.array(sil_returns), dtype=torch.float32, device=device
            ).unsqueeze(1)

            with torch.no_grad():
                # V(s) ≈ Q(s, pi(s))
                v_est = self.q(sil_states, self.pi(sil_states))
                advantage = sil_returns - v_est

                if self.sil_advantage_margin > 0.0:
                    mask = (advantage > self.sil_advantage_margin).float()
                else:
                    mask = (advantage > 0.0).float()

            if mask.sum().item() > 0:
                pi_actions = self.pi(sil_states)
                per_action_mse = F.mse_loss(
                    pi_actions, sil_actions, reduction="none"
                )
                per_sample_mse = per_action_mse.mean(dim=1, keepdim=True)

                # Normalize by number of positive-advantage samples so the
                # scale is stable as mask changes.
                actor_loss_sil = (
                    (per_sample_mse * mask).sum()
                    / (mask.sum() + 1e-6)
                )

        # --- 3. Combined actor loss ---
        total_actor_loss = actor_loss + self.sil_loss_weight * actor_loss_sil

        self.pi_optim.zero_grad()
        total_actor_loss.backward()
        self.pi_optim.step()

        # --- 4. Soft updates, same as DDPG ---
        cu.soft_update_params(self.q, self.q_target, self.tau)
        cu.soft_update_params(self.pi, self.pi_target, self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "actor_loss_sil": float(actor_loss_sil.item()),
        }

    # ------------------------------------------------------------------
    # train_iteration: same as DDPG, but store good episodes for SIL
    # ------------------------------------------------------------------
    
    def train_iteration(self):
        """
        Equivalent structure to DDPGAgent.train_iteration, plus:

        - We record each episode as a trajectory.
        - If its total return > min_sil_reward, we push it into good_buffer
          for future SIL updates.
        """
        reward_sum, timesteps, done = 0, 0, False
        current_episode_trajectory = []

        obs, _ = self.env.reset()

        while not done:
            # Use base get_action (with our possibly overridden noise params)
            action, _ = self.get_action(obs)

            np_action = to_numpy(action)
            next_obs, reward, done, _, _ = self.env.step(np_action)

            done_bool = float(done) if timesteps < self.max_episode_steps else 0.0

            # Standard DDPG replay buffer
            self.record(obs, action, next_obs, reward, done_bool)

            # Also store for SIL
            current_episode_trajectory.append(
                (obs, np_action, next_obs, reward, done_bool)
            )

            reward_sum += reward
            timesteps += 1
            if timesteps >= self.max_episode_steps:
                done = True

            obs = next_obs.copy()

        # Update (DDPG + SIL)
        info = self.update()

        # Store only actually good episodes
        if reward_sum > self.min_sil_reward:
            self.good_buffer.append(current_episode_trajectory)

        info.update(
            {
                "episode_length": timesteps,
                "ep_reward": reward_sum,
            }
        )
        return info
