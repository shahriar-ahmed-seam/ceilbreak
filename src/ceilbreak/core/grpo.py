"""Group Relative Policy Optimization (GRPO) Engine.

Implements online advantage calculation without a separate value/critic network by
normalizing rewards across groups of G rollouts per prompt:
    A_i = (r_i - mean(r)) / (std(r) + eps)

Directly resolves ReST-EM failure modes:
- Penalizes easy failures (negative advantage downweights errors)
- Highly rewards rare hard wins (high positive advantage)
- Zero-advantages mastered problems (eliminating difficulty skew)
- Integrates clipped surrogate objective and reference policy KL regularization.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from ceilbreak.core.types import (
    AdvantageEstimate,
    RolloutGroup,
    Trajectory,
)


class GRPOMetrics(BaseModel):
    """Output metrics from a GRPO optimization step."""
    group_count: int
    total_rollouts: int
    mean_advantage: float
    advantage_std: float
    clipped_ratio: float
    mean_surrogate_loss: float
    mean_kl_divergence: float
    easy_failures_penalized: int
    hard_wins_rewarded: int
    trivial_tasks_zeroed: int


class GRPOEngine:
    """Group Relative Policy Optimization advantage and loss computer."""

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.04,
        eps: float = 1e-6,
    ):
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff
        self.eps = eps

    def compute_group_advantages(self, group: RolloutGroup) -> List[AdvantageEstimate]:
        """Calculates normalized relative surprise advantages for all rollouts in group."""
        if not group.trajectories:
            return []

        group.compute_group_statistics()
        rewards = np.array([t.primary_reward for t in group.trajectories], dtype=np.float64)
        mean_r = float(group.reward_mean)
        std_r = float(group.reward_std)

        estimates: List[AdvantageEstimate] = []

        # If variance is zero (all succeeded or all failed), advantages are zero
        if std_r < self.eps:
            for t in group.trajectories:
                estimates.append(
                    AdvantageEstimate(
                        trajectory_id=t.trajectory_id,
                        prompt_id=group.prompt.prompt_id,
                        raw_reward=t.primary_reward,
                        group_mean=mean_r,
                        group_std=std_r,
                        advantage=0.0,
                        is_easy_failure=False,
                        is_hard_win=False,
                    )
                )
            return estimates

        # Standard group-relative advantage
        norm_advantages = (rewards - mean_r) / (std_r + self.eps)

        for i, t in enumerate(group.trajectories):
            adv = float(norm_advantages[i])
            is_easy_fail = (mean_r >= 0.7) and (t.primary_reward < 0.5)
            is_hard_win = (mean_r <= 0.3) and (t.primary_reward >= 0.7)

            estimates.append(
                AdvantageEstimate(
                    trajectory_id=t.trajectory_id,
                    prompt_id=group.prompt.prompt_id,
                    raw_reward=t.primary_reward,
                    group_mean=mean_r,
                    group_std=std_r,
                    advantage=adv,
                    is_easy_failure=is_easy_fail,
                    is_hard_win=is_hard_win,
                )
            )

        return estimates

    def compute_surrogate_loss(
        self,
        trajectories: List[Trajectory],
        advantages: List[AdvantageEstimate],
        current_policy_logprobs: Optional[List[List[float]]] = None,
    ) -> Tuple[float, float, float]:
        """Computes clipped surrogate objective and token-level KL divergence.

        Returns (surrogate_loss, kl_div, fraction_clipped).
        """
        if not trajectories or not advantages:
            return 0.0, 0.0, 0.0

        adv_map = {a.trajectory_id: a.advantage for a in advantages}
        losses: List[float] = []
        kl_values: List[float] = []
        clipped_flags: List[bool] = []

        for idx, traj in enumerate(trajectories):
            adv = adv_map.get(traj.trajectory_id, 0.0)

            # Approximate token-level ratio pi_curr / pi_old
            if current_policy_logprobs and idx < len(current_policy_logprobs):
                curr_lp = np.array(current_policy_logprobs[idx], dtype=np.float64)
            else:
                curr_lp = np.array(traj.token_logprobs, dtype=np.float64)

            old_lp = np.array(traj.token_logprobs, dtype=np.float64)

            if len(curr_lp) > 0 and len(old_lp) == len(curr_lp):
                # Importance sampling ratio r_t(theta) = exp(log pi_curr - log pi_old)
                log_ratio = curr_lp - old_lp
                ratio = np.exp(np.clip(log_ratio, -10.0, 10.0))
                mean_ratio = float(np.mean(ratio))
            else:
                mean_ratio = 1.0

            # Clipped objective
            surr1 = mean_ratio * adv
            clipped_ratio = np.clip(mean_ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            surr2 = clipped_ratio * adv
            loss_i = -min(surr1, surr2)
            losses.append(loss_i)

            is_clipped = bool(abs(mean_ratio - clipped_ratio) > 1e-4)
            clipped_flags.append(is_clipped)

            # KL divergence against reference policy: log pi_curr - log pi_ref
            if traj.ref_logprobs and len(traj.ref_logprobs) == len(curr_lp):
                ref_lp = np.array(traj.ref_logprobs, dtype=np.float64)
                # D_KL approximation: exp(ref - curr) - (ref - curr) - 1 (Schulman form)
                diff = curr_lp - ref_lp
                kl_approx = np.mean(np.exp(-diff) + diff - 1.0)
                kl_values.append(float(max(0.0, kl_approx)))
            else:
                kl_values.append(0.0)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_kl = float(np.mean(kl_values)) if kl_values else 0.0
        clip_frac = float(np.mean(clipped_flags)) if clipped_flags else 0.0

        return mean_loss, mean_kl, clip_frac

    def evaluate_batch_step(self, groups: List[RolloutGroup]) -> GRPOMetrics:
        """Processes a full batch of groups, calculating advantages and metrics."""
        all_advantages: List[AdvantageEstimate] = []
        all_trajectories: List[Trajectory] = []
        easy_failures = 0
        hard_wins = 0
        trivial_tasks = 0

        for group in groups:
            advs = self.compute_group_advantages(group)
            all_advantages.extend(advs)
            all_trajectories.extend(group.trajectories)

            if group.group_size > 0 and group.reward_std < self.eps:
                trivial_tasks += 1

            for a in advs:
                if a.is_easy_failure:
                    easy_failures += 1
                if a.is_hard_win:
                    hard_wins += 1

        loss, kl, clip_frac = self.compute_surrogate_loss(all_trajectories, all_advantages)

        adv_vals = [a.advantage for a in all_advantages]
        mean_adv = float(np.mean(adv_vals)) if adv_vals else 0.0
        std_adv = float(np.std(adv_vals)) if adv_vals else 0.0

        return GRPOMetrics(
            group_count=len(groups),
            total_rollouts=len(all_trajectories),
            mean_advantage=mean_adv,
            advantage_std=std_adv,
            clipped_ratio=clip_frac,
            mean_surrogate_loss=loss,
            mean_kl_divergence=kl,
            easy_failures_penalized=easy_failures,
            hard_wins_rewarded=hard_wins,
            trivial_tasks_zeroed=trivial_tasks,
        )
