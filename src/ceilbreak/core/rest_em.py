"""ReST-EM (Rejection Sampling Fine-Tuning) Engine.

Implements Expectation (rejection sampling on verifier threshold) and Maximization
(positive-only Maximum Likelihood Estimation) steps, exposing the 4 fundamental
failure modes of filtering:
1. Invisible Failures (r=0 discards gradient signals on error attractors)
2. No Difficulty Sense (trivial wins flood dataset; hard tasks count equally)
3. Stale Off-Policy Data (re-fitting old rollouts collapses entropy and kills exploration)
4. No Partial Credit (binary filter discards near-miss reasoning chains)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from pydantic import BaseModel, Field

from ceilbreak.core.types import (
    DifficultyTier,
    RolloutGroup,
    Trajectory,
)


class FilterResult(BaseModel):
    """Telemetry produced by the Expectation (Rejection Sampling) filter step."""
    total_rollouts: int
    accepted_trajectories: List[Trajectory] = Field(default_factory=list)
    discarded_count: int
    accepted_count: int
    yield_rate: float
    invisible_failure_count: int
    invisible_failure_rate: float
    difficulty_breakdown: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    discarded_failure_reasons: Dict[str, int] = Field(default_factory=dict)


class MLESnapshot(BaseModel):
    """Output metrics from Maximization step."""
    total_training_examples: int
    mean_token_nll: float
    difficulty_skew_ratio: float
    easy_examples_count: int
    hard_examples_count: int
    gini_skew_coefficient: float
    effective_gradient_norm: float


class ReSTFilter:
    """Expectation (E-step) Rejection Sampling Filter."""

    def __init__(self, reward_threshold: float = 1.0, allow_partial: bool = False):
        self.reward_threshold = reward_threshold
        self.allow_partial = allow_partial

    def filter_rollouts(self, groups: List[RolloutGroup]) -> FilterResult:
        """Filters candidates based on verifier threshold, recording failure blindness."""
        accepted: List[Trajectory] = []
        discarded_count = 0
        invisible_failures = 0
        failure_reasons: Dict[str, int] = {}
        difficulty_breakdown: Dict[str, Dict[str, int]] = {
            tier.value: {"accepted": 0, "discarded": 0} for tier in DifficultyTier
        }

        total_rollouts = 0

        for group in groups:
            tier_key = group.prompt.difficulty.value
            for traj in group.trajectories:
                total_rollouts += 1
                score = (
                    traj.reward_signal.fractional_reward
                    if self.allow_partial
                    else float(traj.reward_signal.binary_reward)
                )

                if score >= self.reward_threshold:
                    accepted.append(traj)
                    difficulty_breakdown[tier_key]["accepted"] += 1
                else:
                    discarded_count += 1
                    difficulty_breakdown[tier_key]["discarded"] += 1
                    if traj.reward_signal.binary_reward == 0:
                        invisible_failures += 1
                    reason = traj.reward_signal.verifier_feedback or "verifier_rejection"
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        yield_rate = (len(accepted) / total_rollouts) if total_rollouts > 0 else 0.0
        invisible_failure_rate = (
            (invisible_failures / total_rollouts) if total_rollouts > 0 else 0.0
        )

        return FilterResult(
            total_rollouts=total_rollouts,
            accepted_trajectories=accepted,
            discarded_count=discarded_count,
            accepted_count=len(accepted),
            yield_rate=yield_rate,
            invisible_failure_count=invisible_failures,
            invisible_failure_rate=invisible_failure_rate,
            difficulty_breakdown=difficulty_breakdown,
            discarded_failure_reasons=failure_reasons,
        )


class ReSTMLEMaximizer:
    """Maximization (M-step) positive-only Maximum Likelihood Estimation engine."""

    def __init__(self, learning_rate: float = 1e-4):
        self.learning_rate = learning_rate

    def compute_mle_update(self, accepted_trajectories: List[Trajectory]) -> MLESnapshot:
        """Computes positive-only SFT loss and analyzes difficulty skew."""
        if not accepted_trajectories:
            return MLESnapshot(
                total_training_examples=0,
                mean_token_nll=0.0,
                difficulty_skew_ratio=1.0,
                easy_examples_count=0,
                hard_examples_count=0,
                gini_skew_coefficient=0.0,
                effective_gradient_norm=0.0,
            )

        nll_values: List[float] = []
        easy_count = 0
        hard_count = 0
        prompt_counts: Dict[str, int] = {}

        for traj in accepted_trajectories:
            # -log P(y|x) approximation from stored logprobs
            if traj.token_logprobs:
                token_nll = -float(np.mean(traj.token_logprobs))
            else:
                token_nll = 0.5  # fallback baseline
            nll_values.append(token_nll)

            prompt_counts[traj.prompt_id] = prompt_counts.get(traj.prompt_id, 0) + 1
            meta_difficulty = traj.metadata.get("difficulty", "medium")
            if meta_difficulty in ("easy", DifficultyTier.EASY.value):
                easy_count += 1
            elif meta_difficulty in ("hard", "frontier", DifficultyTier.HARD.value, DifficultyTier.FRONTIER.value):
                hard_count += 1

        # Difficulty skew: ratio of easy updates to hard updates
        denominator = hard_count if hard_count > 0 else 1
        skew_ratio = float(easy_count / denominator)

        # Gini coefficient of prompt representations in dataset
        counts = list(prompt_counts.values())
        gini = self._compute_gini(counts)

        mean_nll = float(np.mean(nll_values)) if nll_values else 0.0
        # Simulated gradient norm proportional to batch size and loss
        grad_norm = float(mean_nll * np.sqrt(len(accepted_trajectories)) * 0.1)

        return MLESnapshot(
            total_training_examples=len(accepted_trajectories),
            mean_token_nll=mean_nll,
            difficulty_skew_ratio=skew_ratio,
            easy_examples_count=easy_count,
            hard_examples_count=hard_count,
            gini_skew_coefficient=gini,
            effective_gradient_norm=grad_norm,
        )

    def _compute_gini(self, array: List[int]) -> float:
        """Computes Gini inequality coefficient across per-prompt sample counts."""
        if not array or len(array) <= 1:
            return 0.0
        arr = np.array(array, dtype=np.float64)
        if np.amin(arr) < 0:
            arr -= np.amin(arr)
        arr += 0.0000001
        arr = np.sort(arr)
        index = np.arange(1, arr.shape[0] + 1)
        n = arr.shape[0]
        return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


class OffPolicyBuffer:
    """Historical rollout buffer tracking stale data and policy divergence."""

    def __init__(self, max_capacity: int = 10000, max_age_iterations: int = 4):
        self.max_capacity = max_capacity
        self.max_age_iterations = max_age_iterations
        self.buffer: List[Trajectory] = []

    def add_trajectories(self, trajectories: List[Trajectory]) -> None:
        """Appends new filtered trajectories, evicting oldest if capacity reached."""
        for t in trajectories:
            self.buffer.append(t)
            if len(self.buffer) > self.max_capacity:
                self.buffer.pop(0)

    def get_stale_data_metrics(self, current_iteration: int) -> Dict[str, float]:
        """Calculates stale data ratio and off-policy age distribution."""
        if not self.buffer:
            return {
                "buffer_size": 0.0,
                "stale_data_ratio": 0.0,
                "mean_policy_age": 0.0,
                "diversity_collapse_score": 0.0,
            }

        ages = [max(0, current_iteration - t.policy_iteration) for t in self.buffer]
        stale_items = sum(1 for age in ages if age >= self.max_age_iterations)
        stale_ratio = stale_items / len(self.buffer)
        mean_age = float(np.mean(ages))

        # Diversity score based on unique completion prefixes
        prefixes = {t.completion_text[:40] for t in self.buffer}
        diversity_score = len(prefixes) / len(self.buffer)
        # Collapse score is 1.0 - diversity
        collapse_score = max(0.0, min(1.0, 1.0 - diversity_score))

        return {
            "buffer_size": float(len(self.buffer)),
            "stale_data_ratio": float(stale_ratio),
            "mean_policy_age": mean_age,
            "diversity_collapse_score": float(collapse_score),
        }

    def sample_batch(self, batch_size: int) -> List[Trajectory]:
        """Uniformly samples a batch from the historical buffer."""
        if not self.buffer:
            return []
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[int(idx)] for idx in indices]
