"""Reasoning telemetry suite: pass@k, entropy collapse, and failure signal ratio."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ceilbreak.core.types import (
    RolloutGroup,
    TelemetrySnapshot,
    Trajectory,
    TransitionRegime,
)


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """Calculates unbiased pass@k using standard hypergeometric estimator.

    Formula: 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)
    Args:
        n: total rollouts sampled for the prompt.
        c: number of correct rollouts.
        k: top-k evaluation budget.
    """
    if n <= 0 or k <= 0 or k > n:
        return 0.0
    if c >= n:
        return 1.0
    if n - c < k:
        return 1.0

    # Log-space computation for numerical stability
    try:
        log_prob_fail = 0.0
        for i in range(k):
            log_prob_fail += math.log(n - c - i) - math.log(n - i)
        return float(1.0 - math.exp(log_prob_fail))
    except (ValueError, OverflowError):
        # Fallback to direct calculation
        prod = 1.0
        for i in range(k):
            prod *= (n - c - i) / (n - i)
        return float(1.0 - prod)


def estimate_token_entropy(trajectories: List[Trajectory]) -> float:
    """Estimates average token-level Shannon entropy across trajectory rollouts."""
    if not trajectories:
        return 0.0

    entropies: List[float] = []
    for t in trajectories:
        if t.token_logprobs:
            # For each token logprob p_t = exp(lp_t), local binary entropy approx:
            # -p log p - (1-p) log (1-p)
            lps = np.array(t.token_logprobs, dtype=np.float64)
            # Shannon entropy of probability distribution approx from log-likelihood
            probs = np.exp(np.clip(lps, -15.0, 0.0))
            ent = -np.mean(probs * np.log(probs + 1e-12) + (1.0 - probs) * np.log(1.0 - probs + 1e-12))
            entropies.append(float(ent))
        else:
            # Estimate from token vocabulary variety
            words = t.completion_text.split()
            if words:
                _, counts = np.unique(words, return_counts=True)
                freqs = counts / len(words)
                ent = -float(np.sum(freqs * np.log2(freqs + 1e-12)))
                entropies.append(ent)
            else:
                entropies.append(0.0)

    if not entropies:
        return 0.0
    return float(max(0.0, float(np.mean(entropies))))


def calculate_failure_signal_ratio(groups: List[RolloutGroup]) -> float:
    """Computes the ratio of informative failure signals to redundant positive signals.

    When pass rate plateaus, easy problems repeat 100% positive tokens (zero new info),
    while hard problems produce diverse near-miss failures containing high gradient
    correction signal.
    """
    if not groups:
        return 0.0

    informative_failures = 0
    redundant_successes = 0

    for g in groups:
        g.compute_group_statistics()
        for t in g.trajectories:
            if not t.is_success:
                # Failure is informative if it has partial credit or rich feedback
                if t.reward_signal.fractional_reward > 0.1 or t.reward_signal.verifier_feedback:
                    informative_failures += 1
            else:
                # Success on trivial/mastered prompt is redundant
                if g.reward_mean >= 0.8:
                    redundant_successes += 1

    denominator = max(1, redundant_successes)
    return float(informative_failures / denominator)


class TelemetryEngine:
    """Aggregates and tracks reasoning model alignment metrics across steps."""

    def __init__(self):
        self.history: List[TelemetrySnapshot] = []

    def record_step(
        self,
        step: int,
        regime: TransitionRegime,
        groups: List[RolloutGroup],
        stale_data_ratio: float = 0.0,
        buffer_size: int = 0,
        difficulty_skew_index: float = 1.0,
        off_policy_kl: float = 0.0,
        ref_policy_kl: float = 0.0,
        plateau_confidence: float = 0.0,
    ) -> TelemetrySnapshot:
        """Computes pass@k, entropy, and signals from a step's groups."""
        all_trajectories: List[Trajectory] = []
        p1_list: List[float] = []
        p4_list: List[float] = []
        p8_list: List[float] = []

        for g in groups:
            n = len(g.trajectories)
            c = sum(1 for t in g.trajectories if t.is_success)
            all_trajectories.extend(g.trajectories)

            if n >= 1:
                p1_list.append(calculate_pass_at_k(n, c, 1))
            if n >= 4:
                p4_list.append(calculate_pass_at_k(n, c, 4))
            elif n >= 1:
                p4_list.append(calculate_pass_at_k(n, c, min(n, 4)))

            if n >= 8:
                p8_list.append(calculate_pass_at_k(n, c, 8))
            elif n >= 1:
                p8_list.append(calculate_pass_at_k(n, c, min(n, 8)))

        mean_p1 = float(np.mean(p1_list)) if p1_list else 0.0
        mean_p4 = float(np.mean(p4_list)) if p4_list else mean_p1
        mean_p8 = float(np.mean(p8_list)) if p8_list else mean_p4

        entropy = estimate_token_entropy(all_trajectories)
        failure_signal = calculate_failure_signal_ratio(groups)

        # Advantage variance across groups
        var_list = [g.reward_std ** 2 for g in groups if g.group_size > 0]
        mean_var = float(np.mean(var_list)) if var_list else 0.0

        snapshot = TelemetrySnapshot(
            step=step,
            regime=regime,
            pass_at_1=mean_p1,
            pass_at_4=mean_p4,
            pass_at_8=mean_p8,
            token_entropy=entropy,
            failure_signal_ratio=failure_signal,
            difficulty_skew_index=difficulty_skew_index,
            off_policy_kl=off_policy_kl,
            ref_policy_kl=ref_policy_kl,
            mean_advantage_variance=mean_var,
            filtered_buffer_size=buffer_size,
            stale_data_ratio=stale_data_ratio,
            plateau_confidence=plateau_confidence,
        )

        self.history.append(snapshot)
        return snapshot

    def get_latest(self) -> Optional[TelemetrySnapshot]:
        return self.history[-1] if self.history else None
