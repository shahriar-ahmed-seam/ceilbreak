"""Unified reasoning optimization pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ceilbreak.core.grpo import GRPOEngine
from ceilbreak.core.rest_em import (
    OffPolicyBuffer,
    ReSTFilter,
    ReSTMLEMaximizer,
)
from ceilbreak.core.telemetry import TelemetryEngine
from ceilbreak.core.transition_gate import PlateauDetector, TransitionGate
from ceilbreak.core.types import (
    DifficultyTier,
    DomainType,
    PromptItem,
    RewardSignal,
    RolloutGroup,
    TelemetrySnapshot,
    Trajectory,
    TransitionRegime,
)


class ReasoningPipeline:
    """End-to-end training orchestrator demonstrating ReST-EM plateau and GRPO transition."""

    def __init__(
        self,
        prompts: Optional[List[PromptItem]] = None,
        group_size: int = 4,
        plateau_window: int = 3,
        auto_transition: bool = True,
        on_step_callback: Optional[Callable[[TelemetrySnapshot, Dict[str, Any]], None]] = None,
    ):
        self.prompts = prompts or self._generate_default_prompts()
        self.group_size = group_size
        self.auto_transition = auto_transition
        self.on_step_callback = on_step_callback

        # Core engines
        self.rest_filter = ReSTFilter(reward_threshold=1.0)
        self.rest_maximizer = ReSTMLEMaximizer()
        self.off_policy_buffer = OffPolicyBuffer(max_capacity=5000)
        self.grpo_engine = GRPOEngine()
        self.telemetry = TelemetryEngine()
        self.transition_gate = TransitionGate(
            detector=PlateauDetector(window_size=plateau_window, slope_threshold=0.02)
        )

        self.current_step = 0
        self.active_regime = TransitionRegime.REST_EM_WARMUP
        self.policy_latent_capability = 0.35  # internal capability scale

    def _generate_default_prompts(self) -> List[PromptItem]:
        """Creates benchmark reasoning prompts across difficulty tiers."""
        tiers = [
            (DifficultyTier.EASY, "Calculate arithmetic: 45 * 2 - 12", "78", 0.85),
            (DifficultyTier.EASY, "Find the perimeter of a 7x4 rectangle", "22", 0.90),
            (DifficultyTier.MEDIUM, "Solve quadratic roots: x^2 - 5x + 6 = 0", "2, 3", 0.50),
            (DifficultyTier.MEDIUM, "Prime factorization of 144", "2^4 * 3^2", 0.55),
            (DifficultyTier.HARD, "Evaluate sum_{n=1}^inf 1/(n*(n+1))", "1", 0.20),
            (DifficultyTier.HARD, "Find smallest positive integer with 12 divisors", "72", 0.15),
            (DifficultyTier.FRONTIER, "Prove modular congruence for Euler totient theorem", "QED", 0.05),
        ]
        items = []
        for idx, (tier, prob, ans, exp_p) in enumerate(tiers):
            items.append(
                PromptItem(
                    prompt_id=f"prompt_{idx+1}",
                    domain=DomainType.MATH,
                    difficulty=tier,
                    prompt_text=prob,
                    ground_truth=ans,
                    expected_difficulty_rate=exp_p,
                )
            )
        return items

    def _sample_rollout_group(self, prompt: PromptItem, step: int) -> RolloutGroup:
        """Simulates rollout sampling reflecting policy state and active regime."""
        trajectories: List[Trajectory] = []

        # Base pass probability depends on prompt difficulty and current capability
        base_pass_prob = prompt.expected_difficulty_rate * self.policy_latent_capability * 1.8
        base_pass_prob = min(0.98, max(0.02, base_pass_prob))

        for idx in range(self.group_size):
            passed = bool(np.random.rand() < base_pass_prob)
            if passed:
                reward = RewardSignal(
                    binary_reward=1,
                    fractional_reward=1.0,
                    unit_tests_passed=5,
                    total_unit_tests=5,
                    syntax_valid=True,
                    reasoning_length=max(10, int(np.random.normal(120, 20))),
                    verifier_feedback="Verified solution.",
                )
                comp = f"<think>Sound reasoning step for {prompt.prompt_text}</think><answer>{prompt.ground_truth}</answer>"
                lps = list(-np.random.exponential(0.15, size=8))
            else:
                # Failure: can be a near-miss or complete error
                partial = 0.8 if np.random.rand() < 0.4 else 0.0
                feedback = "Near-miss edge case failure" if partial > 0 else "Calculation error"
                reward = RewardSignal(
                    binary_reward=0,
                    fractional_reward=partial,
                    unit_tests_passed=4 if partial > 0 else 1,
                    total_unit_tests=5,
                    syntax_valid=True,
                    reasoning_length=max(10, int(np.random.normal(80, 25))),
                    verifier_feedback=feedback,
                )
                comp = f"<think>Erroneous reasoning</think><answer>Incorrect candidate {idx}</answer>"
                lps = list(-np.random.exponential(0.6, size=8))

            ref_lps = [lp - 0.05 for lp in lps]

            t = Trajectory(
                trajectory_id=f"step_{step}_{prompt.prompt_id}_rollout_{idx}",
                prompt_id=prompt.prompt_id,
                prompt_text=prompt.prompt_text,
                completion_text=comp,
                token_logprobs=lps,
                ref_logprobs=ref_lps,
                reward_signal=reward,
                policy_iteration=step,
                metadata={"difficulty": prompt.difficulty.value},
            )
            trajectories.append(t)

        group = RolloutGroup(
            group_id=f"group_{step}_{prompt.prompt_id}",
            prompt=prompt,
            trajectories=trajectories,
        )
        group.compute_group_statistics()
        return group

    def run_step(self, step: Optional[int] = None) -> TelemetrySnapshot:
        """Executes a single iteration step of the training pipeline."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        step_num = self.current_step

        # 1. Roll out candidates across prompts
        groups = [self._sample_rollout_group(p, step_num) for p in self.prompts]

        step_info: Dict[str, Any] = {}

        # 2. Process according to current regime
        if self.active_regime == TransitionRegime.REST_EM_WARMUP:
            filter_res = self.rest_filter.filter_rollouts(groups)
            self.off_policy_buffer.add_trajectories(filter_res.accepted_trajectories)
            mle_snapshot = self.rest_maximizer.compute_mle_update(filter_res.accepted_trajectories)

            step_info["filter_yield"] = filter_res.yield_rate
            step_info["invisible_failures"] = filter_res.invisible_failure_count
            step_info["difficulty_skew"] = mle_snapshot.difficulty_skew_ratio

            # In ReST-EM, capability grows fast initially then hits a ceiling around step 4-5
            if step_num <= 3:
                self.policy_latent_capability += 0.08
            else:
                # Diminishing returns & plateau
                self.policy_latent_capability += 0.005

        else:
            # GRPO Online Advantage Training
            grpo_metrics = self.grpo_engine.evaluate_batch_step(groups)
            step_info["surrogate_loss"] = grpo_metrics.mean_surrogate_loss
            step_info["mean_advantage"] = grpo_metrics.mean_advantage
            step_info["easy_failures_penalized"] = grpo_metrics.easy_failures_penalized
            step_info["hard_wins_rewarded"] = grpo_metrics.hard_wins_rewarded

            # GRPO breaks through ceiling by exploiting failure feedback & hard wins
            self.policy_latent_capability += 0.07

        # 3. Compute telemetry
        buffer_metrics = self.off_policy_buffer.get_stale_data_metrics(step_num)
        snapshot = self.telemetry.record_step(
            step=step_num,
            regime=self.active_regime,
            groups=groups,
            stale_data_ratio=buffer_metrics["stale_data_ratio"],
            buffer_size=int(buffer_metrics["buffer_size"]),
            difficulty_skew_index=step_info.get("difficulty_skew", 1.0),
            plateau_confidence=self.transition_gate.state.plateau_confidence,
        )

        # 4. Check transition gate
        if self.auto_transition and self.active_regime != TransitionRegime.GRPO_ONLINE:
            trans_state = self.transition_gate.check_and_update(step_num, self.telemetry.history)
            if trans_state.current_regime in (TransitionRegime.PLATEAU_DETECTED, TransitionRegime.GRPO_ONLINE):
                self.active_regime = trans_state.current_regime

        step_info["transition_state"] = self.transition_gate.state.model_dump()

        if self.on_step_callback:
            self.on_step_callback(snapshot, step_info)

        return snapshot

    def run_pipeline(self, total_steps: int = 8) -> List[TelemetrySnapshot]:
        """Executes full multi-step training run."""
        results = []
        for s in range(1, total_steps + 1):
            snap = self.run_step(step=s)
            results.append(snap)
        return results

    def export_telemetry_json(self, output_path: str) -> None:
        """Saves telemetry history to JSON."""
        data = [s.model_dump() for s in self.telemetry.history]
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
