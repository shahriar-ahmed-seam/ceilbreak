"""Canonical data models, schemas, and typing contracts for CeilBreak."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DifficultyTier(str, Enum):
    """Categorical difficulty tier for reasoning benchmarks."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    FRONTIER = "frontier"


class DomainType(str, Enum):
    """Reasoning problem domain."""
    MATH = "math"
    CODE = "code"
    LOGIC = "logic"
    SYMBOLIC = "symbolic"


class TransitionRegime(str, Enum):
    """Active training regime."""
    REST_EM_WARMUP = "rest_em_warmup"
    PLATEAU_DETECTED = "plateau_detected"
    GRPO_ONLINE = "grpo_online"


class RewardSignal(BaseModel):
    """Multi-dimensional verification feedback for a single trajectory."""
    binary_reward: int = Field(..., description="Binary correctness: 1 for pass, 0 for fail.")
    fractional_reward: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous reward reflecting partial test passes or step verification."
    )
    unit_tests_passed: int = Field(default=0, ge=0)
    total_unit_tests: int = Field(default=0, ge=0)
    syntax_valid: bool = Field(default=True)
    reasoning_length: int = Field(default=0, ge=0, description="Token count in chain-of-thought.")
    verifier_feedback: str = Field(default="", description="Detailed error trace or verification string.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """Single rollout candidate generated from a policy."""
    trajectory_id: str = Field(..., description="Unique trajectory identifier.")
    prompt_id: str = Field(..., description="Corresponding prompt ID.")
    prompt_text: str = Field(..., description="Input prompt text.")
    completion_text: str = Field(..., description="Generated chain-of-thought and answer.")
    tokens: List[str] = Field(default_factory=list, description="Tokenized sequence.")
    token_logprobs: List[float] = Field(
        default_factory=list,
        description="Log-probabilities under generation policy."
    )
    ref_logprobs: List[float] = Field(
        default_factory=list,
        description="Log-probabilities under reference frozen policy."
    )
    reward_signal: RewardSignal = Field(..., description="Verification reward breakdown.")
    policy_iteration: int = Field(default=0, ge=0, description="Training iteration when rolled out.")
    generation_entropy: float = Field(default=0.0, ge=0.0, description="Per-token Shannon entropy.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.reward_signal.binary_reward == 1

    @property
    def primary_reward(self) -> float:
        """Returns fractional reward if present, otherwise binary reward."""
        return self.reward_signal.fractional_reward


class PromptItem(BaseModel):
    """Evaluation or training prompt specification."""
    prompt_id: str = Field(..., description="Unique prompt identifier.")
    domain: DomainType = Field(default=DomainType.MATH)
    difficulty: DifficultyTier = Field(default=DifficultyTier.MEDIUM)
    prompt_text: str = Field(..., description="Problem description.")
    ground_truth: str = Field(..., description="Canonical expected answer.")
    test_cases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Unit test inputs/expected outputs for verifiable code/math tasks."
    )
    expected_difficulty_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Prior expectation of pass rate."
    )


class RolloutGroup(BaseModel):
    """Group of G rollouts sampled for prompt x, used for GRPO group baseline."""
    group_id: str = Field(..., description="Unique group identifier.")
    prompt: PromptItem = Field(..., description="Associated prompt.")
    trajectories: List[Trajectory] = Field(default_factory=list)
    group_size: int = Field(default=0, ge=0)
    empirical_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_mean: float = Field(default=0.0)
    reward_std: float = Field(default=0.0)

    def compute_group_statistics(self) -> None:
        """Computes empirical pass rate, mean reward, and standard deviation."""
        self.group_size = len(self.trajectories)
        if self.group_size == 0:
            self.empirical_pass_rate = 0.0
            self.reward_mean = 0.0
            self.reward_std = 0.0
            return

        rewards = [t.primary_reward for t in self.trajectories]
        passes = sum(1 for t in self.trajectories if t.is_success)
        self.empirical_pass_rate = passes / self.group_size
        self.reward_mean = sum(rewards) / self.group_size
        variance = sum((r - self.reward_mean) ** 2 for r in rewards) / self.group_size
        self.reward_std = variance ** 0.5


class AdvantageEstimate(BaseModel):
    """Normalized advantage score for a trajectory within its group."""
    trajectory_id: str
    prompt_id: str
    raw_reward: float
    group_mean: float
    group_std: float
    advantage: float = Field(
        ...,
        description="Normalized relative surprise: (r - mean) / (std + eps)."
    )
    is_easy_failure: bool = Field(
        default=False,
        description="True if trajectory failed on an easy problem (mean reward >= 0.75)."
    )
    is_hard_win: bool = Field(
        default=False,
        description="True if trajectory succeeded on a hard problem (mean reward <= 0.25)."
    )


class TelemetrySnapshot(BaseModel):
    """Comprehensive telemetry recorded at a training step."""
    step: int = Field(..., ge=0)
    regime: TransitionRegime
    pass_at_1: float = Field(..., ge=0.0, le=1.0)
    pass_at_4: float = Field(..., ge=0.0, le=1.0)
    pass_at_8: float = Field(..., ge=0.0, le=1.0)
    token_entropy: float = Field(..., ge=0.0, description="Mean Shannon entropy of generated tokens.")
    failure_signal_ratio: float = Field(
        ...,
        ge=0.0,
        description="Gradient informativeness of negative rollouts vs positive saturation."
    )
    difficulty_skew_index: float = Field(
        ...,
        description="Gini coefficient of gradient weight allocated to easy vs hard tasks."
    )
    off_policy_kl: float = Field(..., ge=0.0, description="D_KL(pi_current || pi_buffer).")
    ref_policy_kl: float = Field(..., ge=0.0, description="D_KL(pi_current || pi_ref).")
    mean_advantage_variance: float = Field(default=0.0, ge=0.0)
    filtered_buffer_size: int = Field(default=0, ge=0)
    stale_data_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    plateau_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TransitionState(BaseModel):
    """State machine tracking regime handoff."""
    current_regime: TransitionRegime = Field(default=TransitionRegime.REST_EM_WARMUP)
    plateau_detected: bool = Field(default=False)
    plateau_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    trigger_reason: str = Field(default="")
    transition_step: Optional[int] = Field(default=None)
    recommended_action: str = Field(default="Continue ReST-EM warmup")
