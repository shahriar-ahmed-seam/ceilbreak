"""CeilBreak: Adaptive Rejection-Sampling-to-GRPO Transition Engine for LLM Post-Training.

Designed for detecting ReST-EM plateau boundaries and orchestrating Group Relative
Policy Optimization (GRPO) transitions in reasoning model alignment pipelines.
"""

__version__ = "0.1.0"
__author__ = "Shahriar Ahmed Seam"

from ceilbreak.core.types import (
    AdvantageEstimate,
    PromptItem,
    RewardSignal,
    RolloutGroup,
    TelemetrySnapshot,
    Trajectory,
    TransitionRegime,
    TransitionState,
)

__all__ = [
    "AdvantageEstimate",
    "PromptItem",
    "RewardSignal",
    "RolloutGroup",
    "TelemetrySnapshot",
    "Trajectory",
    "TransitionRegime",
    "TransitionState",
]
