"""Statistical plateau detector and automated regime switcher."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ceilbreak.core.types import (
    TelemetrySnapshot,
    TransitionRegime,
    TransitionState,
)


class PlateauDetector:
    """Detects when rejection sampling pass rate hits a ceiling."""

    def __init__(
        self,
        window_size: int = 4,
        slope_threshold: float = 0.015,
        entropy_floor: float = 0.30,
        failure_ratio_threshold: float = 1.0,
        confidence_threshold: float = 0.75,
    ):
        self.window_size = window_size
        self.slope_threshold = slope_threshold
        self.entropy_floor = entropy_floor
        self.failure_ratio_threshold = failure_ratio_threshold
        self.confidence_threshold = confidence_threshold

    def evaluate_plateau(
        self, history: List[TelemetrySnapshot]
    ) -> Tuple[bool, float, str]:
        """Evaluates whether the pass rate has plateaued and exploration has stalled.

        Returns (is_plateau, confidence, reasoning).
        """
        if len(history) < self.window_size:
            return False, 0.0, f"Insufficient history (need {self.window_size} steps, got {len(history)})."

        recent = history[-self.window_size:]
        pass1_vals = [s.pass_at_1 for s in recent]
        entropies = [s.token_entropy for s in recent]
        failure_ratios = [s.failure_signal_ratio for s in recent]

        # 1. Slope of pass@1 via linear regression
        x = np.arange(len(pass1_vals), dtype=np.float64)
        y = np.array(pass1_vals, dtype=np.float64)
        if np.std(x) > 0:
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0

        # Slope flatness score (1.0 if slope <= 0, linearly scaling up to threshold)
        if slope <= 0.0:
            slope_score = 1.0
        elif slope < self.slope_threshold:
            slope_score = 1.0 - (slope / self.slope_threshold)
        else:
            slope_score = 0.0

        # 2. Entropy collapse score
        latest_entropy = entropies[-1]
        entropy_score = 1.0 if latest_entropy < self.entropy_floor else max(
            0.0, 1.0 - (latest_entropy - self.entropy_floor) / 0.5
        )

        # 3. Failure signal ratio score
        latest_ratio = failure_ratios[-1]
        ratio_score = min(1.0, latest_ratio / self.failure_ratio_threshold)

        # Composite plateau confidence
        confidence = float(0.50 * slope_score + 0.25 * entropy_score + 0.25 * ratio_score)
        is_plateau = confidence >= self.confidence_threshold

        reasons: List[str] = []
        if slope_score >= 0.7:
            reasons.append(f"Pass rate plateaued (slope={slope:.4f} <= {self.slope_threshold})")
        if latest_entropy < self.entropy_floor:
            reasons.append(f"Token exploration entropy collapsed ({latest_entropy:.3f} < {self.entropy_floor})")
        if latest_ratio >= self.failure_ratio_threshold:
            reasons.append(f"Failure signals dominate ({latest_ratio:.2f} >= {self.failure_ratio_threshold})")

        explanation = "; ".join(reasons) if reasons else "Pass rate continuing healthy growth."

        return is_plateau, confidence, explanation


class TransitionGate:
    """Finite State Machine orchestrating ReST-EM to GRPO transitions."""

    def __init__(self, detector: Optional[PlateauDetector] = None):
        self.detector = detector or PlateauDetector()
        self.state = TransitionState()

    def check_and_update(
        self,
        step: int,
        history: List[TelemetrySnapshot],
        force_transition: bool = False,
    ) -> TransitionState:
        """Evaluates latest metrics and updates transition state."""
        # If already online, maintain state
        if self.state.current_regime == TransitionRegime.GRPO_ONLINE:
            self.state.recommended_action = "Maintain GRPO online training with group advantage."
            return self.state

        if force_transition:
            self.state.current_regime = TransitionRegime.GRPO_ONLINE
            self.state.plateau_detected = True
            self.state.plateau_confidence = 1.0
            self.state.trigger_reason = "Manual override transition trigger."
            self.state.transition_step = step
            self.state.recommended_action = "Switch optimizer to GRPO; initialize reference policy."
            return self.state

        is_plateau, confidence, explanation = self.detector.evaluate_plateau(history)
        self.state.plateau_confidence = confidence

        if is_plateau:
            if self.state.current_regime == TransitionRegime.REST_EM_WARMUP:
                self.state.current_regime = TransitionRegime.PLATEAU_DETECTED
                self.state.plateau_detected = True
                self.state.trigger_reason = explanation
                self.state.recommended_action = "Plateau confirmed. Ready to handoff to GRPO."
            elif self.state.current_regime == TransitionRegime.PLATEAU_DETECTED:
                # Confirmed across multiple checks -> switch to GRPO
                self.state.current_regime = TransitionRegime.GRPO_ONLINE
                self.state.transition_step = step
                self.state.recommended_action = (
                    "Switch to GRPO. Enable group advantage baseline and negative feedback penalties."
                )
        else:
            if self.state.current_regime == TransitionRegime.REST_EM_WARMUP:
                self.state.recommended_action = "Continue ReST-EM warmup filtering."

        return self.state
