"""Unit tests for plateau detection and transition gate state machine."""

from ceilbreak.core.transition_gate import PlateauDetector, TransitionGate
from ceilbreak.core.types import (
    TelemetrySnapshot,
    TransitionRegime,
)


def _make_dummy_snapshots(pass_rates, entropies, failure_ratios):
    snaps = []
    for idx, (p, e, f) in enumerate(zip(pass_rates, entropies, failure_ratios)):
        s = TelemetrySnapshot(
            step=idx + 1,
            regime=TransitionRegime.REST_EM_WARMUP,
            pass_at_1=p,
            pass_at_4=p + 0.1,
            pass_at_8=p + 0.2,
            token_entropy=e,
            failure_signal_ratio=f,
            difficulty_skew_index=1.0,
            off_policy_kl=0.01,
            ref_policy_kl=0.02,
        )
        snaps.append(s)
    return snaps


def test_plateau_detector_negative_on_rapid_growth():
    detector = PlateauDetector(window_size=4, slope_threshold=0.02)
    history = _make_dummy_snapshots(
        pass_rates=[0.20, 0.35, 0.50, 0.65],
        entropies=[0.70, 0.68, 0.65, 0.62],
        failure_ratios=[0.2, 0.3, 0.4, 0.5],
    )
    is_plateau, conf, reason = detector.evaluate_plateau(history)
    assert not is_plateau
    assert conf < 0.5


def test_plateau_detector_positive_on_saturation():
    detector = PlateauDetector(window_size=4, slope_threshold=0.02, entropy_floor=0.40)
    # Stalled pass rate + collapsed entropy + high failure signal
    history = _make_dummy_snapshots(
        pass_rates=[0.60, 0.602, 0.601, 0.600],
        entropies=[0.35, 0.30, 0.28, 0.25],
        failure_ratios=[1.2, 1.4, 1.5, 1.6],
    )
    is_plateau, conf, reason = detector.evaluate_plateau(history)
    assert is_plateau
    assert conf >= 0.75
    assert "plateaued" in reason.lower()


def test_transition_gate_state_machine_flow():
    gate = TransitionGate(
        detector=PlateauDetector(window_size=3, slope_threshold=0.02, confidence_threshold=0.7)
    )
    assert gate.state.current_regime == TransitionRegime.REST_EM_WARMUP

    # Growth phase
    h1 = _make_dummy_snapshots([0.2, 0.4, 0.6], [0.8, 0.7, 0.6], [0.2, 0.3, 0.4])
    s1 = gate.check_and_update(step=3, history=h1)
    assert s1.current_regime == TransitionRegime.REST_EM_WARMUP

    # Plateau phase
    h2 = _make_dummy_snapshots([0.60, 0.60, 0.60], [0.25, 0.24, 0.23], [1.5, 1.5, 1.5])
    s2 = gate.check_and_update(step=4, history=h2)
    assert s2.current_regime == TransitionRegime.PLATEAU_DETECTED
    assert s2.plateau_detected

    # Confirmed plateau next step -> GRPO Online
    s3 = gate.check_and_update(step=5, history=h2)
    assert s3.current_regime == TransitionRegime.GRPO_ONLINE
    assert s3.transition_step == 5


def test_transition_gate_force_override():
    gate = TransitionGate()
    state = gate.check_and_update(step=2, history=[], force_transition=True)
    assert state.current_regime == TransitionRegime.GRPO_ONLINE
    assert state.transition_step == 2
