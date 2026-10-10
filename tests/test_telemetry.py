"""Unit tests for telemetry metrics and pass@k hypergeometric estimator."""

from ceilbreak.core.telemetry import (
    TelemetryEngine,
    calculate_failure_signal_ratio,
    calculate_pass_at_k,
    estimate_token_entropy,
)
from ceilbreak.core.types import (
    DifficultyTier,
    DomainType,
    PromptItem,
    RewardSignal,
    RolloutGroup,
    Trajectory,
    TransitionRegime,
)


def test_calculate_pass_at_k_bounds():
    # Full success
    assert calculate_pass_at_k(n=4, c=4, k=1) == 1.0
    assert calculate_pass_at_k(n=4, c=4, k=4) == 1.0

    # Total failure
    assert calculate_pass_at_k(n=4, c=0, k=1) == 0.0
    assert calculate_pass_at_k(n=4, c=0, k=2) == 0.0

    # Half success: n=4, c=2, k=1 -> 0.5
    assert abs(calculate_pass_at_k(n=4, c=2, k=1) - 0.5) < 1e-4

    # n=4, c=2, k=2: 1 - (2*1)/(4*3) = 1 - 2/12 = 10/12 = 0.8333
    assert abs(calculate_pass_at_k(n=4, c=2, k=2) - (5 / 6)) < 1e-4


def test_estimate_token_entropy():
    trajs = [
        Trajectory(
            trajectory_id="t1",
            prompt_id="p1",
            prompt_text="Solve x",
            completion_text="The solution is x = 42 because 42 is the answer",
            token_logprobs=[-0.1, -0.2, -0.15],
            reward_signal=RewardSignal(binary_reward=1, fractional_reward=1.0),
        )
    ]
    entropy = estimate_token_entropy(trajs)
    assert entropy > 0.0


def test_failure_signal_ratio_calculation():
    prompt = PromptItem(
        prompt_id="p1",
        domain=DomainType.MATH,
        difficulty=DifficultyTier.EASY,
        prompt_text="2+2",
        ground_truth="4",
    )
    # Group with 1 redundant success on easy task and 1 informative failure
    trajs = [
        Trajectory(
            trajectory_id="t1",
            prompt_id="p1",
            prompt_text="2+2",
            completion_text="4",
            reward_signal=RewardSignal(binary_reward=1, fractional_reward=1.0),
        ),
        Trajectory(
            trajectory_id="t2",
            prompt_id="p1",
            prompt_text="2+2",
            completion_text="5",
            reward_signal=RewardSignal(
                binary_reward=0,
                fractional_reward=0.5,
                verifier_feedback="Off-by-one near miss",
            ),
        ),
    ]
    group = RolloutGroup(group_id="g1", prompt=prompt, trajectories=trajs)
    ratio = calculate_failure_signal_ratio([group])
    assert ratio >= 0.5


def test_telemetry_engine_recording():
    engine = TelemetryEngine()
    prompt = PromptItem(
        prompt_id="p1",
        domain=DomainType.MATH,
        difficulty=DifficultyTier.MEDIUM,
        prompt_text="test",
        ground_truth="1",
    )
    trajs = [
        Trajectory(
            trajectory_id="t1",
            prompt_id="p1",
            prompt_text="test",
            completion_text="1",
            reward_signal=RewardSignal(binary_reward=1, fractional_reward=1.0),
        )
    ]
    group = RolloutGroup(group_id="g1", prompt=prompt, trajectories=trajs)

    snap = engine.record_step(
        step=1,
        regime=TransitionRegime.REST_EM_WARMUP,
        groups=[group],
        stale_data_ratio=0.0,
        buffer_size=1,
    )
    assert snap.step == 1
    assert snap.pass_at_1 == 1.0
    assert engine.get_latest() == snap
