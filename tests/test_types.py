"""Unit tests for core schemas and data protocols."""

from ceilbreak.core.types import (
    DifficultyTier,
    DomainType,
    PromptItem,
    RewardSignal,
    RolloutGroup,
    Trajectory,
    TransitionRegime,
    TransitionState,
)


def test_reward_signal_validation():
    signal = RewardSignal(
        binary_reward=1,
        fractional_reward=0.85,
        unit_tests_passed=9,
        total_unit_tests=10,
        syntax_valid=True,
        reasoning_length=142,
    )
    assert signal.binary_reward == 1
    assert signal.fractional_reward == 0.85
    assert signal.unit_tests_passed == 9


def test_trajectory_properties():
    signal = RewardSignal(
        binary_reward=0,
        fractional_reward=0.9,
        unit_tests_passed=9,
        total_unit_tests=10,
        verifier_feedback="AssertionError on test 10",
    )
    traj = Trajectory(
        trajectory_id="traj_1",
        prompt_id="prompt_gsm8k_1",
        prompt_text="Calculate 15 * 4 + 2",
        completion_text="15 * 4 = 60; 60 + 2 = 62",
        token_logprobs=[-0.12, -0.05, -0.08],
        reward_signal=signal,
    )
    assert not traj.is_success
    assert traj.primary_reward == 0.9
    assert traj.reward_signal.verifier_feedback == "AssertionError on test 10"


def test_rollout_group_statistics():
    prompt = PromptItem(
        prompt_id="p1",
        domain=DomainType.MATH,
        difficulty=DifficultyTier.MEDIUM,
        prompt_text="Solve x^2 - 4 = 0",
        ground_truth="x = 2 or x = -2",
    )
    t1 = Trajectory(
        trajectory_id="t1",
        prompt_id="p1",
        prompt_text="Solve x^2 - 4 = 0",
        completion_text="x = 2",
        reward_signal=RewardSignal(binary_reward=1, fractional_reward=1.0),
    )
    t2 = Trajectory(
        trajectory_id="t2",
        prompt_id="p1",
        prompt_text="Solve x^2 - 4 = 0",
        completion_text="x = 0",
        reward_signal=RewardSignal(binary_reward=0, fractional_reward=0.0),
    )
    group = RolloutGroup(group_id="g1", prompt=prompt, trajectories=[t1, t2])
    group.compute_group_statistics()

    assert group.group_size == 2
    assert group.empirical_pass_rate == 0.5
    assert group.reward_mean == 0.5
    assert abs(group.reward_std - 0.5) < 1e-5


def test_transition_state_defaults():
    state = TransitionState()
    assert state.current_regime == TransitionRegime.REST_EM_WARMUP
    assert not state.plateau_detected
    assert state.transition_step is None
