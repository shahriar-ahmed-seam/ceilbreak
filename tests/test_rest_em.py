"""Unit tests for ReST-EM engine and its 4 failure modes."""

from ceilbreak.core.rest_em import (
    OffPolicyBuffer,
    ReSTFilter,
    ReSTMLEMaximizer,
)
from ceilbreak.core.types import (
    DifficultyTier,
    DomainType,
    PromptItem,
    RewardSignal,
    RolloutGroup,
    Trajectory,
)


def _make_sample_groups():
    # Easy prompt: high pass rate (4 passes, 0 fails)
    p_easy = PromptItem(
        prompt_id="easy_1",
        domain=DomainType.MATH,
        difficulty=DifficultyTier.EASY,
        prompt_text="What is 2 + 2?",
        ground_truth="4",
    )
    trajs_easy = [
        Trajectory(
            trajectory_id=f"easy_t_{i}",
            prompt_id="easy_1",
            prompt_text="What is 2 + 2?",
            completion_text="The answer is 4.",
            token_logprobs=[-0.01, -0.02],
            reward_signal=RewardSignal(binary_reward=1, fractional_reward=1.0),
            metadata={"difficulty": "easy"},
        )
        for i in range(4)
    ]
    g_easy = RolloutGroup(group_id="g_easy", prompt=p_easy, trajectories=trajs_easy)

    # Hard prompt: low pass rate (1 pass, 3 fails)
    p_hard = PromptItem(
        prompt_id="hard_1",
        domain=DomainType.MATH,
        difficulty=DifficultyTier.HARD,
        prompt_text="Prove the Riemann hypothesis for s=1/2.",
        ground_truth="QED",
    )
    trajs_hard = [
        Trajectory(
            trajectory_id="hard_t_0",
            prompt_id="hard_1",
            prompt_text="Hard prompt",
            completion_text="Valid proof step",
            token_logprobs=[-0.8, -0.9],
            reward_signal=RewardSignal(binary_reward=1, fractional_reward=1.0),
            metadata={"difficulty": "hard"},
        ),
        Trajectory(
            trajectory_id="hard_t_1",
            prompt_id="hard_1",
            prompt_text="Hard prompt",
            completion_text="Erroneous calculation step",
            token_logprobs=[-0.4, -0.5],
            reward_signal=RewardSignal(
                binary_reward=0,
                fractional_reward=0.0,
                verifier_feedback="Calculation error on step 3",
            ),
            metadata={"difficulty": "hard"},
        ),
        Trajectory(
            trajectory_id="hard_t_2",
            prompt_id="hard_1",
            prompt_text="Hard prompt",
            completion_text="Near-miss proof with 9/10 steps correct",
            token_logprobs=[-0.2, -0.3],
            reward_signal=RewardSignal(
                binary_reward=0,
                fractional_reward=0.9,
                verifier_feedback="Boundary condition missed",
            ),
            metadata={"difficulty": "hard"},
        ),
        Trajectory(
            trajectory_id="hard_t_3",
            prompt_id="hard_1",
            prompt_text="Hard prompt",
            completion_text="Complete gibberish hallucination",
            token_logprobs=[-1.2, -1.3],
            reward_signal=RewardSignal(
                binary_reward=0,
                fractional_reward=0.0,
                verifier_feedback="Syntax failure",
            ),
            metadata={"difficulty": "hard"},
        ),
    ]
    g_hard = RolloutGroup(group_id="g_hard", prompt=p_hard, trajectories=trajs_hard)

    return [g_easy, g_hard]


def test_rest_filter_expectation_step():
    groups = _make_sample_groups()
    filter_engine = ReSTFilter(reward_threshold=1.0, allow_partial=False)
    result = filter_engine.filter_rollouts(groups)

    assert result.total_rollouts == 8
    # 4 easy passes + 1 hard pass = 5 accepted
    assert result.accepted_count == 5
    assert result.discarded_count == 3
    assert result.yield_rate == 5 / 8
    # Failure Mode 1: Invisible Failures (r=0 gives zero gradient)
    assert result.invisible_failure_count == 3
    assert result.invisible_failure_rate == 3 / 8
    assert "Calculation error on step 3" in result.discarded_failure_reasons


def test_failure_mode_no_partial_credit():
    groups = _make_sample_groups()
    # Binary filter discards 9/10 near miss
    binary_filter = ReSTFilter(reward_threshold=1.0, allow_partial=False)
    bin_res = binary_filter.filter_rollouts(groups)
    assert bin_res.accepted_count == 5

    # Partial credit filter can retain near misses if threshold lowered
    relaxed_filter = ReSTFilter(reward_threshold=0.8, allow_partial=True)
    rel_res = relaxed_filter.filter_rollouts(groups)
    # 4 easy + 1 hard win + 1 hard near-miss (0.9) = 6 accepted
    assert rel_res.accepted_count == 6


def test_failure_mode_no_difficulty_sense():
    groups = _make_sample_groups()
    filter_engine = ReSTFilter(reward_threshold=1.0)
    result = filter_engine.filter_rollouts(groups)

    maximizer = ReSTMLEMaximizer()
    snapshot = maximizer.compute_mle_update(result.accepted_trajectories)

    # Failure Mode 2: Easy wins flood dataset (4 easy vs 1 hard)
    assert snapshot.easy_examples_count == 4
    assert snapshot.hard_examples_count == 1
    assert snapshot.difficulty_skew_ratio == 4.0
    assert snapshot.gini_skew_coefficient > 0.0


def test_failure_mode_stale_off_policy_buffer():
    buffer = OffPolicyBuffer(max_capacity=100, max_age_iterations=3)
    trajs = [
        Trajectory(
            trajectory_id=f"t_{i}",
            prompt_id="p1",
            prompt_text="prompt",
            completion_text="repeated answer" if i > 2 else f"unique answer {i}",
            reward_signal=RewardSignal(binary_reward=1, fractional_reward=1.0),
            policy_iteration=0,
        )
        for i in range(10)
    ]
    buffer.add_trajectories(trajs)

    # When current iteration is 4, all items from iteration 0 are stale (age=4 >= 3)
    metrics = buffer.get_stale_data_metrics(current_iteration=4)
    assert metrics["stale_data_ratio"] == 1.0
    assert metrics["mean_policy_age"] == 4.0
    # High repetition causes diversity collapse score to be positive
    assert metrics["diversity_collapse_score"] > 0.0
