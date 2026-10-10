"""Unit tests for Group Relative Policy Optimization (GRPO) advantage engine."""

from ceilbreak.core.grpo import GRPOEngine
from ceilbreak.core.types import (
    DifficultyTier,
    DomainType,
    PromptItem,
    RewardSignal,
    RolloutGroup,
    Trajectory,
)


def _build_group(rewards, prompt_tier=DifficultyTier.MEDIUM):
    prompt = PromptItem(
        prompt_id="test_p",
        domain=DomainType.MATH,
        difficulty=prompt_tier,
        prompt_text="Solve equation",
        ground_truth="42",
    )
    trajs = []
    for idx, r in enumerate(rewards):
        bin_r = 1 if r >= 1.0 else 0
        sig = RewardSignal(binary_reward=bin_r, fractional_reward=float(r))
        t = Trajectory(
            trajectory_id=f"t_{idx}",
            prompt_id="test_p",
            prompt_text="Solve equation",
            completion_text=f"Solution {idx}",
            token_logprobs=[-0.1, -0.2],
            reward_signal=sig,
        )
        trajs.append(t)
    return RolloutGroup(group_id="g_test", prompt=prompt, trajectories=trajs)


def test_grpo_homogeneous_group_zero_advantage():
    # If all rollouts pass, standard deviation is zero: advantage is zeroed out
    group = _build_group([1.0, 1.0, 1.0, 1.0], DifficultyTier.EASY)
    engine = GRPOEngine()
    advs = engine.compute_group_advantages(group)

    assert len(advs) == 4
    for a in advs:
        assert a.advantage == 0.0
        assert not a.is_easy_failure
        assert not a.is_hard_win


def test_grpo_penalizes_easy_failures():
    # 3 passes, 1 failure on easy task: failure receives negative advantage
    group = _build_group([1.0, 1.0, 1.0, 0.0], DifficultyTier.EASY)
    engine = GRPOEngine()
    advs = engine.compute_group_advantages(group)

    failed_adv = advs[3]
    assert failed_adv.raw_reward == 0.0
    assert failed_adv.advantage < 0.0
    assert failed_adv.is_easy_failure
    # Negative advantage on failure is approx -1.732
    assert failed_adv.advantage < -1.5


def test_grpo_rewards_hard_wins():
    # 3 fails, 1 win on hard task: the rare win receives strong positive advantage
    group = _build_group([0.0, 0.0, 0.0, 1.0], DifficultyTier.HARD)
    engine = GRPOEngine()
    advs = engine.compute_group_advantages(group)

    winning_adv = advs[3]
    assert winning_adv.raw_reward == 1.0
    assert winning_adv.advantage > 0.0
    assert winning_adv.is_hard_win
    assert winning_adv.advantage > 1.5


def test_grpo_partial_credit_advantage():
    # Fractional rewards: 0.0, 0.3, 0.8, 1.0
    group = _build_group([0.0, 0.3, 0.8, 1.0])
    engine = GRPOEngine()
    advs = engine.compute_group_advantages(group)

    # Monotonically increasing advantages corresponding to partial scores
    scores = [a.advantage for a in advs]
    assert scores[0] < scores[1] < scores[2] < scores[3]


def test_grpo_surrogate_loss_and_clipping():
    group = _build_group([0.0, 1.0])
    engine = GRPOEngine(clip_epsilon=0.2)
    advs = engine.compute_group_advantages(group)

    # Simulated large ratio shift
    curr_lps = [[-0.01, -0.01], [-0.01, -0.01]]
    loss, kl, clip_frac = engine.compute_surrogate_loss(group.trajectories, advs, curr_lps)

    assert isinstance(loss, float)
    assert isinstance(kl, float)
    assert clip_frac >= 0.0


def test_grpo_batch_step_metrics():
    g1 = _build_group([1.0, 1.0, 1.0, 0.0])
    g2 = _build_group([0.0, 0.0, 0.0, 1.0])
    engine = GRPOEngine()
    metrics = engine.evaluate_batch_step([g1, g2])

    assert metrics.group_count == 2
    assert metrics.total_rollouts == 8
    assert metrics.easy_failures_penalized >= 1
    assert metrics.hard_wins_rewarded >= 1
