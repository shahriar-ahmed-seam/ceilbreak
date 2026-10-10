"""Unit tests for dataset adapters and export pipelines."""

import json

from ceilbreak.adapters.dataset import DatasetAdapter
from ceilbreak.adapters.export import ExportAdapter
from ceilbreak.core.types import (
    DifficultyTier,
    DomainType,
    PromptItem,
    RewardSignal,
    RolloutGroup,
    Trajectory,
)


def test_synthetic_suite_creation():
    suite = DatasetAdapter.create_synthetic_suite(count=12)
    assert len(suite) == 12
    assert suite[0].difficulty in [tier for tier in DifficultyTier]


def test_load_gsm8k_jsonl(tmp_path):
    sample_file = tmp_path / "gsm8k.jsonl"
    sample_file.write_text(
        json.dumps({"question": "Janet has 16 eggs.", "answer": "16"}) + "\n"
        + json.dumps({"question": "A baker bakes 20 pies.", "answer": "20"}) + "\n"
    )
    items = DatasetAdapter.load_gsm8k_jsonl(sample_file)
    assert len(items) == 2
    assert items[0].domain == DomainType.MATH
    assert "eggs" in items[0].prompt_text


def test_export_sft_sharegpt(tmp_path):
    trajs = [
        Trajectory(
            trajectory_id="t1",
            prompt_id="p1",
            prompt_text="Solve 1+1",
            completion_text="2",
            reward_signal=RewardSignal(binary_reward=1, fractional_reward=1.0),
        ),
        Trajectory(
            trajectory_id="t2",
            prompt_id="p1",
            prompt_text="Solve 1+1",
            completion_text="3",
            reward_signal=RewardSignal(binary_reward=0, fractional_reward=0.0),
        ),
    ]
    out_file = tmp_path / "sharegpt.jsonl"
    count = ExportAdapter.export_sft_sharegpt(trajs, out_file, filter_positives_only=True)

    # Only 1 positive is exported
    assert count == 1
    lines = out_file.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["conversations"][0]["value"] == "Solve 1+1"
    assert data["conversations"][1]["value"] == "2"


def test_export_grpo_rl_buffer(tmp_path):
    prompt = PromptItem(
        prompt_id="p1",
        domain=DomainType.MATH,
        difficulty=DifficultyTier.MEDIUM,
        prompt_text="Solve x",
        ground_truth="5",
    )
    t1 = Trajectory(
        trajectory_id="t1",
        prompt_id="p1",
        prompt_text="Solve x",
        completion_text="5",
        reward_signal=RewardSignal(binary_reward=1, fractional_reward=1.0),
    )
    group = RolloutGroup(group_id="g1", prompt=prompt, trajectories=[t1])
    group.compute_group_statistics()

    out_file = tmp_path / "rl_buffer.jsonl"
    count = ExportAdapter.export_grpo_rl_buffer([group], out_file)
    assert count == 1

    lines = out_file.read_text().strip().splitlines()
    data = json.loads(lines[0])
    assert data["group_id"] == "g1"
    assert len(data["rollouts"]) == 1
