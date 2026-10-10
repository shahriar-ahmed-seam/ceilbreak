"""Export adapters for SFT datasets and RL trajectory buffers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from ceilbreak.core.types import RolloutGroup, Trajectory


class ExportAdapter:
    """Serializes reasoning trajectories into standard industry formats."""

    @staticmethod
    def export_sft_sharegpt(
        trajectories: List[Trajectory],
        output_file: Union[str, Path],
        filter_positives_only: bool = True,
    ) -> int:
        """Exports filtered rollouts into ShareGPT multi-turn JSONL format."""
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(path, "w", encoding="utf-8") as f:
            for t in trajectories:
                if filter_positives_only and not t.is_success:
                    continue
                record = {
                    "id": t.trajectory_id,
                    "conversations": [
                        {"from": "human", "value": t.prompt_text},
                        {"from": "gpt", "value": t.completion_text},
                    ],
                    "metadata": {
                        "reward": t.primary_reward,
                        "iteration": t.policy_iteration,
                    },
                }
                f.write(json.dumps(record) + "\n")
                count += 1

        return count

    @staticmethod
    def export_grpo_rl_buffer(
        groups: List[RolloutGroup],
        output_file: Union[str, Path],
    ) -> int:
        """Exports rollout groups with normalized relative advantages to RL JSONL."""
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(path, "w", encoding="utf-8") as f:
            for g in groups:
                record = {
                    "group_id": g.group_id,
                    "prompt": g.prompt.prompt_text,
                    "difficulty": g.prompt.difficulty.value,
                    "rollouts": [
                        {
                            "trajectory_id": t.trajectory_id,
                            "completion": t.completion_text,
                            "binary_reward": t.reward_signal.binary_reward,
                            "fractional_reward": t.reward_signal.fractional_reward,
                            "is_success": t.is_success,
                        }
                        for t in g.trajectories
                    ],
                    "group_statistics": {
                        "pass_rate": g.empirical_pass_rate,
                        "reward_mean": g.reward_mean,
                        "reward_std": g.reward_std,
                    },
                }
                f.write(json.dumps(record) + "\n")
                count += 1

        return count
