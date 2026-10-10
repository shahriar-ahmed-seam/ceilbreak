"""Dataset loaders and adapters for GSM8K, MATH, and HumanEval formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from ceilbreak.core.types import DifficultyTier, DomainType, PromptItem


class DatasetAdapter:
    """Parses and normalizes reasoning datasets into PromptItem instances."""

    @staticmethod
    def load_gsm8k_jsonl(file_path: Union[str, Path]) -> List[PromptItem]:
        """Loads GSM8K format lines: {'question': ..., 'answer': ...}."""
        prompts: List[PromptItem] = []
        path = Path(file_path)
        if not path.exists():
            return prompts

        for idx, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            q = item.get("question", item.get("prompt", ""))
            ans = item.get("answer", "")
            prompts.append(
                PromptItem(
                    prompt_id=f"gsm8k_{idx+1}",
                    domain=DomainType.MATH,
                    difficulty=DifficultyTier.MEDIUM,
                    prompt_text=q,
                    ground_truth=ans,
                    expected_difficulty_rate=0.55,
                )
            )
        return prompts

    @staticmethod
    def load_humaneval_jsonl(file_path: Union[str, Path]) -> List[PromptItem]:
        """Loads HumanEval format lines: {'task_id': ..., 'prompt': ..., 'canonical_solution': ..., 'test': ...}."""
        prompts: List[PromptItem] = []
        path = Path(file_path)
        if not path.exists():
            return prompts

        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            task_id = item.get("task_id", "he_task")
            q = item.get("prompt", "")
            sol = item.get("canonical_solution", "")
            test = item.get("test", "")

            # Create test case structure
            test_cases = [{"raw_test": test}] if test else []

            prompts.append(
                PromptItem(
                    prompt_id=task_id,
                    domain=DomainType.CODE,
                    difficulty=DifficultyTier.HARD,
                    prompt_text=q,
                    ground_truth=sol,
                    test_cases=test_cases,
                    expected_difficulty_rate=0.35,
                )
            )
        return prompts

    @staticmethod
    def create_synthetic_suite(count: int = 10) -> List[PromptItem]:
        """Generates a balanced synthetic reasoning benchmark across tiers."""
        suite: List[PromptItem] = []
        tier_cycle = [DifficultyTier.EASY, DifficultyTier.MEDIUM, DifficultyTier.HARD, DifficultyTier.FRONTIER]

        for i in range(count):
            tier = tier_cycle[i % len(tier_cycle)]
            domain = DomainType.MATH if i % 2 == 0 else DomainType.CODE
            expected_p = {
                DifficultyTier.EASY: 0.85,
                DifficultyTier.MEDIUM: 0.50,
                DifficultyTier.HARD: 0.20,
                DifficultyTier.FRONTIER: 0.05,
            }[tier]

            suite.append(
                PromptItem(
                    prompt_id=f"synth_{i+1}",
                    domain=domain,
                    difficulty=tier,
                    prompt_text=f"Solve problem {i+1} in {tier.value} tier",
                    ground_truth=f"Solution_{i+1}",
                    expected_difficulty_rate=expected_p,
                )
            )
        return suite
