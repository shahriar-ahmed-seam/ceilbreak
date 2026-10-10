"""Verifiable reward harnesses for math, code, and partial-credit reasoning."""

from __future__ import annotations

import ast
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional

from ceilbreak.core.types import RewardSignal


class UnitTestCaseReward:
    """Evaluates Python code solutions with partial credit for passing k out of N tests."""

    @staticmethod
    def evaluate(code_str: str, test_cases: List[Dict[str, Any]]) -> RewardSignal:
        """Parses syntax, checks safety, and executes test cases returning fractional credit."""
        # Check syntax safety first
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return RewardSignal(
                binary_reward=0,
                fractional_reward=0.0,
                unit_tests_passed=0,
                total_unit_tests=len(test_cases),
                syntax_valid=False,
                reasoning_length=len(code_str.split()),
                verifier_feedback=f"SyntaxError: {str(e)}",
            )

        if not test_cases:
            # If no tests provided, syntax validity alone gives partial credit
            return RewardSignal(
                binary_reward=1,
                fractional_reward=1.0,
                unit_tests_passed=1,
                total_unit_tests=1,
                syntax_valid=True,
                reasoning_length=len(code_str.split()),
                verifier_feedback="Syntax passed; no test cases required.",
            )

        # Execute tests in restricted global namespace
        scope: Dict[str, Any] = {}
        try:
            compiled = compile(tree, filename="<candidate>", mode="exec")
            exec(compiled, scope)
        except Exception as e:
            return RewardSignal(
                binary_reward=0,
                fractional_reward=0.0,
                unit_tests_passed=0,
                total_unit_tests=len(test_cases),
                syntax_valid=True,
                reasoning_length=len(code_str.split()),
                verifier_feedback=f"RuntimeError during definition: {type(e).__name__}: {str(e)}",
            )

        passed = 0
        feedback_messages: List[str] = []

        for idx, tc in enumerate(test_cases):
            func_name = tc.get("function_name", "solution")
            inputs = tc.get("inputs", [])
            expected = tc.get("expected", None)

            if func_name not in scope or not callable(scope[func_name]):
                feedback_messages.append(f"Function '{func_name}' not defined.")
                break

            try:
                actual = scope[func_name](*inputs)
                if actual == expected:
                    passed += 1
                else:
                    feedback_messages.append(
                        f"Test {idx+1} mismatch: got {repr(actual)}, expected {repr(expected)}"
                    )
            except Exception as e:
                feedback_messages.append(f"Test {idx+1} error: {type(e).__name__}: {str(e)}")

        total = len(test_cases)
        fractional = passed / total if total > 0 else 0.0
        binary = 1 if passed == total else 0
        feedback = "; ".join(feedback_messages) if feedback_messages else "All tests passed."

        return RewardSignal(
            binary_reward=binary,
            fractional_reward=float(fractional),
            unit_tests_passed=passed,
            total_unit_tests=total,
            syntax_valid=True,
            reasoning_length=len(code_str.split()),
            verifier_feedback=feedback,
        )


class MathSymbolicReward:
    """Verifies mathematical reasoning and equivalence with canonical expressions."""

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """Extracts answer enclosed in \\boxed{...}, <answer>...</answer>, or trailing numeric."""
        # 1. LaTeX boxed
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # 2. XML tags
        tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if tag_match:
            return tag_match.group(1).strip()

        # 3. 'The answer is X' pattern
        phrase_match = re.search(r"(?:answer is|equals|result is)\s*[:=]?\s*([^\n\.,;]+)", text, re.IGNORECASE)
        if phrase_match:
            return phrase_match.group(1).strip()

        # 4. Trailing line number
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if lines:
            last_tokens = lines[-1].split()
            if last_tokens:
                return last_tokens[-1].strip("$.,")

        return None

    @classmethod
    def evaluate(cls, solution_text: str, ground_truth: str) -> RewardSignal:
        """Compares extracted candidate answer against ground truth symbolically."""
        extracted = cls.extract_answer(solution_text)
        token_count = len(solution_text.split())

        if not extracted:
            return RewardSignal(
                binary_reward=0,
                fractional_reward=0.0,
                reasoning_length=token_count,
                verifier_feedback="No answer candidate extracted.",
            )

        normalized_cand = cls._normalize_math(extracted)
        normalized_truth = cls._normalize_math(ground_truth)

        # Exact match after normalization
        if normalized_cand == normalized_truth:
            return RewardSignal(
                binary_reward=1,
                fractional_reward=1.0,
                reasoning_length=token_count,
                verifier_feedback="Exact symbolic match.",
            )

        # Numeric float or fraction equivalence check
        try:
            cand_val = cls._to_numeric(normalized_cand)
            truth_val = cls._to_numeric(normalized_truth)
            if cand_val is not None and truth_val is not None and abs(cand_val - truth_val) < 1e-6:
                return RewardSignal(
                    binary_reward=1,
                    fractional_reward=1.0,
                    reasoning_length=token_count,
                    verifier_feedback="Numeric equivalence match.",
                )
        except Exception:
            pass

        return RewardSignal(
            binary_reward=0,
            fractional_reward=0.0,
            reasoning_length=token_count,
            verifier_feedback=f"Mismatch: got '{extracted}' vs expected '{ground_truth}'.",
        )

    @staticmethod
    def _normalize_math(val: str) -> str:
        s = val.strip().lower()
        s = re.sub(r"\s+", "", s)
        s = s.replace("$", "").replace("\\%", "%")
        return s

    @staticmethod
    def _to_numeric(val: str) -> Optional[float]:
        try:
            if "/" in val:
                f = Fraction(val)
                return float(f)
            return float(val)
        except (ValueError, ZeroDivisionError):
            return None


class FormatVerificationReward:
    """Verifies compliance with reasoning protocol tags (<think>...</think>)."""

    @staticmethod
    def evaluate(text: str, min_tokens: int = 10, max_tokens: int = 4096) -> float:
        """Returns 1.0 for valid formatting with thought tags, 0.5 for answer only."""
        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        tokens = len(text.split())
        within_bounds = min_tokens <= tokens <= max_tokens

        score = 0.0
        if has_think:
            score += 0.5
        if within_bounds:
            score += 0.5
        return score
