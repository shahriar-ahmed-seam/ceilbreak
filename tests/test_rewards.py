"""Unit tests for verifiable reward harnesses."""

from ceilbreak.core.rewards import (
    FormatVerificationReward,
    MathSymbolicReward,
    UnitTestCaseReward,
)


def test_unit_test_syntax_error():
    broken_code = "def solve(\n    return 42"
    tests = [{"function_name": "solve", "inputs": [], "expected": 42}]
    sig = UnitTestCaseReward.evaluate(broken_code, tests)

    assert sig.binary_reward == 0
    assert sig.fractional_reward == 0.0
    assert not sig.syntax_valid
    assert "SyntaxError" in sig.verifier_feedback


def test_unit_test_partial_credit():
    # Function passes 2 out of 3 tests
    code = """
def is_even(n):
    if n == 2:
        return True
    if n == 4:
        return True
    return False
"""
    tests = [
        {"function_name": "is_even", "inputs": [2], "expected": True},
        {"function_name": "is_even", "inputs": [4], "expected": True},
        {"function_name": "is_even", "inputs": [6], "expected": True},  # will fail
    ]
    sig = UnitTestCaseReward.evaluate(code, tests)

    # In binary filtering, this was discarded as 0/10. Here, 2/3 = 0.6667 partial credit!
    assert sig.binary_reward == 0
    assert sig.unit_tests_passed == 2
    assert sig.total_unit_tests == 3
    assert abs(sig.fractional_reward - (2 / 3)) < 1e-4
    assert sig.syntax_valid
    assert "Test 3 mismatch" in sig.verifier_feedback


def test_unit_test_full_pass():
    code = """
def add(a, b):
    return a + b
"""
    tests = [
        {"function_name": "add", "inputs": [1, 2], "expected": 3},
        {"function_name": "add", "inputs": [10, -5], "expected": 5},
    ]
    sig = UnitTestCaseReward.evaluate(code, tests)
    assert sig.binary_reward == 1
    assert sig.fractional_reward == 1.0
    assert sig.unit_tests_passed == 2


def test_math_symbolic_boxed_extraction():
    sol = "Let us derive the answer step by step. Thus \\boxed{3/4} is the answer."
    sig = MathSymbolicReward.evaluate(sol, "0.75")
    assert sig.binary_reward == 1
    assert sig.fractional_reward == 1.0
    assert "equivalence match" in sig.verifier_feedback.lower()


def test_math_symbolic_tag_extraction():
    sol = "<think>Compute 12 * 12 = 144</think><answer>144</answer>"
    sig = MathSymbolicReward.evaluate(sol, "144")
    assert sig.binary_reward == 1
    assert sig.fractional_reward == 1.0


def test_math_symbolic_mismatch():
    sol = "The answer is 42."
    sig = MathSymbolicReward.evaluate(sol, "100")
    assert sig.binary_reward == 0
    assert sig.fractional_reward == 0.0
    assert "Mismatch" in sig.verifier_feedback


def test_format_verification_reward():
    valid = "<think>Step 1: factorize equation.</think>\n<answer>5</answer>"
    score_valid = FormatVerificationReward.evaluate(valid, min_tokens=5, max_tokens=100)
    assert score_valid == 1.0

    no_think = "Just the answer 5."
    score_no_think = FormatVerificationReward.evaluate(no_think, min_tokens=1, max_tokens=100)
    assert score_no_think == 0.5
