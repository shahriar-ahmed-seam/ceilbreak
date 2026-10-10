"""Custom verifier demonstration with partial unit-test credit and math equivalence."""

from ceilbreak.core.rewards import MathSymbolicReward, UnitTestCaseReward


def main():
    print("=== CeilBreak Verifier Suite Demo ===\n")

    # 1. Partial-credit code verification
    sample_code = """
def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
    test_cases = [
        {"function_name": "fibonacci", "inputs": [0], "expected": 0},
        {"function_name": "fibonacci", "inputs": [1], "expected": 1},
        {"function_name": "fibonacci", "inputs": [5], "expected": 5},
        {"function_name": "fibonacci", "inputs": [7], "expected": 13},
    ]

    sig = UnitTestCaseReward.evaluate(sample_code, test_cases)
    print(f"Code Evaluation: passed {sig.unit_tests_passed}/{sig.total_unit_tests} tests")
    print(f"Binary Reward: {sig.binary_reward} | Fractional Reward: {sig.fractional_reward:.2f}")
    print(f"Verifier Feedback: {sig.verifier_feedback}\n")

    # 2. Mathematical Symbolic Equivalence
    math_candidate = "By algebraic simplification, the final result is \\boxed{3/4}."
    ground_truth = "0.75"

    math_sig = MathSymbolicReward.evaluate(math_candidate, ground_truth)
    print("Math Evaluation:")
    print(f"Candidate: {math_candidate}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Match Feedback: {math_sig.verifier_feedback}")
    print(f"Primary Reward: {math_sig.fractional_reward}")


if __name__ == "__main__":
    main()
