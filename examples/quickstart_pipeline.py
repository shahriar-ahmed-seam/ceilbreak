"""Minimal runnable example demonstrating ReST-EM to GRPO transition in CeilBreak."""

from ceilbreak.core.pipeline import ReasoningPipeline


def main():
    print("Initializing CeilBreak Reasoning Pipeline...")
    pipeline = ReasoningPipeline(group_size=4, plateau_window=3, auto_transition=True)

    print("\nRunning training iterations:")
    print("Step | Regime           | Pass@1 | Pass@4 | Entropy | Plateau Conf")
    print("-" * 65)

    for step in range(1, 9):
        snap = pipeline.run_step(step=step)
        print(
            f"{snap.step:4d} | "
            f"{snap.regime.value:16s} | "
            f"{snap.pass_at_1 * 100:5.1f}% | "
            f"{snap.pass_at_4 * 100:5.1f}% | "
            f"{snap.token_entropy:7.3f} | "
            f"{snap.plateau_confidence * 100:5.1f}%"
        )

    latest = pipeline.telemetry.get_latest()
    print("-" * 65)
    print(f"Final Regime: {latest.regime.value}")
    print(f"Final Pass@1 Accuracy: {latest.pass_at_1 * 100:.1f}%")
    print(f"Transition Occurred: {pipeline.transition_gate.state.plateau_detected}")


if __name__ == "__main__":
    main()
