# Changelog

All notable changes to CeilBreak are documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

---

## [0.1.0] - 2026-10-10

### Added
- Core architecture data models and protocol schemas (`Trajectory`, `RolloutGroup`, `PromptItem`, `RewardSignal`, `AdvantageEstimate`, `TelemetrySnapshot`, `TransitionState`).
- ReST-EM engine implementing Expectation (rejection sampling on verifier threshold) and Maximization (positive-only MLE loss and difficulty skew analyzer).
- Group Relative Policy Optimization (GRPO) advantage engine with group mean and standard deviation baseline normalization, clipped surrogate objective, and reference policy KL divergence regularizer.
- Multi-objective verifiable reward harnesses:
  - Unit test code execution verifier with partial credit ($k/N$ assertions passed).
  - Mathematical symbolic equivalence verifier with LaTeX `\boxed{...}`, XML `<answer>`, and fraction normalization.
  - Reasoning length and format compliance verifier.
- Reasoning telemetry engine computing unbiased hypergeometric pass@k (k=1, 4, 8), token Shannon entropy, and failure-to-success signal ratios.
- Statistical plateau detector and automated TransitionGate state machine (`REST_EM_WARMUP` -> `PLATEAU_DETECTED` -> `GRPO_ONLINE`).
- ReasoningPipeline orchestrator connecting rollout groups, verifiers, telemetry, and automated regime transitions.
- Anti-AI-slop developer telemetry console and FastAPI streaming server with Linear/Datadog developer tool design standards.
- Typer + Rich CLI suite with subcommands: `train`, `transition-check`, `serve`, `export`, and `benchmark`.
- Dataset loaders for GSM8K and HumanEval formats, with synthetic reasoning suite generator.
- SFT ShareGPT and GRPO RL JSONL export adapters.
- Comprehensive publication documentation suite (`README.md`, `docs/ARCHITECTURE.md`, `docs/SPECIFICATION.md`, `docs/BENCHMARKS.md`, `CITATION.cff`, `LICENSE`, `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`).
