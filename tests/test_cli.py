"""Unit tests for CeilBreak Typer CLI commands."""

import json

from typer.testing import CliRunner

from ceilbreak.cli.main import app

runner = CliRunner()


def test_cli_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "Adaptive Rejection-Sampling-to-GRPO" in res.stdout


def test_cli_train(tmp_path):
    out_dir = str(tmp_path / "runs")
    res = runner.invoke(app, ["train", "--steps", "2", "--group-size", "2", "--output-dir", out_dir])
    assert res.exit_code == 0
    assert "Live Reasoning Telemetry Stream" in res.stdout
    assert "Run completed" in res.stdout


def test_cli_benchmark():
    res = runner.invoke(app, ["benchmark"])
    assert res.exit_code == 0
    assert "Benchmark Comparison" in res.stdout
    assert "Pass@1 Accuracy" in res.stdout


def test_cli_export(tmp_path):
    out_file = str(tmp_path / "test_exp.jsonl")
    res = runner.invoke(app, ["export", "--format", "sft", "--output", out_file])
    assert res.exit_code == 0
    assert "Exported" in res.stdout


def test_cli_transition_check(tmp_path):
    # Create sample telemetry JSON
    sample_file = tmp_path / "telemetry_test.json"
    data = [
        {
            "step": 1,
            "regime": "rest_em_warmup",
            "pass_at_1": 0.5,
            "pass_at_4": 0.6,
            "pass_at_8": 0.7,
            "token_entropy": 0.5,
            "failure_signal_ratio": 0.3,
            "difficulty_skew_index": 1.0,
            "off_policy_kl": 0.0,
            "ref_policy_kl": 0.0,
        },
        {
            "step": 2,
            "regime": "rest_em_warmup",
            "pass_at_1": 0.501,
            "pass_at_4": 0.601,
            "pass_at_8": 0.701,
            "token_entropy": 0.25,
            "failure_signal_ratio": 1.5,
            "difficulty_skew_index": 1.2,
            "off_policy_kl": 0.0,
            "ref_policy_kl": 0.0,
        },
        {
            "step": 3,
            "regime": "rest_em_warmup",
            "pass_at_1": 0.502,
            "pass_at_4": 0.602,
            "pass_at_8": 0.702,
            "token_entropy": 0.20,
            "failure_signal_ratio": 1.6,
            "difficulty_skew_index": 1.2,
            "off_policy_kl": 0.0,
            "ref_policy_kl": 0.0,
        },
    ]
    sample_file.write_text(json.dumps(data))

    res = runner.invoke(app, ["transition-check", str(sample_file), "--window", "3"])
    assert res.exit_code == 0
    assert "Plateau Detection Analysis" in res.stdout
    assert "Plateau Detected" in res.stdout
