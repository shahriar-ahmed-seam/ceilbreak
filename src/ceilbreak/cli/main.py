"""Typer command-line interface for CeilBreak."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ceilbreak.adapters.dataset import DatasetAdapter
from ceilbreak.adapters.export import ExportAdapter
from ceilbreak.core.pipeline import ReasoningPipeline
from ceilbreak.core.transition_gate import PlateauDetector
from ceilbreak.core.types import (
    TelemetrySnapshot,
    TransitionRegime,
)

app = typer.Typer(
    name="ceilbreak",
    help="Adaptive Rejection-Sampling-to-GRPO Transition Engine & Telemetry Suite for LLM Reasoning",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    steps: int = typer.Option(8, "--steps", "-s", help="Total pipeline training steps to execute"),
    group_size: int = typer.Option(4, "--group-size", "-g", help="Rollouts sampled per prompt group"),
    plateau_window: int = typer.Option(3, "--window", "-w", help="Sliding window size for plateau detection"),
    auto_transition: bool = typer.Option(True, "--auto-transition/--manual", help="Automatically switch to GRPO when plateau detected"),
    output_dir: str = typer.Option("data/runs", "--output-dir", "-o", help="Directory to save telemetry results"),
):
    """Executes reasoning pipeline run, detecting ReST-EM plateaus and switching to GRPO."""
    console.print(Panel.fit(
        "[bold cyan]CeilBreak Reasoning Engine[/bold cyan]\n"
        "[dim]Rejection Sampling vs RL: Transition Telemetry[/dim]",
        border_style="cyan",
    ))

    prompts = DatasetAdapter.create_synthetic_suite(count=8)
    pipeline = ReasoningPipeline(
        prompts=prompts,
        group_size=group_size,
        plateau_window=plateau_window,
        auto_transition=auto_transition,
    )

    table = Table(title="Live Reasoning Telemetry Stream", border_style="dim")
    table.add_column("Step", justify="center", style="bold")
    table.add_column("Regime", justify="center")
    table.add_column("Pass@1", justify="right")
    table.add_column("Pass@4", justify="right")
    table.add_column("Entropy", justify="right")
    table.add_column("Failure Signal", justify="right")
    table.add_column("Plateau Conf", justify="right")

    for s in range(1, steps + 1):
        snap = pipeline.run_step(step=s)
        regime_style = "green" if snap.regime == TransitionRegime.GRPO_ONLINE else "yellow"
        table.add_row(
            str(snap.step),
            f"[{regime_style}]{snap.regime.value}[/{regime_style}]",
            f"{snap.pass_at_1 * 100:.1f}%",
            f"{snap.pass_at_4 * 100:.1f}%",
            f"{snap.token_entropy:.3f}",
            f"{snap.failure_signal_ratio:.2f}x",
            f"{snap.plateau_confidence * 100:.1f}%",
        )

    console.print(table)

    out_path = Path(output_dir) / "telemetry_run.json"
    pipeline.export_telemetry_json(str(out_path))
    console.print(f"[bold green]Run completed.[/bold green] Telemetry exported to: {out_path}")


@app.command()
def transition_check(
    telemetry_file: str = typer.Argument(..., help="Path to telemetry JSON log file"),
    window: int = typer.Option(3, "--window", "-w", help="Window size for slope evaluation"),
):
    """Evaluates whether current ReST-EM checkpoint has plateaued and should switch to GRPO."""
    path = Path(telemetry_file)
    if not path.exists():
        console.print(f"[bold red]File not found:[/bold red] {telemetry_file}")
        raise typer.Exit(code=1)

    raw = json.loads(path.read_text())
    snapshots = [TelemetrySnapshot(**item) for item in raw]

    detector = PlateauDetector(window_size=window)
    is_plateau, conf, reason = detector.evaluate_plateau(snapshots)

    console.print(Panel.fit(
        f"[bold]Plateau Detection Analysis[/bold]\n\n"
        f"Plateau Detected: [{'bold red' if is_plateau else 'bold green'}]{is_plateau}[/]\n"
        f"Confidence: {conf * 100:.1f}%\n"
        f"Evaluation Rationale: {reason}\n\n"
        f"[bold]Recommendation:[/bold] "
        + ("Switch to GRPO immediately; exploit relative surprise and negative feedback."
           if is_plateau else "Continue ReST-EM warmup filtering."),
        border_style="magenta",
    ))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on"),
):
    """Starts the CeilBreak FastAPI server and developer telemetry console."""
    console.print(f"[bold green]Starting CeilBreak Console[/bold green] on http://{host}:{port}")
    uvicorn.run("ceilbreak.server.app:app", host=host, port=port, log_level="info")


@app.command()
def export(
    format_type: str = typer.Option("sft", "--format", "-f", help="Export format: 'sft' or 'rl'"),
    output_file: str = typer.Option("data/export.jsonl", "--output", "-o", help="Output JSONL file path"),
):
    """Exports synthetic rollouts to ShareGPT SFT or GRPO RL trajectory format."""
    prompts = DatasetAdapter.create_synthetic_suite(count=5)
    pipeline = ReasoningPipeline(prompts=prompts, group_size=4)
    pipeline.run_step(step=1)

    all_trajs = []
    groups = [pipeline._sample_rollout_group(p, 1) for p in prompts]
    for g in groups:
        all_trajs.extend(g.trajectories)

    if format_type.lower() == "sft":
        count = ExportAdapter.export_sft_sharegpt(all_trajs, output_file)
        console.print(f"Exported {count} verified SFT examples to {output_file}")
    else:
        count = ExportAdapter.export_grpo_rl_buffer(groups, output_file)
        console.print(f"Exported {count} RL rollout groups to {output_file}")


@app.command()
def benchmark():
    """Runs comparative benchmark of ReST-EM vs GRPO across difficulty tiers."""
    console.print("[bold]Running ReST-EM vs GRPO Benchmark...[/bold]\n")

    # ReST-EM warmup run
    rest_pipe = ReasoningPipeline(group_size=4, auto_transition=False)
    for _ in range(6):
        rest_pipe.run_step()
    rest_snap = rest_pipe.telemetry.get_latest()

    # GRPO online run
    grpo_pipe = ReasoningPipeline(group_size=4, auto_transition=True)
    # Step 1-3 warmup, then force switch
    for s in range(1, 4):
        grpo_pipe.run_step(s)
    grpo_pipe.transition_gate.check_and_update(4, grpo_pipe.telemetry.history, force_transition=True)
    grpo_pipe.active_regime = TransitionRegime.GRPO_ONLINE
    for s in range(4, 7):
        grpo_pipe.run_step(s)
    grpo_snap = grpo_pipe.telemetry.get_latest()

    table = Table(title="Benchmark Comparison: ReST-EM Plateau vs GRPO Online Advantage", border_style="cyan")
    table.add_column("Evaluation Metric", style="bold")
    table.add_column("ReST-EM (Filtering)", justify="right", style="yellow")
    table.add_column("GRPO (Advantage RL)", justify="right", style="green")
    table.add_column("Delta / Impact", justify="right", style="cyan")

    table.add_row(
        "Pass@1 Accuracy",
        f"{rest_snap.pass_at_1 * 100:.1f}%",
        f"{grpo_snap.pass_at_1 * 100:.1f}%",
        f"+{(grpo_snap.pass_at_1 - rest_snap.pass_at_1) * 100:.1f}%",
    )
    table.add_row(
        "Pass@4 Budget",
        f"{rest_snap.pass_at_4 * 100:.1f}%",
        f"{grpo_snap.pass_at_4 * 100:.1f}%",
        f"+{(grpo_snap.pass_at_4 - rest_snap.pass_at_4) * 100:.1f}%",
    )
    table.add_row(
        "Token Shannon Entropy",
        f"{rest_snap.token_entropy:.3f}",
        f"{grpo_snap.token_entropy:.3f}",
        "Preserves exploration",
    )
    table.add_row(
        "Failure Signal Utilization",
        "0.00x (r=0 discarded)",
        f"{grpo_snap.failure_signal_ratio:.2f}x (Penalized)",
        "Active error gradient",
    )
    table.add_row(
        "Difficulty Skew Mitigation",
        f"{rest_snap.difficulty_skew_index:.1f}:1 (Easy dominated)",
        "Zero-advantage on trivial",
        "Frontier focus",
    )

    console.print(table)


if __name__ == "__main__":
    app()
