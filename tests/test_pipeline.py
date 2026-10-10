"""Unit tests for ReasoningPipeline end-to-end execution and transition."""

import json

from ceilbreak.core.pipeline import ReasoningPipeline
from ceilbreak.core.types import TransitionRegime


def test_pipeline_single_step():
    pipeline = ReasoningPipeline(group_size=2)
    snap = pipeline.run_step(step=1)
    assert snap.step == 1
    assert snap.regime == TransitionRegime.REST_EM_WARMUP
    assert snap.pass_at_1 >= 0.0


def test_pipeline_multi_step_run_and_export(tmp_path):
    pipeline = ReasoningPipeline(group_size=2, plateau_window=3)
    history = pipeline.run_pipeline(total_steps=6)
    assert len(history) == 6

    # Verify export
    export_file = str(tmp_path / "telemetry_run.json")
    pipeline.export_telemetry_json(export_file)

    with open(export_file) as f:
        data = json.load(f)
    assert len(data) == 6
    assert data[0]["step"] == 1
