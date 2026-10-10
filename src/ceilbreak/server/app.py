"""FastAPI server for CeilBreak telemetry streaming and developer console."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ceilbreak.core.pipeline import ReasoningPipeline
from ceilbreak.core.types import (
    TelemetrySnapshot,
    TransitionRegime,
)

# Initialize FastAPI application
app = FastAPI(
    title="CeilBreak Telemetry Engine",
    description="Adaptive Rejection-Sampling-to-GRPO Transition Engine & Telemetry Console",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance for console session
_pipeline: Optional[ReasoningPipeline] = None


def get_pipeline() -> ReasoningPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ReasoningPipeline(group_size=4, plateau_window=3, auto_transition=True)
    return _pipeline


class StepRequest(BaseModel):
    step: Optional[int] = None


class RunRequest(BaseModel):
    total_steps: int = Field(default=8, ge=1, le=50)


class TransitionOverrideRequest(BaseModel):
    force_regime: TransitionRegime = Field(default=TransitionRegime.GRPO_ONLINE)


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "ceilbreak-telemetry",
        "version": "0.1.0",
        "timestamp": "2026-10-10T08:00:00Z",
    }


@app.get("/api/telemetry/latest")
def get_latest_telemetry() -> Dict[str, Any]:
    pipe = get_pipeline()
    latest = pipe.telemetry.get_latest()
    if not latest:
        # Run step 1 if not yet initialized
        latest = pipe.run_step(step=1)
    return {
        "snapshot": latest.model_dump(),
        "transition_state": pipe.transition_gate.state.model_dump(),
        "active_regime": pipe.active_regime.value,
        "step": pipe.current_step,
    }


@app.get("/api/telemetry/history")
def get_telemetry_history() -> List[Dict[str, Any]]:
    pipe = get_pipeline()
    if not pipe.telemetry.history:
        pipe.run_step(step=1)
    return [s.model_dump() for s in pipe.telemetry.history]


@app.post("/api/pipeline/step")
def step_pipeline(req: Optional[StepRequest] = None) -> Dict[str, Any]:
    pipe = get_pipeline()
    step_num = req.step if req and req.step else None
    snapshot = pipe.run_step(step=step_num)
    return {
        "snapshot": snapshot.model_dump(),
        "transition_state": pipe.transition_gate.state.model_dump(),
        "active_regime": pipe.active_regime.value,
        "step": pipe.current_step,
    }


@app.post("/api/pipeline/run")
def run_pipeline_steps(req: RunRequest) -> Dict[str, Any]:
    pipe = get_pipeline()
    results: List[TelemetrySnapshot] = []
    for _ in range(req.total_steps):
        snap = pipe.run_step()
        results.append(snap)
    return {
        "executed_steps": len(results),
        "total_history_count": len(pipe.telemetry.history),
        "latest_snapshot": pipe.telemetry.get_latest().model_dump() if pipe.telemetry.get_latest() else None,
        "transition_state": pipe.transition_gate.state.model_dump(),
        "active_regime": pipe.active_regime.value,
    }


@app.post("/api/pipeline/reset")
def reset_pipeline() -> Dict[str, Any]:
    global _pipeline
    _pipeline = ReasoningPipeline(group_size=4, plateau_window=3, auto_transition=True)
    init_snap = _pipeline.run_step(step=1)
    return {
        "message": "Pipeline reset to Step 1 initial state.",
        "snapshot": init_snap.model_dump(),
        "active_regime": _pipeline.active_regime.value,
    }


@app.post("/api/pipeline/transition")
def override_transition(req: TransitionOverrideRequest) -> Dict[str, Any]:
    pipe = get_pipeline()
    trans_state = pipe.transition_gate.check_and_update(
        step=pipe.current_step,
        history=pipe.telemetry.history,
        force_transition=True,
    )
    pipe.active_regime = trans_state.current_regime
    return {
        "transition_state": trans_state.model_dump(),
        "active_regime": pipe.active_regime.value,
    }


@app.get("/api/inspect/groups")
def inspect_rollouts() -> Dict[str, Any]:
    pipe = get_pipeline()
    # Sample fresh groups for inspection
    groups = [pipe._sample_rollout_group(p, pipe.current_step) for p in pipe.prompts]
    serialized = []
    for g in groups:
        advs = pipe.grpo_engine.compute_group_advantages(g)
        adv_map = {a.trajectory_id: a.advantage for a in advs}
        t_list = []
        for t in g.trajectories:
            t_data = t.model_dump()
            t_data["advantage"] = adv_map.get(t.trajectory_id, 0.0)
            t_list.append(t_data)
        serialized.append({
            "group_id": g.group_id,
            "prompt": g.prompt.model_dump(),
            "group_size": g.group_size,
            "pass_rate": g.empirical_pass_rate,
            "reward_mean": g.reward_mean,
            "reward_std": g.reward_std,
            "trajectories": t_list,
        })
    return {
        "step": pipe.current_step,
        "regime": pipe.active_regime.value,
        "groups": serialized,
    }


# Static directory setup
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<html><body><h1>CeilBreak Console Initializing...</h1></body></html>")
