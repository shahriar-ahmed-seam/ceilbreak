"""Unit tests for FastAPI server endpoints."""

from fastapi.testclient import TestClient

from ceilbreak.server.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ceilbreak-telemetry"


def test_telemetry_latest_endpoint():
    res = client.get("/api/telemetry/latest")
    assert res.status_code == 200
    data = res.json()
    assert "snapshot" in data
    assert "active_regime" in data


def test_pipeline_step_endpoint():
    res = client.post("/api/pipeline/step")
    assert res.status_code == 200
    data = res.json()
    assert data["step"] >= 1
    assert "snapshot" in data


def test_pipeline_run_batch():
    res = client.post("/api/pipeline/run", json={"total_steps": 3})
    assert res.status_code == 200
    data = res.json()
    assert data["executed_steps"] == 3


def test_pipeline_transition_override():
    res = client.post("/api/pipeline/transition", json={"force_regime": "grpo_online"})
    assert res.status_code == 200
    data = res.json()
    assert data["active_regime"] == "grpo_online"


def test_inspect_groups_endpoint():
    res = client.get("/api/inspect/groups")
    assert res.status_code == 200
    data = res.json()
    assert "groups" in data
    assert len(data["groups"]) > 0


def test_serve_dashboard():
    res = client.get("/")
    assert res.status_code == 200
    assert "CeilBreak Telemetry Console" in res.text
