import io
import json
import sqlite3
import numpy as np
import pytest
from training.config import CLASSES, LEADS
from app.services.model_service import ModelService


def register(client, name="researcher"):
    response = client.post(
        "/api/auth/register",
        json={
            "username": name,
            "email": f"{name}@example.com",
            "password": "a-long-test-password",
        },
    )
    assert response.status_code == 201, response.text
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def test_health_and_prediction_schema(client, signal):
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["model_loaded"] is True
    response = client.post(
        "/api/predict/", json={"signal_data": signal.tolist(), "leads": LEADS}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["id"] is None
    assert [p["class"] for p in result["predictions"]] == CLASSES
    assert all(0 <= p["score"] <= 1 and "threshold" in p for p in result["predictions"])
    assert "confidence" not in result
    assert result["signal_quality"]["status"] == "passed"
    assert client.get("/api/model").json()["test_metrics"] is None


def test_preprocessing_equal_to_training(client, signal):
    import torch
    from training.preprocessing import preprocess

    svc = ModelService.get_instance()
    expected_input = torch.from_numpy(
        preprocess(signal, svc.metadata["normalization"])
    ).unsqueeze(0)
    with torch.inference_mode():
        expected = svc.model(expected_input).sigmoid().numpy()[0]
    result = svc.predict(signal)
    np.testing.assert_allclose(
        [p["score"] for p in result["predictions"]], expected, atol=1e-7
    )


def test_no_anonymous_or_implicit_storage(client, signal):
    from app.database import DB_PATH

    body = {"signal_data": signal.tolist(), "case_id": "deidentified-1"}
    client.post("/api/predict/", json=body)
    headers = register(client)
    client.post("/api/predict/", json=body, headers=headers)
    with sqlite3.connect(DB_PATH) as db:
        assert db.execute("SELECT count(*) FROM screening_results").fetchone()[0] == 0
    assert (
        client.post("/api/predict/", json={**body, "save_history": True}).status_code
        == 401
    )
    assert (
        client.post(
            "/api/predict/", json=body, headers={"Authorization": "Bearer invalid"}
        ).status_code
        == 401
    )


def test_private_history_and_delete(client, signal):
    first, second = register(client), register(client, "another")
    response = client.post(
        "/api/predict/",
        json={
            "signal_data": signal.tolist(),
            "save_history": True,
            "case_id": "case-001",
        },
        headers=first,
    )
    assert response.status_code == 200, response.text
    id_ = response.json()["id"]
    assert len(client.get("/api/history/", headers=first).json()) == 1
    assert client.get("/api/history/", headers=second).json() == []
    assert client.get("/api/history/").status_code == 401
    assert client.delete(f"/api/history/{id_}", headers=second).status_code == 404
    assert client.delete(f"/api/history/{id_}", headers=first).status_code == 200
    assert client.get("/api/history/", headers=first).json() == []


@pytest.mark.parametrize("suffix", ["json", "csv", "npy"])
def test_upload_parsing_and_prediction(client, signal, suffix):
    if suffix == "json":
        content = json.dumps({"signal_data": signal.tolist(), "leads": LEADS}).encode()
    elif suffix == "csv":
        stream = io.StringIO()
        np.savetxt(stream, signal, delimiter=",", header=",".join(LEADS), comments="")
        content = stream.getvalue().encode()
    else:
        stream = io.BytesIO()
        np.save(stream, signal)
        content = stream.getvalue()
    response = client.post(
        "/api/predict/validate-file", files={"file": ("signal." + suffix, content)}
    )
    assert response.status_code == 200, response.text
    np.testing.assert_allclose(response.json()["signal_data"], signal)
    assert (
        client.post(
            "/api/predict/file", files={"file": ("signal." + suffix, content)}
        ).status_code
        == 200
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 101])
def test_invalid_samples(client, signal, value):
    signal[0, 0] = value
    response = client.post(
        "/api/predict/",
        content=json.dumps({"signal_data": signal.tolist()}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422, response.text


def test_bad_shape_leads_rate(client, signal):
    for body in (
        {"signal_data": signal[:50].tolist()},
        {"signal_data": signal.tolist(), "sample_rate": 500},
        {"signal_data": signal.tolist(), "leads": LEADS[::-1]},
        {"signal_data": np.zeros((1000, 12)).tolist()},
    ):
        assert client.post("/api/predict/", json=body).status_code == 422
    assert (
        client.post(
            "/api/predict/validate-file", files={"file": ("x.csv", b"II,I\n1,2")}
        ).status_code
        == 422
    )
    bad = io.BytesIO()
    np.save(bad, np.array([{}], dtype=object))
    assert (
        client.post(
            "/api/predict/validate-file", files={"file": ("x.npy", bad.getvalue())}
        ).status_code
        == 422
    )


def test_missing_model_fail_closed(client, signal, tmp_path, monkeypatch):
    service = ModelService(tmp_path / "missing")
    monkeypatch.setattr(ModelService, "_instance", service)
    assert service.model is None
    assert client.get("/health").status_code == 503
    response = client.post("/api/predict/", json={"signal_data": signal.tolist()})
    assert response.status_code == 503 and "predictions" not in response.json()


def test_synthetic_model_never_served(trained_artifact):
    assert ModelService(trained_artifact).state == "unavailable"


def test_corrupt_model_health_and_prediction(client, signal, monkeypatch):
    directory = ModelService.get_instance().directory
    (directory / "best_model.pt").write_bytes(b"broken")
    monkeypatch.setattr(ModelService, "_instance", ModelService(directory))
    assert client.get("/health").json()["model_loaded"] is False
    assert (
        client.post("/api/predict/", json={"signal_data": signal.tolist()}).status_code
        == 503
    )


def test_request_size_limit(client):
    assert (
        client.post(
            "/api/predict/validate-file",
            files={"file": ("huge.json", b"x" * (3 * 1024 * 1024))},
        ).status_code
        == 413
    )


def test_synthetic_label_survives_saved_history(client, signal):
    headers = register(client)
    body = {"signal_data": signal.tolist(), "synthetic": True, "save_history": True}
    assert client.post("/api/predict/", json=body, headers=headers).status_code == 200
    saved = client.get("/api/history/", headers=headers).json()[0]
    assert any("Synthetic input" in warning for warning in saved["warnings"])


def test_production_secret_guard():
    from app.services.auth_service import validate_secret

    for secret in (None, "placeholder-secret", "x" * 40, "change-me" + "aB9" * 20):
        with pytest.raises(RuntimeError):
            validate_secret(secret, "production")
    assert len(validate_secret(None, "development")) >= 32


def test_norm_conflict_is_transparent(client, signal):
    svc = ModelService.get_instance()
    svc.metadata["thresholds"] = dict.fromkeys(CLASSES, 0.0)
    result = svc.predict(signal)
    assert result["positive_classes"] == CLASSES
    assert any("both crossed" in w for w in result["warnings"])
