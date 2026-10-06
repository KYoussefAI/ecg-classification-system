import json
import numpy as np
import pytest


@pytest.fixture(scope="session")
def fixture_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-ptbxl")
    from scripts.smoke import create_fixture

    create_fixture(root)
    return root


@pytest.fixture(scope="session")
def trained_artifact(tmp_path_factory, fixture_dataset):
    from training.train import parser, train

    run = tmp_path_factory.mktemp("runs") / "synthetic-test-only"
    args = parser().parse_args(
        [
            "--data-dir",
            str(fixture_dataset),
            "--run-dir",
            str(run),
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--smoke",
            "--device",
            "cpu",
        ]
    )
    train(args)
    return run


@pytest.fixture
def signal():
    return np.random.default_rng(42).normal(0, 0.2, (1000, 12)).astype(np.float32)


@pytest.fixture
def client(tmp_path, trained_artifact, monkeypatch):
    """Only this isolated test fixture simulates a PTB-XL artifact for API tests.

    No fixture checkpoint is written to a serving directory or published.
    """
    import shutil
    from training.artifacts import write_json
    from app.services.model_service import ModelService
    from app import database
    from app.main import app
    from fastapi.testclient import TestClient

    directory = tmp_path / "test-contract"
    shutil.copytree(trained_artifact, directory)
    m = json.loads((directory / "model_metadata.json").read_text())
    m["synthetic"] = False
    m["dataset"]["name"] = "PTB-XL"
    write_json(directory / "model_metadata.json", m)
    monkeypatch.setenv("MODEL_DIR", str(directory))
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(ModelService, "_instance", None)
    with TestClient(app) as test_client:
        yield test_client
