import json
from argparse import Namespace
import numpy as np
import pytest
import torch
from training.artifacts import load_artifact, validate_metadata
from training.config import CLASSES, LEADS
from training.data import load_metadata, validate_partitions
from training.metrics import select_thresholds
from training.model import ECGNet
from training.preprocessing import fit_normalization, preprocess, validate_signal


def test_model_shape_and_deterministic_eval():
    torch.set_num_threads(4)
    model = ECGNet().eval()
    x = torch.randn(2, 12, 1000)
    with torch.inference_mode():
        a, b = model(x), model(x)
    assert a.shape == (2, 5)
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        model(torch.zeros(2, 1000, 12))


def test_official_partitions_and_labels(fixture_dataset):
    parts, audit = load_metadata(fixture_dataset)
    assert set(parts["train"].strat_fold) == set(range(1, 9))
    assert set(parts["validation"].strat_fold) == {9}
    assert set(parts["test"].strat_fold) == {10}
    assert parts["train"].labels.iloc[0] == [1, 0, 0, 1, 0]
    assert audit["partitions"]["train"]["records"] == 64
    import pandas as pd

    frame = pd.concat(parts.values(), ignore_index=True)
    frame.loc[frame.strat_fold == 10, "patient_id"] = frame.patient_id.iloc[0]
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_partitions(frame)


def test_duplicate_metadata_rejected(fixture_dataset):
    import pandas as pd

    parts, _ = load_metadata(fixture_dataset)
    frame = pd.concat(parts.values(), ignore_index=True)
    frame.loc[1, "filename_lr"] = frame.loc[0, "filename_lr"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_partitions(frame)


def test_train_only_normalization(signal):
    norm = fit_normalization([signal, signal + 1], [1, 8])
    expected = np.stack([signal, signal + 1]).mean((0, 1))
    np.testing.assert_allclose(norm["mean"], expected, atol=1e-6)
    assert preprocess(signal, norm).shape == (12, 1000)
    for invalid in ([9], [10], [1, 9]):
        with pytest.raises(ValueError, match="training folds"):
            fit_normalization([signal], invalid)


def test_threshold_selection_fold_guard():
    y = np.tile([[0] * 5, [1] * 5], (4, 1))
    scores = y * 0.7 + 0.1
    assert list(select_thresholds(y, scores, fold=9)) == CLASSES
    with pytest.raises(ValueError, match="fold 9"):
        select_thresholds(y, scores, fold=10)


def test_checkpoint_roundtrip_and_preprocess_consistency(trained_artifact, signal):
    a, metadata = load_artifact(trained_artifact)
    b, _ = load_artifact(trained_artifact)
    x = torch.from_numpy(preprocess(signal, metadata["normalization"])).unsqueeze(0)
    with torch.inference_mode():
        assert torch.equal(a(x), b(x))
    thresholds = json.loads((trained_artifact / "thresholds.json").read_text())[
        "thresholds"
    ]
    assert thresholds == metadata["thresholds"]
    assert metadata["test_metrics"] is None


@pytest.mark.parametrize(
    "change",
    [
        "classes",
        "lead_order",
        "thresholds",
        "normalization",
        "architecture",
        "calibration",
    ],
)
def test_bad_metadata_rejected(trained_artifact, change):
    m = json.loads((trained_artifact / "model_metadata.json").read_text())
    if change == "classes":
        m["classes"].reverse()
    elif change == "lead_order":
        m["input"]["leads"].reverse()
    elif change == "thresholds":
        m["thresholds"]["MI"] = float("nan")
    elif change == "normalization":
        m["normalization"]["fit_folds"] = [9]
    elif change == "architecture":
        m["architecture"]["name"] = "legacy"
    else:
        m["calibration"] = {"method": "invented"}
    with pytest.raises(ValueError):
        validate_metadata(m)


def test_corrupt_checkpoint(trained_artifact, tmp_path):
    import shutil

    directory = tmp_path / "bad"
    shutil.copytree(trained_artifact, directory)
    (directory / "best_model.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(directory)


def test_evaluation_frozen_and_one_time(trained_artifact, fixture_dataset, tmp_path):
    import shutil
    from training.evaluate import evaluate

    directory = tmp_path / "eval"
    shutil.copytree(trained_artifact, directory)
    before = json.loads((directory / "model_metadata.json").read_text())
    args = Namespace(
        run_dir=directory,
        data_dir=fixture_dataset,
        device="cpu",
        batch_size=8,
        bootstrap=10,
        retry_failed=False,
    )
    report = evaluate(args)
    after = json.loads((directory / "model_metadata.json").read_text())
    assert report["fold"] == 10 and report["record_count"] == 8
    assert before["thresholds"] == after["thresholds"]
    assert before["checkpoint_sha256"] == after["checkpoint_sha256"]
    assert report["synthetic"] is True
    assert len(list((directory / "plots").glob("*.png"))) == 5
    with pytest.raises(ValueError, match="already"):
        evaluate(args)


@pytest.mark.parametrize(
    "kind", ["shape", "nan", "inf", "flatline", "amplitude", "leads", "rate", "units"]
)
def test_signal_quality(signal, kind):
    options = {}
    if kind == "shape":
        signal = signal.T
    elif kind == "nan":
        signal[0, 0] = np.nan
    elif kind == "inf":
        signal[0, 0] = np.inf
    elif kind == "flatline":
        signal[:, 4] = 0
    elif kind == "amplitude":
        signal[0, 0] = 101
    elif kind == "leads":
        options["leads"] = LEADS[::-1]
    elif kind == "rate":
        options["sample_rate"] = 500
    else:
        options["units"] = "V"
    with pytest.raises(ValueError):
        validate_signal(signal, **options)


def test_training_does_not_read_test_waveforms(fixture_dataset, tmp_path, monkeypatch):
    import training.data as data
    from training.train import parser, train

    original = data.read_signal
    calls = []

    def guarded(root, name):
        assert int(name.split("_")[1]) <= 72, "Training tried to read test waveforms"
        calls.append(name)
        return original(root, name)

    monkeypatch.setattr(data, "read_signal", guarded)
    run = tmp_path / "guarded"
    train(
        parser().parse_args(
            [
                "--data-dir",
                str(fixture_dataset),
                "--run-dir",
                str(run),
                "--epochs",
                "1",
                "--smoke",
            ]
        )
    )
    assert len(calls) == 72


def test_resume_matches_uninterrupted(fixture_dataset, tmp_path, monkeypatch):
    import training.train as trainer

    original = trainer.atomic_save
    interrupted = tmp_path / "interrupted"

    def fail_after_last(value, path):
        original(value, path)
        if path.name == "last_checkpoint.pt":
            raise RuntimeError("simulated interruption")

    args = [
        "--data-dir",
        str(fixture_dataset),
        "--epochs",
        "2",
        "--smoke",
        "--batch-size",
        "16",
    ]
    monkeypatch.setattr(trainer, "atomic_save", fail_after_last)
    with pytest.raises(RuntimeError, match="simulated"):
        trainer.train(
            trainer.parser().parse_args([*args, "--run-dir", str(interrupted)])
        )
    monkeypatch.setattr(trainer, "atomic_save", original)
    trainer.train(
        trainer.parser().parse_args([*args, "--run-dir", str(interrupted), "--resume"])
    )
    full = tmp_path / "full"
    trainer.train(trainer.parser().parse_args([*args, "--run-dir", str(full)]))
    a, ma = load_artifact(interrupted)
    b, mb = load_artifact(full)
    assert ma["thresholds"] == mb["thresholds"]
    assert all(torch.equal(a.state_dict()[k], v) for k, v in b.state_dict().items())


def test_evaluation_finalization_recovery(
    trained_artifact, fixture_dataset, tmp_path, monkeypatch
):
    import shutil
    import training.evaluate as evaluator

    directory = tmp_path / "recover"
    shutil.copytree(trained_artifact, directory)
    args = Namespace(
        run_dir=directory,
        data_dir=fixture_dataset,
        device="cpu",
        batch_size=8,
        bootstrap=0,
        retry_failed=False,
    )
    finalize = evaluator.finalize_metadata

    def interrupt(*args):
        raise RuntimeError("interrupted finalization")

    monkeypatch.setattr(evaluator, "finalize_metadata", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluator.evaluate(args)
    monkeypatch.setattr(evaluator, "finalize_metadata", finalize)
    monkeypatch.setattr(
        evaluator,
        "load_partition",
        lambda *args: pytest.fail("Must not reread test signals"),
    )
    args.retry_failed = True
    evaluator.evaluate(args)
    assert (
        json.loads((directory / "model_metadata.json").read_text())["test_metrics"][
            "fold"
        ]
        == 10
    )
