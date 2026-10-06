"""Metrics and validation-only threshold selection; undefined values are null."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from .config import CLASSES


def select_thresholds(targets, scores, *, fold):
    if fold != 9:
        raise ValueError("Threshold selection requires validation fold 9")
    thresholds = {}
    for i, cls in enumerate(CLASSES):
        if len(np.unique(targets[:, i])) != 2:
            raise ValueError(
                f"Validation must contain positives and negatives for {cls}"
            )
        p, r, t = precision_recall_curve(targets[:, i], scores[:, i])
        f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
        # Deterministic tie break: highest threshold among equal maxima.
        thresholds[cls] = float(t[np.flatnonzero(f1 == f1.max())[-1]])
    return thresholds


def metrics(targets, scores, thresholds):
    predictions = scores >= np.array([thresholds[c] for c in CLASSES])
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, zero_division=0
    )
    confusion = multilabel_confusion_matrix(targets, predictions)
    aucs, aps, per_class = [], [], {}
    for i, cls in enumerate(CLASSES):
        auc = (
            float(roc_auc_score(targets[:, i], scores[:, i]))
            if len(np.unique(targets[:, i])) == 2
            else None
        )
        ap = (
            float(average_precision_score(targets[:, i], scores[:, i]))
            if targets[:, i].sum()
            else None
        )
        tn, fp, fn, tp = confusion[i].ravel()
        aucs.append(auc)
        aps.append(ap)
        per_class[cls] = {
            "auroc": auc,
            "auprc": ap,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "sensitivity": float(recall[i]),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "f1": float(f1[i]),
            "support": int(support[i]),
            "threshold": thresholds[cls],
            "confusion_matrix": confusion[i].tolist(),
        }
    return {
        "record_count": len(targets),
        "macro_auroc": float(np.mean(aucs)) if None not in aucs else None,
        "micro_auroc": float(roc_auc_score(targets.ravel(), scores.ravel()))
        if len(np.unique(targets)) == 2
        else None,
        "macro_auprc": float(np.mean(aps)) if None not in aps else None,
        "micro_auprc": float(average_precision_score(targets.ravel(), scores.ravel()))
        if targets.sum()
        else None,
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(targets, predictions, average="micro", zero_division=0)
        ),
        "per_class": per_class,
        "thresholds": thresholds,
        "auprc_definition": "average_precision (non-interpolated)",
    }


def bootstrap_auroc(targets, scores, patient_ids, repeats, seed):
    """Patient-cluster bootstrap preserves correlation between repeated ECGs."""
    rng = np.random.default_rng(seed)
    patients = np.unique(patient_ids)
    groups = {p: np.flatnonzero(patient_ids == p) for p in patients}
    values = {c: [] for c in [*CLASSES, "macro"]}
    for _ in range(repeats):
        indices = np.concatenate(
            [groups[p] for p in rng.choice(patients, len(patients), replace=True)]
        )
        aucs = []
        for i, cls in enumerate(CLASSES):
            if len(np.unique(targets[indices, i])) == 2:
                auc = float(roc_auc_score(targets[indices, i], scores[indices, i]))
                values[cls].append(auc)
                aucs.append(auc)
        if len(aucs) == 5:
            values["macro"].append(float(np.mean(aucs)))
    return {
        "method": "patient_cluster_percentile",
        "repeats": repeats,
        "seed": seed,
        "intervals": {
            k: {
                "lower": float(np.quantile(v, 0.025)),
                "upper": float(np.quantile(v, 0.975)),
                "valid_repeats": len(v),
            }
            if v
            else None
            for k, v in values.items()
        },
    }
