# Verification record — 6 October 2026

These checks validate the implementation. They are not evidence of ECG classifier
performance or clinical effectiveness. No full PTB-XL training run was performed.

| Check | Result |
|---|---|
| Python ML/API suite | 43 passed; no warnings on the final run |
| Synthetic WFDB training smoke | Two epochs, best checkpoint, stored normalization and validation thresholds generated |
| Synthetic held-out evaluation | Metrics, patient-bootstrap intervals and all five requested plots generated |
| Interrupted training | Resume reproduces uninterrupted model weights and thresholds on the deterministic fixture |
| Interrupted evaluation | Metadata finalization recovers without recomputing test predictions |
| JSON / CSV / NPY | Validation and prediction routes exercised with synthetic fixtures |
| Live API without a model | Health 503, prediction 503, metadata 200, upload validation 200 |
| Valid test checkpoint | Successful health/inference checked inside isolated FastAPI TestClient fixtures |
| Anonymous history | No persistence; authenticated storage requires explicit opt-in |
| History ownership | Cross-account access/deletion blocked |
| Frontend | Clean `npm ci`, ESLint and production build pass |
| Browser | Desktop and 390px mobile Playwright checks pass, including reduced motion and missing-model state |
| Dependency checks | npm audit: 0 vulnerabilities; Python installed dependency constraints compatible |
| Repository | `git diff --check` and Python compile checks pass |

## Real metadata audit

Official PTB-XL v1.0.3 metadata downloaded from PhysioNet; no real waveform data
were opened during preparation. Canonical superclass mapping and patient/fold
checks yielded:

| Partition | Records | Patients |
|---|---:|---:|
| Train, folds 1–8 | 17,084 | 14,823 |
| Validation, fold 9 | 2,146 | 1,917 |
| Held-out test, fold 10 | 2,158 | 1,877 |

411 records without mapped diagnostic superclasses were excluded. The generated
local audit is `artifacts/metadata_audit.json`. These are cohort counts, not
performance metrics. The dataset's `records100/` files still need downloading.

## CPU engineering check

The default network has 8,739,973 parameters. On this workspace, batch size one,
four CPU threads, `torch.inference_mode()`, three warmups and twenty measured
forward passes averaged approximately **7.95 ms** per forward pass. This excludes
HTTP, parsing and preprocessing, uses a synthetic tensor, and is not a deployment
latency guarantee. Python 3.12.15 / PyTorch 2.14.1+cpu were used.

## Practical limits

- CUDA was unavailable; CUDA/AMP code was not exercised on GPU hardware.
- Docker was unavailable; Dockerfiles and Compose files were reviewed, but images
  were not built or run locally. CI workflow was added but not executed on GitHub.
- Full real training, fold-10 performance, calibration, external validation and
  clinical validation remain unperformed. Calibration is explicitly `none`.
- Browser WFDB pair upload is deferred. WFDB training ingestion is implemented.
- No repository push, remote publication or deployment was performed.

## Repeat checks

From the repository root with the Python 3.12 environment selected:

```powershell
python -m pytest -q
python -m scripts.smoke --output artifacts/another-smoke-run
python -m training.audit --data-dir data/ptb-xl/1.0.3 --check-files
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run lint
npm.cmd --prefix frontend run build
```

The file preflight intentionally reports missing waveforms until `records100` is
installed. Browser checks require the backend running without a model; in
`frontend/`, set `$env:PW_CHANNEL='msedge'` on Windows or install Playwright
Chromium, then run `npx.cmd playwright test`.
