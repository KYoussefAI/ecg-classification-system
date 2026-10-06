# CardioScan model card

**Model family:** ECGResNet1D-v1. **Status:** awaiting real training and final evaluation.

## Intended use

Research on PTB-XL diagnostic superclass classification, ML engineering demonstrations and de-identified educational waveform exploration. Users should understand multilabel classification and dataset limitations.

## Out-of-scope use

Clinical diagnosis, emergency triage, treatment selection, patient reassurance, autonomous screening deployment and any claim of regulatory approval. This is not a medical device.

## Training data

PTB-XL v1.0.3, 100 Hz twelve-lead, ten-second recordings. Folds 1–8 train, fold 9 validates, fold 10 is the held-out test. Exact included counts, patients and IDs are generated in `split_audit.json`; records without mapped diagnostic superclasses are excluded. Labels use diagnostic SCP-code presence and the supplied superclass mapping. Original release data were collected in historical clinical settings; the distribution does not represent every population or device.

## Method

15-sample stem, four downsampling residual stages with two kernel-7 blocks each and channels 64/128/256/512, BatchNorm/ReLU, adaptive pooling, dropout and five logits. Weighted BCE uses training-only positive weights. Per-lead mean/std are fitted on training data only. No filtering, sampling reweighting or focal loss is used by default.

Validation macro AUROC chooses the checkpoint, scheduler and stopping point. Per-class F1 thresholds are selected on fold 9 using the frozen best checkpoint. No calibration is fitted. Validation is reused for selection and thresholding; validation performance is not an unbiased final estimate.

## Evaluation procedure and metrics

The separate evaluation command opens fold 10 only after all model decisions are frozen. It uses saved normalization and thresholds without fitting. Macro/micro/per-class AUROC and average precision, F1, precision, sensitivity, specificity, support and confusion matrices are exported. Optional AUROC intervals use a patient-cluster percentile bootstrap, conditional on the fitted model.

**Real metrics: unavailable until training/evaluation is run.** Read the active artifact's `metrics.json` and `model_metadata.json`; no historical scores or synthetic-fixture scores are valid substitutes. Development/test duplicate waveform hashes and patient separation are checked. A changed model requires a new artifact; test results must not become tuning feedback.

## Input requirements

Physical millivolts, numeric `(1000, 12)`, 100 Hz, ten seconds. Order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6. Finite values are mandatory. No silent resampling, reordering, reshaping or unit conversion. Basic serving checks reject flat leads (<1e-5 mV standard deviation) and amplitudes outside ±100 mV. These limits are engineering guardrails, not validated clinical signal-quality criteria. Training/evaluation retain finite labeled dataset signals, including natural quality variation. Benchmark results are not restricted to cases passing the serving quality gate.

## Output interpretation

Fixed order: CD, HYP, MI, NORM, STTC. Five independent sigmoid model scores and validation-selected thresholds; a positive flag is a threshold comparison. Scores are not calibrated disease probabilities, and no class is a definitive clinical finding. NORM and abnormal labels can both cross thresholds; the UI warns and preserves the outputs. Absence of a positive flag does not establish a normal heart or exclude disease.

## Limitations, bias and domain shift

Potential label noise, historical acquisition bias, class imbalance, repeated-patient correlation and differences in age, sex, devices, clinical prevalence and setting. No subgroup fairness, prospective effectiveness or external transportability claim is made. The superclass task does not cover all clinically relevant ECG conditions. Synthetic demos are illustrations with no diagnostic ground truth.

## Serving and privacy

Missing, corrupt, incompatible, nonfinite or synthetic artifacts cannot produce API predictions. CPU is the default device. Artifact checks validate structure and hash consistency, not authenticity; trust the source. Guest analyses are never stored. Signed-in storage is explicit and contains result summaries and optional de-identified Case IDs, not raw signals. Authentication is a portfolio baseline; production operations need independent hardening and review. No HIPAA/GDPR compliance claim is made.

## Provenance and attribution

Artifacts record architecture, class/lead order, normalization, validation thresholds, dataset version, seed, training date, best epoch, git commit/dirty status, software and data/checkpoint hashes. See [README citations](README.md#21-dataset-citation--license) and [PTB-XL on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/), CC BY 4.0.

**Research prototype — not a medical device and not intended for clinical diagnosis or emergency decision-making.** Performance on PTB-XL does not establish clinical effectiveness.
