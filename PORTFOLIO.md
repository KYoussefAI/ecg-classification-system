# Publication and recorded demo

## GitHub About

Description: **End-to-end 12-lead ECG classification system using PyTorch, PTB-XL, FastAPI and React.**

Topics: `ecg`, `deep-learning`, `pytorch`, `fastapi`, `react`, `ptb-xl`, `medical-ai`, `healthcare-ai`, `time-series`, `mlops`.

Use a 1280×640 social preview crop of the waveform workstation, with the title
“CardioScan · 12-lead ECG research workspace”. Keep “Synthetic demo signal” visible
if using the built-in preview. Avoid performance badges until real metrics exist.
Screenshot path: `docs/images/workstation.png`; optional recording: `docs/demo.gif`.

## Before recording

- Commit the reviewed implementation, obtain PTB-XL v1.0.3 and run the documented training command.
- Freeze all decisions, evaluate fold 10 once, then point `MODEL_DIR` at that entire run directory and restart the backend.
- Check `/health`: `model_loaded=true`, expected version, `demo_mode=false`.
- Use a de-identified exported recording with permitted attribution, or clearly label the synthetic demo. Never display identifying data.
- Ensure `/api/model` reports the active artifact's metrics; if unevaluated, retain “Awaiting final evaluation”.

## Suggested 90-second sequence

1. Home: explain the research purpose and five independent superclasses.
2. Workstation: load a signal; show all leads, time/mV axes and quality checks.
3. Run classification; show all five scores and each threshold. Explain NORM conflicts if present.
4. Performance: show actual fold-10 results and the 1–8 / 9 / 10 protocol. Do not infer clinical effectiveness.
5. Optional account: explicitly save a de-identified Case ID; show owner-only history and delete the demo result.
6. Close on the architecture and reproducibility commands. Keep the research disclaimer visible.

If real training is not yet complete, record the waveform and fail-closed state as
an engineering preview. Do not enable a random model or label smoke metrics as results.

## Publication language

“Built a reproducible ECG classification research prototype with patient-separated
PTB-XL folds, validation-only thresholds, shared preprocessing, artifact-checked
inference and a twelve-lead waveform workstation.” Add actual measured results
only after evaluation, with dataset version, split, sample count and limitations.
