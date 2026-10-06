# Historical material — not the current experiment

This directory preserves the original notebook, reports, sample generators,
Streamlit app and duplicate API for historical reference. They are **not supported
runtime paths**. Their old scores are not evidence for the current model.

The original experiment used fold 10 during model/threshold selection and had
inconsistent normalization. Do not run it to train or evaluate the current system.
Some legacy interfaces also described uncalibrated outputs as confidence or made
overly definitive interpretations. Those behaviors have been removed from the
canonical application. No archived results are loaded by the current UI or API.

Canonical code: `training/`, `backend/app/`, `frontend/src/`.
Synthetic legacy samples are not patient ECGs, regardless of their old filenames
or class-like labels. The empty root npm lockfile is preserved here too.
