# Exploratory notebooks

Notebooks are optional EDA/walkthroughs only. Use `python -m training.train` and
`python -m training.evaluate` for reproducible experiments. The original notebook
and rendered exports live under `archive/legacy/` with methodology warnings.

For new EDA, load metadata with `training.data.load_metadata`, inspect train folds,
and use `training.data.read_signal`. Do not inspect fold-10 outcomes until the
experiment is frozen. Never estimate normalization or thresholds in a notebook.
