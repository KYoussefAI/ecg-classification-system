"""Install the pinned PyTorch CPU wheel first, then the selected requirements.

Keeps CI and the serving image free of unnecessary CUDA runtime downloads.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    args = parser.parse_args()
    text = args.requirements.read_text(encoding="utf-8")
    match = re.search(r"^torch==([^\s;]+)$", text, re.MULTILINE)
    if not match:
        raise ValueError("Requirements must pin exactly one PyTorch version")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "torch==" + match.group(1),
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            str(args.requirements),
        ],
        check=True,
    )
