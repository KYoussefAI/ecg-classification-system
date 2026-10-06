"""Read-only official fold/label metadata audit; no waveform loading."""

import argparse
from pathlib import Path
from .data import load_metadata
from .artifacts import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/metadata_audit.json")
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Check that all required .hea/.dat files exist without reading waveforms",
    )
    args = parser.parse_args()
    parts, audit = load_metadata(args.data_dir)
    write_json(args.output, audit)
    for name, values in audit["partitions"].items():
        print(
            f"{name}: {values['records']} records, {values['patients']} patients; counts {values['class_counts']}"
        )
    print(
        f"Excluded {len(audit['excluded_unmapped_ids'])} unmapped records. Audit: {args.output}"
    )
    if args.check_files:
        missing = [
            str(args.data_dir / (name + suffix))
            for part in parts.values()
            for name in part.filename_lr
            for suffix in (".hea", ".dat")
            if not (args.data_dir / (name + suffix)).is_file()
        ]
        if missing:
            raise SystemExit(
                f"Missing {len(missing)} waveform files. First: {missing[0]}. Download records100 before training."
            )
        print("All required waveform files exist; no test signals were opened.")
