from __future__ import annotations

import argparse
from pathlib import Path

from project_paths import ProjectPaths, display_path, project_root


def _status_line(label: str, path: Path, root: Path) -> str:
    exists = "exists" if path.exists() else "missing"
    return f"{label}: {display_path(path, root=root)} ({exists})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely report configured project data/output paths."
    )
    parser.add_argument("--data-root", help="Optional data root override.")
    parser.add_argument("--output-root", help="Optional output root override.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Accepted for clarity; this diagnostic never modifies files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_overrides(
        data_root=args.data_root,
        output_root=args.output_root,
    )
    root = project_root()

    checks = [
        ("Project root", paths.root),
        ("Data root", paths.data_dir),
        ("Raw data root", paths.raw_data_dir),
        ("Output root", paths.outputs_dir),
        ("CT case directory", paths.raw_data_path("LCTSC_offsite")),
        ("MRI case directory", paths.raw_data_path("MRI_cases")),
    ]

    print("Safe path diagnostic")
    for label, path in checks:
        print(_status_line(label, path, root))

    print("No files were created or modified.")
    print("No DICOM files or patient metadata were inspected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

