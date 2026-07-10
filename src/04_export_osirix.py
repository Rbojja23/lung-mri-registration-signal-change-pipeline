from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from dicom_reference import inspect_reference_series
from osirix_export import export_osirix_package
from project_paths import DEFAULT_OUTPUTS_DIR, resolve_user_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or export OsiriX-compatible DICOM review objects.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Read-only DICOM reference validation.")
    inspect_parser.add_argument("--dicom-dir", required=True)

    export_parser = subparsers.add_parser("export", help="Export RTSTRUCT or fallback review package.")
    export_parser.add_argument("--dicom-dir", required=True)
    export_parser.add_argument("--image", required=True)
    export_parser.add_argument("--mask", required=True)
    export_parser.add_argument("--individual-mask", action="append", default=[], help="name=path. May repeat.")
    export_parser.add_argument("--bbox-json")
    export_parser.add_argument("--case-id", required=True)
    export_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUTS_DIR / "osirix"))
    export_parser.add_argument("--format", choices=["auto", "rtstruct", "dicom-seg"], default="auto")
    export_parser.add_argument("--overwrite", action="store_true")
    export_parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_individual(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--individual-mask must use name=path.")
    name, path_text = value.split("=", 1)
    if not name.strip():
        raise ValueError("Individual mask name cannot be empty.")
    path = resolve_user_path(path_text, default=Path.cwd(), allow_outside=False)
    return name.strip(), path


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "inspect":
            dicom_dir = resolve_user_path(args.dicom_dir, default=Path.cwd(), allow_outside=False)
            series = inspect_reference_series(dicom_dir)
            print(json.dumps(series.safe_summary(), indent=2))
            return 0

        dicom_dir = resolve_user_path(args.dicom_dir, default=Path.cwd(), allow_outside=False)
        image = resolve_user_path(args.image, default=Path.cwd(), allow_outside=False)
        mask = resolve_user_path(args.mask, default=Path.cwd(), allow_outside=False)
        bbox_json = (
            resolve_user_path(args.bbox_json, default=Path.cwd(), allow_outside=False)
            if args.bbox_json
            else None
        )
        output_root = resolve_user_path(args.output_root, default=DEFAULT_OUTPUTS_DIR, allow_outside=False)
        individuals = [parse_individual(value) for value in args.individual_mask]
        report = export_osirix_package(
            dicom_dir=dicom_dir,
            image_path=image,
            mask_path=mask,
            individual_masks=individuals,
            bbox_json=bbox_json,
            case_id=args.case_id,
            output_root=output_root,
            export_format=args.format,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps(report, indent=2))
        if args.dry_run:
            print("Dry run: no DICOM objects or review package files were written.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
