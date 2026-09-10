#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from project_paths import DEFAULT_OUTPUTS_DIR
from rgb_secondary_capture_export import export_rgb_secondary_capture


def build_parser() -> argparse.ArgumentParser:
    registration = DEFAULT_OUTPUTS_DIR / "registration"
    comparison = registration / "case_comparison" / "equal_window_registration_differences"
    parser = argparse.ArgumentParser(
        description=(
            "Export anonymized, research-only RGB DICOM Secondary Capture series from the "
            "post-registration shared-window turbo visualization."
        )
    )
    parser.add_argument("--case-1-fixed", default=str(registration / "registration_case_1/full/preprocessed/fixed.nii.gz"))
    parser.add_argument("--case-1-registered", default=str(registration / "registration_case_1/full/deformable/moving_registered.nii.gz"))
    parser.add_argument("--case-2-fixed", default=str(registration / "registration_case_2/full/preprocessed/fixed.nii.gz"))
    parser.add_argument("--case-2-registered", default=str(registration / "registration_case_2/full/deformable/moving_registered.nii.gz"))
    parser.add_argument("--shared-window-report", default=str(comparison / "equal_window_registration_difference_report.json"))
    parser.add_argument("--output-dir", default=str(comparison / "rgb_dicom_secondary_capture"))
    parser.add_argument("--opacity", type=float, default=0.45)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_rgb_secondary_capture(
            case_1_fixed=args.case_1_fixed,
            case_1_registered=args.case_1_registered,
            case_2_fixed=args.case_2_fixed,
            case_2_registered=args.case_2_registered,
            shared_window_report=args.shared_window_report,
            output_dir=args.output_dir,
            opacity=args.opacity,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        limits = result["display"]["shared_difference_limits"]
        print(f"{result['analysis_type']}: {result['completion_status']}")
        print(f"Shared limits: [{limits['vmin']}, {limits['vmax']}]")
        print(f"Output: {result['outputs']['directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
