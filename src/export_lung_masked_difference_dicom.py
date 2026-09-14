#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from lung_masked_difference_visualization import DEFAULT_OUTPUT_DIR
from lung_masked_rgb_secondary_capture_export import export_lung_masked_rgb_secondary_capture
from project_paths import DEFAULT_OUTPUTS_DIR


def build_parser() -> argparse.ArgumentParser:
    registration = DEFAULT_OUTPUTS_DIR / "registration"
    comparison = registration / "case_comparison" / "equal_window_registration_differences"
    masks = DEFAULT_OUTPUTS_DIR / "totalsegmentator"
    parser = argparse.ArgumentParser(
        description="Export research-only lung-masked turbo differences as RGB DICOM Secondary Capture."
    )
    parser.add_argument("--case-1-dir", default=str(registration / "registration_case_1"))
    parser.add_argument("--case-2-dir", default=str(registration / "registration_case_2"))
    parser.add_argument("--case-1-lung-mask", default=str(masks / "registration_case_1_fixed/lungs/combined_lungs.nii.gz"))
    parser.add_argument("--case-2-lung-mask", default=str(masks / "registration_case_2_fixed/lungs/combined_lungs.nii.gz"))
    parser.add_argument("--shared-window-report", default=str(comparison / "equal_window_registration_difference_report.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "rgb_dicom_secondary_capture"))
    parser.add_argument("--opacity", type=float, default=0.45)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_lung_masked_rgb_secondary_capture(
            case_1_dir=args.case_1_dir,
            case_2_dir=args.case_2_dir,
            case_1_lung_mask=args.case_1_lung_mask,
            case_2_lung_mask=args.case_2_lung_mask,
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
