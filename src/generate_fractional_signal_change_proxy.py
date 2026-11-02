#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fractional_signal_change_proxy import (
    DEFAULT_GAUSSIAN_SIGMA_MM,
    DEFAULT_GAUSSIAN_TRUNCATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRIOR_PROXY_DIR,
    generate_fractional_signal_change_comparison,
)
from project_paths import DEFAULT_OUTPUTS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a research-only registered-space fractional MRI signal-change "
            "proxy; this is not validated fractional ventilation."
        )
    )
    parser.add_argument(
        "--case-1-dir",
        default=str(DEFAULT_OUTPUTS_DIR / "registration" / "registration_case_1"),
    )
    parser.add_argument(
        "--case-2-dir",
        default=str(DEFAULT_OUTPUTS_DIR / "registration" / "registration_case_2"),
    )
    parser.add_argument(
        "--case-1-lung-mask",
        default=str(DEFAULT_OUTPUTS_DIR / "totalsegmentator" / "registration_case_1_fixed" / "lungs" / "combined_lungs.nii.gz"),
    )
    parser.add_argument(
        "--case-2-lung-mask",
        default=str(DEFAULT_OUTPUTS_DIR / "totalsegmentator" / "registration_case_2_fixed" / "lungs" / "combined_lungs.nii.gz"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument("--denominator-threshold", type=float)
    thresholds.add_argument(
        "--denominator-threshold-fraction",
        type=float,
        default=0.01,
        help="Fraction of each case's median abs registered-moving lung signal (default: 0.01).",
    )
    parser.add_argument("--display-percentile", type=float, default=99.0)
    parser.add_argument("--overlay-opacity", type=float, default=0.62)
    parser.add_argument(
        "--gaussian-sigma-mm",
        type=float,
        default=DEFAULT_GAUSSIAN_SIGMA_MM,
        help=(
            "Isotropic Gaussian sigma in physical millimeters "
            f"(default: {DEFAULT_GAUSSIAN_SIGMA_MM})."
        ),
    )
    parser.add_argument(
        "--gaussian-truncate",
        type=float,
        default=DEFAULT_GAUSSIAN_TRUNCATE,
        help="Kernel radius as a multiple of sigma when no explicit radius is supplied.",
    )
    parser.add_argument(
        "--gaussian-support-radius-mm",
        type=float,
        help="Optional explicit physical support radius; overrides truncation for radius selection.",
    )
    parser.add_argument(
        "--prior-proxy-dir",
        default=str(DEFAULT_PRIOR_PROXY_DIR),
        help="Completed native/7x7x7 proxy directory protected and referenced by this experiment.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fraction = args.denominator_threshold_fraction
    if args.denominator_threshold is not None:
        fraction = None
    result = generate_fractional_signal_change_comparison(
        case_1_dir=args.case_1_dir,
        case_2_dir=args.case_2_dir,
        case_1_lung_mask=args.case_1_lung_mask,
        case_2_lung_mask=args.case_2_lung_mask,
        output_dir=args.output_dir,
        denominator_threshold=args.denominator_threshold,
        denominator_threshold_fraction=fraction,
        display_percentile=args.display_percentile,
        overlay_opacity=args.overlay_opacity,
        gaussian_sigma_mm=args.gaussian_sigma_mm,
        gaussian_truncate=args.gaussian_truncate,
        gaussian_support_radius_mm=args.gaussian_support_radius_mm,
        prior_proxy_dir=args.prior_proxy_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        window = result["display"]["shared_symmetric_window"]
        print(f"{result['analysis_type']}: {result['completion_status']}")
        print(f"Shared signed limits: [{window['vmin']}, {window['vmax']}]")
        print(f"Report: {result['outputs']['figures']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
