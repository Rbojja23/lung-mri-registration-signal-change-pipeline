#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from project_paths import DEFAULT_OUTPUTS_DIR
from registration_difference_comparison import compare_registration_differences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create research-only Case 1-versus-Case 2 registration-difference figures "
            "with one pooled display window. Analytical NIfTIs are read-only."
        )
    )
    registration_root = DEFAULT_OUTPUTS_DIR / "registration"
    parser.add_argument(
        "--case-1-dir", default=str(registration_root / "registration_case_1")
    )
    parser.add_argument(
        "--case-2-dir", default=str(registration_root / "registration_case_2")
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            registration_root
            / "case_comparison"
            / "equal_window_registration_differences"
        ),
    )
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--opacity", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_registration_differences(
        case_1_dir=args.case_1_dir,
        case_2_dir=args.case_2_dir,
        output_dir=args.output_dir,
        percentile=args.percentile,
        opacity=args.opacity,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        window = result["shared_difference_window"]
        print(f"{result['analysis_type']}: {result['completion_status']}")
        print(f"Shared display limits: [{window['vmin']}, {window['vmax']}]")
        print(f"Report: {result['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
