#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from registration_metric_analysis import analyze_registration_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze metrics for an existing registration without running registration.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--registered", required=True)
    parser.add_argument("--lung-mask")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--histogram-bins", type=int, default=100)
    parser.add_argument("--mi-bins", type=int, default=64)
    parser.add_argument("--display-percentile", type=float, default=99.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_registration_metrics(
        fixed_path=args.fixed, moving_path=args.moving, registered_path=args.registered,
        lung_mask_path=args.lung_mask, output_dir=args.output_dir,
        histogram_bins=args.histogram_bins, mi_bins=args.mi_bins,
        display_percentile=args.display_percentile, overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Analysis: {result['completion_status']}")
        for name, values in result["metrics"]["whole_image"]["metrics"].items():
            print(f"{name}: before={values['before']} after={values['after']} improvement={values['absolute_improvement']}")
        print(f"Report: {result['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
