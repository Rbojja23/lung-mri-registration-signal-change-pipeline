#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from registration_qc import compare_registration_experiments, evaluate_registration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing deedsBCV registration or compare completed QC manifests."
    )
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--experiment-dir")
    parser.add_argument("--fixed")
    parser.add_argument("--moving")
    parser.add_argument("--registered")
    parser.add_argument("--registration-manifest")
    parser.add_argument("--fixed-mask")
    parser.add_argument("--moving-mask")
    parser.add_argument("--registered-mask")
    parser.add_argument("--histogram-bins", type=int, default=64)
    parser.add_argument("--full-qc")
    parser.add_argument("--cropped-qc")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--launch-itksnap", action="store_true")
    parser.add_argument("--dry-run-launch", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        _require(args, "full_qc", "cropped_qc")
        if args.launch_itksnap or args.dry_run_launch:
            raise ValueError("ITK-SNAP launch options are not valid in comparison mode.")
        result = compare_registration_experiments(
            case_id=args.case_id,
            full_qc_path=args.full_qc,
            cropped_qc_path=args.cropped_qc,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    else:
        _require(
            args,
            "experiment",
            "experiment_dir",
            "fixed",
            "moving",
            "registered",
            "registration_manifest",
        )
        result = evaluate_registration(
            case_id=args.case_id,
            experiment=args.experiment,
            experiment_dir=args.experiment_dir,
            fixed_path=args.fixed,
            moving_path=args.moving,
            registered_path=args.registered,
            registration_manifest_path=args.registration_manifest,
            fixed_mask_path=args.fixed_mask,
            moving_mask_path=args.moving_mask,
            registered_mask_path=args.registered_mask,
            histogram_bins=args.histogram_bins,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            launch_itksnap=args.launch_itksnap,
            dry_run_launch=args.dry_run_launch,
        )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Case: {result['case_id']}")
        if args.compare:
            print(f"Runtime winner: {result['runtime_winner']}")
            print(f"Metric-wise winners: {result['metric_winners']}")
            print(result["interpretation"])
        else:
            print(f"Experiment: {result['experiment']}")
            print(f"MSE before: {result['metrics']['before']['mean_squared_error']}")
            print(f"MSE after: {result['metrics']['after']['mean_squared_error']}")
            print(f"Completion status: {result['completion_status']}")
    return 0


def _require(args, *names: str) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if not getattr(args, name)]
    if missing:
        raise ValueError("Missing required arguments: " + ", ".join(missing))


if __name__ == "__main__":
    raise SystemExit(main())
