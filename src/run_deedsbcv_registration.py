#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from deedsbcv_runner import run_deedsbcv_registration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run default full-volume deedsBCV affine and deformable registration."
    )
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--experiment", default="full")
    parser.add_argument("--bin-dir")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_deedsbcv_registration(
        prepared_dir=args.prepared_dir,
        case_id=args.case_id,
        experiment=args.experiment,
        bin_dir=args.bin_dir,
        timeout_seconds=args.timeout_seconds,
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Case: {result['case_id']}")
        print(f"Experiment: {result['experiment']}")
        print(f"Completion status: {result['completion_status']}")
        print("Affine command:", " ".join(result["commands"]["affine"]))
        print("Deformable command:", " ".join(result["commands"]["deformable"]))
        for name, path in result["outputs"].items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
