#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from case_comparison import compare_registration_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two completed MRI registration cases from JSON reports only."
    )
    parser.add_argument("--case-1-dir", required=True)
    parser.add_argument("--case-2-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_registration_cases(
        case_1_dir=args.case_1_dir,
        case_2_dir=args.case_2_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['analysis_type']}: {result['completion_status']}")
        print(f"Compatible methods: {result['compatibility_validation']['core_method_compatible']}")
        print(f"Normalized comparison: {result['epsilon_normalized_comparison']['available']}")
        print(f"Report: {result['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
