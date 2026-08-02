#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from normalized_intensity_map import generate_normalized_intensity_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an Exploratory normalized registered intensity-change map.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--registered", required=True)
    parser.add_argument("--lung-mask", required=True)
    parser.add_argument("--left-lung")
    parser.add_argument("--right-lung")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--denominator-policy", choices=("absolute",), required=True)
    parser.add_argument("--denominator-threshold", type=float, required=True)
    parser.add_argument("--display-percentile", type=float, default=99.0)
    parser.add_argument("--overlay-opacity", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_normalized_intensity_map(
        fixed_path=args.fixed, registered_path=args.registered, lung_mask_path=args.lung_mask,
        left_lung_path=args.left_lung, right_lung_path=args.right_lung,
        output_dir=args.output_dir, denominator_policy=args.denominator_policy,
        denominator_threshold=args.denominator_threshold, display_percentile=args.display_percentile,
        overlay_opacity=args.overlay_opacity, dry_run=args.dry_run, overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['analysis_type']}: {result['completion_status']}")
        print(f"Valid voxels: {result['denominator_policy']['valid_voxel_count']}")
        print(f"Invalid denominator voxels: {result['denominator_policy']['invalid_denominator_count']}")
        print(f"Report: {result['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
