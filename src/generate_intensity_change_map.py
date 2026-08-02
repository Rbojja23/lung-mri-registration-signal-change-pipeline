#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from intensity_change_map import generate_intensity_change_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate research-only registered intensity-change maps.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--registered", required=True)
    parser.add_argument("--lung-mask", required=True)
    parser.add_argument("--left-lung-mask")
    parser.add_argument("--right-lung-mask")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--experiment", default="full")
    parser.add_argument("--output-root", default="outputs/registration")
    parser.add_argument("--denominator-policy", choices=("absolute", "percentile"))
    parser.add_argument("--denominator-threshold", type=float)
    parser.add_argument("--denominator-percentile", type=float)
    parser.add_argument("--skip-normalized", action="store_true")
    parser.add_argument("--display-percentile", type=float, default=99.0)
    parser.add_argument("--overlay-opacity", type=float, default=0.45)
    parser.add_argument("--histogram-bins", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_intensity_change_map(
        fixed_path=args.fixed, registered_path=args.registered, lung_mask_path=args.lung_mask,
        left_lung_mask_path=args.left_lung_mask, right_lung_mask_path=args.right_lung_mask,
        case_id=args.case_id, experiment=args.experiment, output_root=args.output_root,
        denominator_policy=args.denominator_policy, denominator_threshold=args.denominator_threshold,
        denominator_percentile=args.denominator_percentile, skip_normalized=args.skip_normalized,
        display_percentile=args.display_percentile, overlay_opacity=args.overlay_opacity,
        histogram_bins=args.histogram_bins, overwrite=args.overwrite, dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['terminology']}: {result['completion_status']}")
        print(f"Lung voxels: {result['domain_counts']['combined_lung_voxels']}")
        print(f"Invalid denominator voxels: {result['denominator_policy']['invalid_denominator_voxel_count']}")
        print(f"Manifest: {result['outputs']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
