#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from functional_lung_analysis import analyze_functional_lung


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore fixed-space regional lung intensity changes without claiming ventilation.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--registered", required=True)
    parser.add_argument("--lung-mask", required=True)
    parser.add_argument("--left-lung")
    parser.add_argument("--right-lung")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--display-percentile", type=float, default=99.0)
    parser.add_argument("--overlay-opacity", type=float, default=0.45)
    parser.add_argument("--histogram-bins", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_functional_lung(
        fixed_path=args.fixed, registered_path=args.registered, lung_mask_path=args.lung_mask,
        left_lung_path=args.left_lung, right_lung_path=args.right_lung,
        output_dir=args.output_dir, display_percentile=args.display_percentile,
        overlay_opacity=args.overlay_opacity, histogram_bins=args.histogram_bins,
        dry_run=args.dry_run, overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['analysis_type']}: {result['completion_status']}")
        print(f"Whole-lung voxels: {result['lung_volumes']['whole_lung']['voxel_count']}")
        print(f"Report: {result['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
