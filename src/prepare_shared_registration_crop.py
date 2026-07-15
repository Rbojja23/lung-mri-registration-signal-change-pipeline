#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_paths import DEFAULT_OUTPUTS_DIR
from registration_crop import SharedCropRequest, prepare_shared_registration_crop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply one fixed-image bbox to same-grid fixed/moving inputs; no registration is run."
    )
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--bbox-json", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--source-experiment", default="full")
    parser.add_argument("--experiment", default="cropped")
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUTS_DIR / "registration")
    )
    parser.add_argument(
        "--role-assignment-source", default="inferred-from-repeated-bimg1"
    )
    parser.add_argument("--roles-confirmed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_shared_registration_crop(
        SharedCropRequest(
            fixed=Path(args.fixed),
            moving=Path(args.moving),
            bbox_json=Path(args.bbox_json),
            case_id=args.case_id,
            source_experiment=args.source_experiment,
            experiment=args.experiment,
            role_assignment_source=args.role_assignment_source,
            roles_confirmed=args.roles_confirmed,
        ),
        output_root=args.output_root,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        experiment = result.get("experiment", result.get("output_experiment"))
        start = result.get("start_index_xyz", result.get("simpleitk_start_index_xyz"))
        size = result.get("size_xyz", result.get("simpleitk_size_xyz"))
        fixed_geometry = result.get("fixed_cropped_geometry", {})
        origin = result.get(
            "expected_origin_xyz_mm", fixed_geometry.get("origin_xyz_mm")
        )
        print(f"Case: {result['case_id']}")
        print(f"Experiment: {experiment}")
        print(f"BBox z,y,x: {result['bbox_half_open_zyx']}")
        print(f"SimpleITK start x,y,z: {start}")
        print(f"Crop size x,y,z: {size}")
        print(f"Expected origin x,y,z mm: {origin}")
        print(f"Boundary touch: {result['boundary_touch']}")
        for name, path in result.get("planned_outputs", {}).items():
            print(f"Planned {name}: {path}")
        print(f"Completion status: {result['completion_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
