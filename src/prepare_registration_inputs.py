#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_paths import DEFAULT_OUTPUTS_DIR
from registration_inputs import RegistrationInputPair, prepare_registration_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and prepare explicit fixed/moving deedsBCV inputs; no registration is run."
    )
    parser.add_argument("--fixed", required=True, help="Explicit fixed .nii or .nii.gz image.")
    parser.add_argument("--moving", required=True, help="Explicit moving .nii or .nii.gz image.")
    parser.add_argument("--case-id", required=True, help="Non-identifying registration case ID.")
    parser.add_argument("--experiment", default="full")
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUTS_DIR / "registration")
    )
    parser.add_argument("--role-assignment-source", default="manual")
    parser.add_argument("--roles-confirmed", action="store_true")
    parser.add_argument("--isotropy-tolerance", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pair = RegistrationInputPair(
        fixed_source=Path(args.fixed),
        moving_source=Path(args.moving),
        case_id=args.case_id,
        experiment=args.experiment,
        role_assignment_source=args.role_assignment_source,
        roles_confirmed=args.roles_confirmed,
    )
    result = prepare_registration_inputs(
        pair,
        output_root=args.output_root,
        isotropy_tolerance=args.isotropy_tolerance,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        comparison = result["geometry_comparison"]
        print(f"Case: {result['case_id']}")
        print(f"Experiment: {result['experiment']}")
        print(f"Roles confirmed: {result['role_assignment']['confirmed']}")
        print(f"Geometry classification: {comparison['classification']}")
        print(f"Gzip preparation required: {comparison['gzip_preparation_required']}")
        print(f"Resampling required: {comparison['resampling_required']}")
        print(f"Completion status: {result['completion_status']}")
        for name, path in result["working_copies"].items():
            print(f"Planned {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
