from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from project_paths import DEFAULT_OUTPUTS_DIR, display_path, resolve_user_path
from totalseg_lungs import run_lung_segmentation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local TotalSegmentator lung segmentation for CT or MR."
    )
    parser.add_argument("--input", required=True, help="Input DICOM directory or NIfTI file.")
    parser.add_argument(
        "--input-type",
        choices=["auto", "dicom", "nifti"],
        default="auto",
        help="Input type. Prefer nifti for this stage.",
    )
    parser.add_argument(
        "--modality",
        choices=["ct", "mr"],
        required=True,
        help="Clinical modality. CT uses task total; MR uses task total_mr.",
    )
    parser.add_argument("--case-id", required=True, help="Non-identifying output case ID.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUTS_DIR / "totalsegmentator"),
        help="Output root for generated TotalSegmentator results.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device. auto resolves to CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator fast mode.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the generated output directory for this case.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the planned command without running inference.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        input_path = resolve_user_path(
            args.input,
            default=Path.cwd(),
            allow_outside=False,
        )
        output_root = resolve_user_path(
            args.output_root,
            default=DEFAULT_OUTPUTS_DIR / "totalsegmentator",
            allow_outside=False,
        )
        result = run_lung_segmentation(
            input_path=input_path,
            input_type=args.input_type,
            modality=args.modality,
            case_id=args.case_id,
            output_root=output_root,
            device=args.device,
            fast=args.fast,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
        if args.dry_run:
            print("Dry run: TotalSegmentator was not executed and no output directories were created.")
        else:
            print(f"Completed segmentation for case: {args.case_id}")
            print(f"Output root: {display_path(output_root)}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
