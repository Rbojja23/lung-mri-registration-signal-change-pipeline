from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from medical_image_io import (
    image_summary_without_pixels,
    load_medical_image,
    save_summary_json,
    write_nifti,
)
from project_paths import display_path, resolve_user_path, project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load CT/MR DICOM or NIfTI images without inspecting patient metadata."
    )
    parser.add_argument("--input", required=True, help="DICOM directory or NIfTI file.")
    parser.add_argument(
        "--input-type",
        choices=["auto", "dicom", "nifti"],
        default="auto",
        help="Input type. Default: auto.",
    )
    parser.add_argument(
        "--modality",
        choices=["auto", "ct", "mr"],
        default="auto",
        help="Clinical modality. NIfTI requires ct or mr.",
    )
    parser.add_argument("--series-id", help="DICOM SeriesInstanceUID to load.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search DICOM files recursively. Default inspects only the input directory.",
    )
    parser.add_argument("--summary-json", help="Optional safe metadata JSON output path.")
    parser.add_argument("--convert-to-nifti", help="Optional .nii or .nii.gz output path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting requested output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and summarize only; do not write requested outputs.",
    )
    parser.add_argument(
        "--force-modality",
        action="store_true",
        help="Allow an explicit modality to override conflicting DICOM metadata.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        input_path = resolve_user_path(
            args.input,
            default=project_root(),
            allow_outside=False,
        )
        loaded = load_medical_image(
            input_path,
            input_type=args.input_type,
            modality=args.modality,
            series_id=args.series_id,
            recursive=args.recursive,
            force_modality=args.force_modality,
        )

        summary = image_summary_without_pixels(loaded)
        print(json.dumps(summary, indent=2))

        if args.dry_run:
            print("Dry run: no files were written.")
            return 0

        if args.summary_json:
            summary_path = resolve_user_path(
                args.summary_json,
                default=project_root(),
                allow_outside=False,
            )
            save_summary_json(loaded, summary_path, overwrite=args.overwrite)
            print(f"Wrote summary JSON: {display_path(summary_path)}")

        if args.convert_to_nifti:
            nifti_path = resolve_user_path(
                args.convert_to_nifti,
                default=project_root(),
                allow_outside=False,
            )
            write_nifti(loaded, nifti_path, overwrite=args.overwrite)
            print(f"Wrote NIfTI: {display_path(nifti_path)}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
