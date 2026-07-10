from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from project_paths import DEFAULT_OUTPUTS_DIR, resolve_user_path
from segmentation_qc import (
    DEFAULT_CT_WINDOW_CENTER,
    DEFAULT_CT_WINDOW_WIDTH,
    DEFAULT_MR_LOWER_PERCENTILE,
    DEFAULT_MR_UPPER_PERCENTILE,
    IndividualMask,
    run_segmentation_qc,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create TotalSegmentator lung segmentation QC figures and metrics."
    )
    parser.add_argument("--image", required=True, help="Source CT/MR NIfTI image.")
    parser.add_argument("--mask", required=True, help="Combined binary lung mask NIfTI.")
    parser.add_argument("--modality", choices=["ct", "mr"], required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--output-dir",
        help="QC output directory. Default: outputs/figures/<case-id>/",
    )
    parser.add_argument(
        "--individual-mask",
        action="append",
        default=[],
        help="Optional label=path mask. May be repeated. If label is omitted, filename stem is used.",
    )
    parser.add_argument("--show", action="store_true", help="Open figures interactively.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mr-lower-percentile",
        type=float,
        default=DEFAULT_MR_LOWER_PERCENTILE,
        help="MRI display lower percentile. Default: 1.",
    )
    parser.add_argument(
        "--mr-upper-percentile",
        type=float,
        default=DEFAULT_MR_UPPER_PERCENTILE,
        help="MRI display upper percentile. Default: 99.",
    )
    parser.add_argument(
        "--ct-window-center",
        type=float,
        default=DEFAULT_CT_WINDOW_CENTER,
        help="CT display window center. Default: -500.",
    )
    parser.add_argument(
        "--ct-window-width",
        type=float,
        default=DEFAULT_CT_WINDOW_WIDTH,
        help="CT display window width. Default: 1500.",
    )
    return parser


def parse_individual_mask(value: str) -> IndividualMask:
    if "=" in value:
        label, path_text = value.split("=", 1)
        label = label.strip()
    else:
        path_text = value
        label = Path(path_text).name
        for suffix in [".nii.gz", ".nii"]:
            if label.endswith(suffix):
                label = label[: -len(suffix)]
                break
    if not label:
        raise ValueError("Individual mask label cannot be empty.")
    path = resolve_user_path(path_text, default=Path.cwd(), allow_outside=False)
    return IndividualMask(label=label, path=path)


def main() -> int:
    args = build_parser().parse_args()
    try:
        image_path = resolve_user_path(args.image, default=Path.cwd(), allow_outside=False)
        mask_path = resolve_user_path(args.mask, default=Path.cwd(), allow_outside=False)
        output_dir = (
            resolve_user_path(args.output_dir, default=DEFAULT_OUTPUTS_DIR, allow_outside=False)
            if args.output_dir
            else DEFAULT_OUTPUTS_DIR / "figures" / args.case_id
        )
        individual_masks = [parse_individual_mask(value) for value in args.individual_mask]
        result = run_segmentation_qc(
            image_path=image_path,
            mask_path=mask_path,
            modality=args.modality,
            case_id=args.case_id,
            output_dir=output_dir,
            individual_masks=individual_masks,
            show=args.show,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            mr_lower_percentile=args.mr_lower_percentile,
            mr_upper_percentile=args.mr_upper_percentile,
            ct_window_center=args.ct_window_center,
            ct_window_width=args.ct_window_width,
        )
        print(json.dumps(result, indent=2))
        if args.dry_run:
            print("Dry run: no figures or JSON were written and no GUI was opened.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
