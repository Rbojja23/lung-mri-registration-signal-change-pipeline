from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from bounding_box import compute_bbox_and_crops
from project_paths import DEFAULT_OUTPUTS_DIR, resolve_user_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a lung bbox from an existing binary mask and crop image/masks."
    )
    parser.add_argument("--image", required=True, help="Source image NIfTI.")
    parser.add_argument("--mask", required=True, help="Combined binary lung mask NIfTI.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--padding-mm",
        type=float,
        nargs="+",
        help="Physical padding in mm. Provide one value or z y x values.",
    )
    parser.add_argument(
        "--padding-voxels",
        type=int,
        nargs="+",
        help="Voxel padding. Provide one value or z y x values.",
    )
    parser.add_argument(
        "--crop-individual-masks",
        action="store_true",
        help="Crop masks supplied by --individual-mask.",
    )
    parser.add_argument(
        "--individual-mask",
        action="append",
        default=[],
        help="Optional individual mask path. May be repeated.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUTS_DIR),
        help="Output root. Default: outputs/.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        image_path = resolve_user_path(args.image, default=Path.cwd(), allow_outside=False)
        mask_path = resolve_user_path(args.mask, default=Path.cwd(), allow_outside=False)
        output_root = resolve_user_path(args.output_root, default=DEFAULT_OUTPUTS_DIR, allow_outside=False)
        individual_masks = []
        if args.crop_individual_masks:
            individual_masks = [
                resolve_user_path(path, default=Path.cwd(), allow_outside=False)
                for path in args.individual_mask
            ]
        elif args.individual_mask:
            raise ValueError("--individual-mask requires --crop-individual-masks.")

        result = compute_bbox_and_crops(
            image_path=image_path,
            mask_path=mask_path,
            case_id=args.case_id,
            padding_mm=args.padding_mm,
            padding_voxels=args.padding_voxels,
            individual_mask_paths=individual_masks,
            output_root=output_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
        if args.dry_run:
            print("Dry run: no files or directories were written.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
