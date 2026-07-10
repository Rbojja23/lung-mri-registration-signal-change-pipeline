from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from bbox_qc import run_bbox_qc
from project_paths import DEFAULT_OUTPUTS_DIR, resolve_user_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create reproducible lung bbox QC figures.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--bbox-json", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--modality", choices=["ct", "mr"], required=True)
    parser.add_argument("--crop-image")
    parser.add_argument("--crop-mask")
    parser.add_argument("--output-dir")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        image = resolve_user_path(args.image, default=Path.cwd(), allow_outside=False)
        mask = resolve_user_path(args.mask, default=Path.cwd(), allow_outside=False)
        bbox_json = resolve_user_path(args.bbox_json, default=Path.cwd(), allow_outside=False)
        crop_image = (
            resolve_user_path(args.crop_image, default=Path.cwd(), allow_outside=False)
            if args.crop_image
            else None
        )
        crop_mask = (
            resolve_user_path(args.crop_mask, default=Path.cwd(), allow_outside=False)
            if args.crop_mask
            else None
        )
        output_dir = (
            resolve_user_path(args.output_dir, default=DEFAULT_OUTPUTS_DIR, allow_outside=False)
            if args.output_dir
            else None
        )
        result = run_bbox_qc(
            image_path=image,
            mask_path=mask,
            bbox_json_path=bbox_json,
            case_id=args.case_id,
            modality=args.modality,
            crop_image_path=crop_image,
            crop_mask_path=crop_mask,
            output_dir=output_dir,
            show=args.show,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
        if args.dry_run:
            print("Dry run: no bbox QC files were written and no GUI was opened.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
