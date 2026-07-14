from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from itksnap_labels import prepare_itksnap_products
from project_paths import DEFAULT_OUTPUTS_DIR, resolve_user_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ITK-SNAP lung and bbox label products.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--left-lung", required=True)
    parser.add_argument("--right-lung", required=True)
    parser.add_argument("--bbox-json", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--modality", choices=["mr", "ct"], required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUTS_DIR / "itksnap"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_itksnap_products(
            image_path=resolve_user_path(args.image, default=Path.cwd(), allow_outside=False),
            left_lung_path=resolve_user_path(
                args.left_lung, default=Path.cwd(), allow_outside=False
            ),
            right_lung_path=resolve_user_path(
                args.right_lung, default=Path.cwd(), allow_outside=False
            ),
            bbox_json_path=resolve_user_path(
                args.bbox_json, default=Path.cwd(), allow_outside=False
            ),
            case_id=args.case_id,
            modality=args.modality,
            output_root=resolve_user_path(
                args.output_root, default=DEFAULT_OUTPUTS_DIR / "itksnap", allow_outside=False
            ),
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
