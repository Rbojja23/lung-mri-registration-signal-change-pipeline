#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from fractional_proxy_rgb_secondary_capture import (
    DEFAULT_OPACITY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT,
    export_fractional_proxy_rgb_secondary_capture,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export six Horos-compatible, research-only RGB DICOM Secondary Capture "
            "series for native, mean, and Gaussian fractional MRI signal-change proxies."
        )
    )
    parser.add_argument("--comparison-report", default=str(DEFAULT_REPORT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--opacity", type=float, default=DEFAULT_OPACITY)
    parser.add_argument("--no-contour", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_fractional_proxy_rgb_secondary_capture(
            comparison_report=args.comparison_report,
            output_dir=args.output_dir,
            opacity=args.opacity,
            draw_contour=not args.no_contour,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        limits = result["display"]["shared_signed_limits"]
        print(f"{result['analysis_type']}: {result['completion_status']}")
        print(f"Shared signed limits: [{limits['vmin']}, {limits['vmax']}]")
        print(f"Output: {result['outputs']['directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
