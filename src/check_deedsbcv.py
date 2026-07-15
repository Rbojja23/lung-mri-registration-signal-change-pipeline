#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from deedsbcv_tools import (
    check_deedsbcv_health,
    discover_deedsbcv,
    installation_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover deedsBCV and optionally run safe help-only health probes."
    )
    parser.add_argument("--bin-dir", help="Explicit directory containing deedsBCV binaries.")
    parser.add_argument("--health-check", action="store_true", help="Run help-only binary probes.")
    parser.add_argument("--json", action="store_true", help="Print a JSON report.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and report only; suppress health-check subprocesses.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    installation = discover_deedsbcv(args.bin_dir)
    if args.health_check and not args.dry_run:
        installation = check_deedsbcv_health(installation)
    summary = installation_summary(installation)
    summary["dry_run"] = args.dry_run
    summary["health_check_requested"] = args.health_check
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"deedsBCV status: {summary['status']}")
        print(f"Discovery source: {summary['source'] or 'none'}")
        patch = summary["patch"]
        print(f"Pinned commit expected: {patch['pinned_commit_expected']}")
        print(f"Patch expected: {patch['patch_expected']}")
        print(f"Patch SHA-256: {patch['patch_sha256'] or 'unavailable'}")
        print(f"Patch status: {patch['status']} - {patch['message']}")
        for name, item in summary["binaries"].items():
            state = "found" if item["found"] else "missing"
            requirement = "required" if item["required"] else "optional"
            print(f"{name}: {state} ({requirement})")
        for name, item in summary["health"].items():
            print(f"{name} health: {item['status']} - {item['message']}")
        if args.dry_run and args.health_check:
            print("Dry run: health-check subprocesses were not executed.")
    return 0 if installation.status == "available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
