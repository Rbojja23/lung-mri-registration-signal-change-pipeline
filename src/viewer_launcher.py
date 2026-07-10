from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
from typing import Callable, Sequence

from project_paths import display_path


VIEWER_APPS = {
    "osirix-md": "OsiriX MD",
    "osirix-lite": "OsiriX Lite",
    "osirix": "OsiriX",
    "horos": "Horos",
    "slicer": "3D Slicer",
    "finder": "Finder",
}


@dataclass(frozen=True)
class LaunchPlan:
    viewer: str
    commands: list[list[str]]
    warnings: list[str]


def detect_viewers() -> dict[str, bool]:
    if platform.system() != "Darwin":
        return {key: False for key in VIEWER_APPS}
    found = {}
    for key, app_name in VIEWER_APPS.items():
        found[key] = (Path("/Applications") / f"{app_name}.app").exists()
    return found


def choose_viewer(preferred: str | None = None) -> str | None:
    found = detect_viewers()
    if preferred:
        key = preferred.lower()
        if key not in VIEWER_APPS:
            raise ValueError(f"Unknown viewer: {preferred}")
        return key if found.get(key) else None
    for key in ["osirix-md", "osirix-lite", "osirix", "horos"]:
        if found.get(key):
            return key
    return None


def build_launch_plan(
    *,
    dicom_dir: str | Path | None = None,
    rtstruct_path: str | Path | None = None,
    fallback_dir: str | Path | None = None,
    nifti_paths: Sequence[str | Path] = (),
    viewer: str | None = None,
) -> LaunchPlan:
    if platform.system() != "Darwin":
        raise RuntimeError("Viewer launching is supported only on macOS.")
    selected = choose_viewer(viewer)
    warnings: list[str] = []
    commands: list[list[str]] = []
    if selected is None:
        raise RuntimeError("No supported OsiriX/Horos viewer was found.")
    app_name = VIEWER_APPS[selected]
    if dicom_dir:
        commands.append(["open", "-a", app_name, str(dicom_dir)])
    if rtstruct_path:
        commands.append(["open", "-a", app_name, str(rtstruct_path)])
        warnings.append("First-time manual import/association may still be required in OsiriX.")
    if fallback_dir:
        commands.append(["open", str(fallback_dir)])
        warnings.append("Fallback package is not a registered OsiriX DICOM overlay.")
    if nifti_paths and detect_viewers().get("slicer"):
        for path in nifti_paths:
            commands.append(["open", "-a", VIEWER_APPS["slicer"], str(path)])
    elif nifti_paths:
        warnings.append("3D Slicer not detected; NIfTI files were not launched.")
    return LaunchPlan(viewer=selected, commands=commands, warnings=warnings)


def launch_plan(
    plan: LaunchPlan,
    *,
    dry_run: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    if dry_run:
        return {
            "viewer": plan.viewer,
            "commands": [[_safe_arg(arg) for arg in command] for command in plan.commands],
            "warnings": plan.warnings,
            "status": "dry-run",
        }
    for command in plan.commands:
        runner(command, shell=False, check=True)
    return {
        "viewer": plan.viewer,
        "commands": [[_safe_arg(arg) for arg in command] for command in plan.commands],
        "warnings": plan.warnings,
        "status": "launched",
    }


def _safe_arg(arg: str) -> str:
    return display_path(arg) if "/" in arg else arg
