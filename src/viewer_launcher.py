from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import plistlib
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


@dataclass(frozen=True)
class ITKSnapInstallation:
    status: str
    bundle_path: Path | None = None
    executable_path: Path | None = None
    short_version: str | None = None
    bundle_version: str | None = None
    message: str | None = None


def detect_itksnap(
    *,
    application_paths: Sequence[str | Path] | None = None,
) -> ITKSnapInstallation:
    """Inspect only the supported system and per-user ITK-SNAP locations."""

    if platform.system() != "Darwin":
        return ITKSnapInstallation(
            status="not_found", message="ITK-SNAP detection is supported only on macOS."
        )
    candidates = application_paths or (
        Path("/Applications/ITK-SNAP.app"),
        Path.home() / "Applications" / "ITK-SNAP.app",
    )
    for candidate in candidates:
        bundle = Path(candidate).expanduser()
        if not bundle.is_dir():
            continue
        plist_path = bundle / "Contents" / "Info.plist"
        try:
            with plist_path.open("rb") as stream:
                metadata = plistlib.load(stream)
        except (OSError, plistlib.InvalidFileException) as exc:
            return ITKSnapInstallation(
                status="unhealthy",
                bundle_path=bundle,
                message=f"Could not read ITK-SNAP Info.plist: {exc}",
            )
        executable_name = metadata.get("CFBundleExecutable")
        if not isinstance(executable_name, str) or not executable_name.strip():
            return ITKSnapInstallation(
                status="unhealthy",
                bundle_path=bundle,
                short_version=metadata.get("CFBundleShortVersionString"),
                bundle_version=metadata.get("CFBundleVersion"),
                message="ITK-SNAP Info.plist does not define CFBundleExecutable.",
            )
        executable = bundle / "Contents" / "MacOS" / executable_name
        if not executable.is_file():
            message = "The ITK-SNAP bundle executable is missing."
        elif not os.access(executable, os.X_OK):
            message = "The ITK-SNAP bundle executable is not executable."
        else:
            return ITKSnapInstallation(
                status="available",
                bundle_path=bundle,
                executable_path=executable,
                short_version=metadata.get("CFBundleShortVersionString"),
                bundle_version=metadata.get("CFBundleVersion"),
            )
        return ITKSnapInstallation(
            status="unhealthy",
            bundle_path=bundle,
            executable_path=executable,
            short_version=metadata.get("CFBundleShortVersionString"),
            bundle_version=metadata.get("CFBundleVersion"),
            message=message,
        )
    return ITKSnapInstallation(status="not_found", message="ITK-SNAP was not found.")


def check_itksnap_health(
    installation: ITKSnapInstallation,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 5.0,
) -> ITKSnapInstallation:
    """Run a read-only --help probe without passing medical-image paths."""

    if installation.status != "available" or installation.executable_path is None:
        return installation
    try:
        result = runner(
            [str(installation.executable_path), "--help"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        message = f"ITK-SNAP --help timed out after {timeout:g} seconds."
    except (OSError, subprocess.SubprocessError) as exc:
        message = f"ITK-SNAP --help could not run: {exc}"
    else:
        if result.returncode == 0:
            return installation
        message = f"ITK-SNAP --help exited with status {result.returncode}."
    return ITKSnapInstallation(
        status="unhealthy",
        bundle_path=installation.bundle_path,
        executable_path=installation.executable_path,
        short_version=installation.short_version,
        bundle_version=installation.bundle_version,
        message=message,
    )


def build_itksnap_launch_plan(
    *,
    image_path: str | Path | None = None,
    segmentation_paths: Sequence[str | Path] = (),
    label_description_path: str | Path | None = None,
    overlay_paths: Sequence[str | Path] = (),
    workspace_path: str | Path | None = None,
    installation: ITKSnapInstallation | None = None,
) -> LaunchPlan:
    installation = installation or detect_itksnap()
    if installation.status != "available" or installation.executable_path is None:
        raise RuntimeError(installation.message or "ITK-SNAP is unavailable.")

    direct_values = (image_path, *segmentation_paths, label_description_path, *overlay_paths)
    if workspace_path is not None and any(value is not None for value in direct_values):
        raise ValueError("Workspace mode cannot be combined with direct image arguments.")
    executable = str(installation.executable_path)
    if workspace_path is not None:
        workspace = _existing_file(workspace_path, "workspace")
        return LaunchPlan(viewer="itksnap", commands=[[executable, "-w", str(workspace)]], warnings=[])

    if image_path is None:
        if segmentation_paths or overlay_paths or label_description_path is not None:
            raise ValueError("Segmentation, labels, and overlays require a main image.")
        raise ValueError("A main image is required in direct mode.")
    image = _existing_file(image_path, "main image")
    segmentations = [_existing_nifti(path, "segmentation") for path in segmentation_paths]
    overlays = [_existing_file(path, "overlay") for path in overlay_paths]
    labels = (
        _existing_file(label_description_path, "label description")
        if label_description_path is not None
        else None
    )
    command = [executable, "-g", str(image)]
    if segmentations:
        command.extend(["-s", *(str(path) for path in segmentations)])
    if labels is not None:
        command.extend(["-l", str(labels)])
    if overlays:
        command.extend(["-o", *(str(path) for path in overlays)])
    return LaunchPlan(viewer="itksnap", commands=[command], warnings=[])


def _existing_file(path: str | Path, role: str) -> Path:
    candidate = Path(path).expanduser().resolve(strict=False)
    if not candidate.is_file():
        raise FileNotFoundError(f"{role.capitalize()} file does not exist: {display_path(candidate)}")
    return candidate


def _existing_nifti(path: str | Path, role: str) -> Path:
    candidate = _existing_file(path, role)
    name = candidate.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise ValueError(f"{role.capitalize()} must be a .nii or .nii.gz file.")
    return candidate


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
