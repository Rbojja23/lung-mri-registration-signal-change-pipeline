from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any, Callable, Sequence

import numpy as np
import SimpleITK as sitk

from medical_image_io import (
    LoadedMedicalImage,
    compare_image_geometry,
    load_medical_image,
    normalize_modality,
    summarize_geometry,
)
from project_paths import DEFAULT_RAW_DATA_DIR, display_path, project_root
from version import PIPELINE_VERSION


TOTALSEG_EXECUTABLE = "TotalSegmentator"

MODALITY_TASKS = {
    "CT": "total",
    "MR": "total_mr",
}

CT_LUNG_LABELS = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
)

MR_LUNG_LABELS = (
    "lung_left",
    "lung_right",
)

LUNG_LABELS_BY_MODALITY = {
    "CT": CT_LUNG_LABELS,
    "MR": MR_LUNG_LABELS,
}


@dataclass(frozen=True)
class DeviceResolution:
    requested: str
    resolved: str
    totalseg_arg: str


@dataclass(frozen=True)
class SegmentationPlan:
    input_path: Path
    modality: str
    case_id: str
    output_root: Path
    case_dir: Path
    raw_dir: Path
    lungs_dir: Path
    manifest_path: Path
    combined_mask_path: Path
    task: str
    device: DeviceResolution
    fast: bool
    command: list[str]
    totalseg_version: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaskStats:
    voxel_count: int
    volume_mm3: float
    volume_ml: float


def task_for_modality(modality: str) -> str:
    normalized = normalize_modality(modality, allow_auto=False)
    return MODALITY_TASKS[normalized]


def totalseg_version() -> str:
    try:
        return importlib.metadata.version("TotalSegmentator")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def ensure_totalseg_available() -> str:
    executable = shutil.which(TOTALSEG_EXECUTABLE)
    if executable is None:
        raise RuntimeError(
            "TotalSegmentator executable was not found. Install TotalSegmentator "
            "in the active environment before running segmentation."
        )
    return executable


def resolve_device(device: str) -> DeviceResolution:
    requested = device.lower()
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("Device must be one of auto, cpu, mps, or cuda.")

    if requested == "cpu":
        return DeviceResolution(requested="cpu", resolved="cpu", totalseg_arg="cpu")
    if requested == "cuda":
        if not _cuda_available():
            raise RuntimeError("Requested CUDA device is not available.")
        return DeviceResolution(requested="cuda", resolved="cuda", totalseg_arg="gpu")
    if requested == "mps":
        if not _mps_available():
            raise RuntimeError("Requested MPS device is not available.")
        return DeviceResolution(requested="mps", resolved="mps", totalseg_arg="mps")

    if _cuda_available():
        return DeviceResolution(requested="auto", resolved="cuda", totalseg_arg="gpu")
    if _mps_available():
        return DeviceResolution(requested="auto", resolved="mps", totalseg_arg="mps")
    return DeviceResolution(requested="auto", resolved="cpu", totalseg_arg="cpu")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        import torch

        return bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
    except Exception:
        return False


def build_totalseg_command(
    *,
    executable: str,
    input_path: Path,
    raw_dir: Path,
    task: str,
    device: DeviceResolution,
    fast: bool,
) -> list[str]:
    # Use a list argv and shell=False so paths are passed as data, not shell text.
    command = [
        executable,
        "-i",
        str(input_path),
        "-o",
        str(raw_dir),
        "-ta",
        task,
        "-ot",
        "nifti",
        "-d",
        device.totalseg_arg,
    ]
    if fast:
        command.append("--fast")
    return command


def make_plan(
    *,
    input_path: str | Path,
    modality: str,
    case_id: str,
    output_root: str | Path,
    device: str,
    fast: bool,
) -> SegmentationPlan:
    normalized_modality = normalize_modality(modality, allow_auto=False)
    resolved_input = Path(input_path).expanduser().resolve(strict=False)
    resolved_output_root = Path(output_root).expanduser().resolve(strict=False)
    _reject_raw_output(resolved_output_root)

    case_dir = resolved_output_root / case_id
    raw_dir = case_dir / "raw"
    lungs_dir = case_dir / "lungs"
    manifest_path = case_dir / "manifest.json"
    combined_mask_path = lungs_dir / "combined_lungs.nii.gz"
    task = task_for_modality(normalized_modality)
    resolved_device = resolve_device(device)
    executable = ensure_totalseg_available()
    version = totalseg_version()

    command = build_totalseg_command(
        executable=executable,
        input_path=resolved_input,
        raw_dir=raw_dir,
        task=task,
        device=resolved_device,
        fast=fast,
    )
    warnings = (
        "TotalSegmentator runs locally. On first use it may download model weights.",
    )

    return SegmentationPlan(
        input_path=resolved_input,
        modality=normalized_modality,
        case_id=case_id,
        output_root=resolved_output_root,
        case_dir=case_dir,
        raw_dir=raw_dir,
        lungs_dir=lungs_dir,
        manifest_path=manifest_path,
        combined_mask_path=combined_mask_path,
        task=task,
        device=resolved_device,
        fast=fast,
        command=command,
        totalseg_version=version,
        warnings=warnings,
    )


def run_lung_segmentation(
    *,
    input_path: str | Path,
    input_type: str,
    modality: str,
    case_id: str,
    output_root: str | Path,
    device: str = "auto",
    fast: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    loaded = load_medical_image(
        input_path,
        input_type=input_type,
        modality=modality,
    )
    plan = make_plan(
        input_path=loaded.source_path,
        modality=loaded.modality,
        case_id=case_id,
        output_root=output_root,
        device=device,
        fast=fast,
    )

    if plan.case_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output case directory already exists: {display_path(plan.case_dir)}. "
            "Use --overwrite only when replacing generated files for this case."
        )

    if dry_run:
        return {
            "status": "dry-run",
            "plan": plan_to_safe_dict(plan),
            "image": loaded.safe_summary(),
        }

    if overwrite and plan.case_dir.exists():
        shutil.rmtree(plan.case_dir)

    plan.raw_dir.mkdir(parents=True, exist_ok=False)
    plan.lungs_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now(timezone.utc)
    start_time = time.perf_counter()

    process = runner(
        plan.command,
        shell=False,
        check=False,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "TotalSegmentator failed with exit code "
            f"{process.returncode}. Stderr: {_safe_process_text(process.stderr)}"
        )

    selected = select_lung_masks(plan.raw_dir, loaded.modality)
    combined_image, stats = combine_lung_masks(
        selected_mask_paths=[path for _, path in selected],
        reference_image=loaded.image,
    )
    sitk.WriteImage(combined_image, str(plan.combined_mask_path))

    for label, raw_path in selected:
        shutil.copy2(raw_path, plan.lungs_dir / raw_path.name)

    end = datetime.now(timezone.utc)
    runtime_seconds = time.perf_counter() - start_time
    manifest = build_manifest(
        plan=plan,
        loaded=loaded,
        selected_labels=[label for label, _ in selected],
        stats=stats,
        start=start,
        end=end,
        runtime_seconds=runtime_seconds,
        status="success",
    )
    plan.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def select_lung_masks(raw_dir: str | Path, modality: str) -> list[tuple[str, Path]]:
    normalized = normalize_modality(modality, allow_auto=False)
    allowed = LUNG_LABELS_BY_MODALITY[normalized]
    raw_path = Path(raw_dir)
    available = {path.name.removesuffix(".nii.gz").removesuffix(".nii"): path for path in raw_path.glob("*.nii*")}

    selected = [(label, available[label]) for label in allowed if label in available]
    if not selected:
        available_names = sorted(available)
        raise FileNotFoundError(
            f"No verified {normalized} lung masks found. Available class filenames: "
            f"{available_names}"
        )
    return selected


def combine_lung_masks(
    *,
    selected_mask_paths: Sequence[Path],
    reference_image: sitk.Image,
) -> tuple[sitk.Image, MaskStats]:
    if not selected_mask_paths:
        raise ValueError("No lung mask paths were provided.")

    reference_geom = summarize_geometry(reference_image)
    combined = np.zeros(reference_geom.array_shape_zyx, dtype=np.uint8)
    for mask_path in selected_mask_paths:
        mask_image = sitk.ReadImage(str(mask_path))
        mismatches = compare_image_geometry(reference_image, mask_image)
        if mismatches:
            raise ValueError(
                f"Geometry mismatch for mask {mask_path.name}: " + "; ".join(mismatches)
            )
        mask_array = sitk.GetArrayFromImage(mask_image)
        combined |= (mask_array > 0).astype(np.uint8)

    voxel_count = int(np.count_nonzero(combined))
    if voxel_count == 0:
        raise ValueError("Combined lung mask is empty.")

    combined_image = sitk.GetImageFromArray(combined)
    combined_image.CopyInformation(reference_image)
    combined_image = sitk.Cast(combined_image, sitk.sitkUInt8)

    mismatches = compare_image_geometry(reference_image, combined_image)
    if mismatches:
        raise ValueError("Combined mask geometry mismatch: " + "; ".join(mismatches))

    volume_mm3 = float(voxel_count * np.prod(reference_geom.spacing_xyz))
    stats = MaskStats(
        voxel_count=voxel_count,
        volume_mm3=volume_mm3,
        volume_ml=volume_mm3 / 1000.0,
    )
    return combined_image, stats


def build_manifest(
    *,
    plan: SegmentationPlan,
    loaded: LoadedMedicalImage,
    selected_labels: Sequence[str],
    stats: MaskStats,
    start: datetime,
    end: datetime,
    runtime_seconds: float,
    status: str,
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": plan.case_id,
        "modality": plan.modality,
        "task": plan.task,
        "totalsegmentator_version": plan.totalseg_version,
        "execution_mode": "fast" if plan.fast else "full-resolution",
        "resolved_device": plan.device.resolved,
        "fast_mode": plan.fast,
        "input_path": display_path(loaded.source_path),
        "selected_lung_labels": list(selected_labels),
        "raw_output_dir": display_path(plan.raw_dir),
        "combined_mask_filename": plan.combined_mask_path.name,
        "image_size_xyz": list(loaded.size_xyz),
        "spacing_xyz_mm": list(loaded.spacing_xyz),
        "mask_voxel_count": stats.voxel_count,
        "mask_volume_mm3": stats.volume_mm3,
        "mask_volume_ml": stats.volume_ml,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "runtime_seconds": runtime_seconds,
        "completion_status": status,
        "warnings": list(plan.warnings),
    }


def plan_to_safe_dict(plan: SegmentationPlan) -> dict[str, Any]:
    return {
        "case_id": plan.case_id,
        "modality": plan.modality,
        "task": plan.task,
        "totalsegmentator_version": plan.totalseg_version,
        "resolved_device": plan.device.resolved,
        "fast_mode": plan.fast,
        "input_path": display_path(plan.input_path),
        "case_dir": display_path(plan.case_dir),
        "raw_dir": display_path(plan.raw_dir),
        "lungs_dir": display_path(plan.lungs_dir),
        "manifest_path": display_path(plan.manifest_path),
        "combined_mask_path": display_path(plan.combined_mask_path),
        "command": [display_path(arg) if "/" in arg else arg for arg in plan.command],
        "warnings": list(plan.warnings),
    }


def _reject_raw_output(path: Path) -> None:
    resolved = path.resolve(strict=False)
    raw_root = DEFAULT_RAW_DATA_DIR.resolve(strict=False)
    try:
        resolved.relative_to(raw_root)
    except ValueError:
        return
    raise ValueError("Refusing to write segmentation outputs under data/raw/.")


def _safe_process_text(text: str | None) -> str:
    if not text:
        return ""
    return text.strip()[-1000:]
