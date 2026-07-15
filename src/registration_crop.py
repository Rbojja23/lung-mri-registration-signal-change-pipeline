from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import SimpleITK as sitk

from medical_image_io import compare_image_geometry, summarize_geometry
from project_paths import DEFAULT_OUTPUTS_DIR, DEFAULT_RAW_DATA_DIR, PROJECT_ROOT, display_path
from registration_inputs import (
    DEFAULT_ISOTROPY_TOLERANCE,
    GEOMETRY_TOLERANCE,
    compare_registration_geometry,
    sha256_file,
    validate_registration_image,
)
from version import PIPELINE_VERSION


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BBOX_CONVENTION = "half-open [min:max), matching Python slicing"


@dataclass(frozen=True)
class SharedCropRequest:
    fixed: Path
    moving: Path
    bbox_json: Path
    case_id: str
    source_experiment: str = "full"
    experiment: str = "cropped"
    role_assignment_source: str = "inferred-from-repeated-bimg1"
    roles_confirmed: bool = False


def prepare_shared_registration_crop(
    request: SharedCropRequest,
    *,
    output_root: str | Path = DEFAULT_OUTPUTS_DIR / "registration",
    isotropy_tolerance: float = DEFAULT_ISOTROPY_TOLERANCE,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply one fixed-image half-open bbox to a same-grid registration pair."""
    _validate_request(request)
    fixed = validate_registration_image(
        request.fixed, role="fixed", isotropy_tolerance=isotropy_tolerance
    )
    moving = validate_registration_image(
        request.moving, role="moving", isotropy_tolerance=isotropy_tolerance
    )
    original_comparison = compare_registration_geometry(fixed, moving)
    if not original_comparison["voxel_grid_matches"]:
        raise ValueError("Fixed and moving images must share the exact original voxel grid.")

    bbox_path = Path(request.bbox_json).expanduser().resolve(strict=False)
    bbox_manifest = _load_bbox_manifest(bbox_path)
    crop = _validate_bbox_manifest(bbox_manifest, fixed.path, fixed.image)
    start_xyz = crop["start_index_xyz"]
    size_xyz = crop["size_xyz"]
    fixed_crop = sitk.RegionOfInterest(fixed.image, size_xyz, start_xyz)
    moving_crop = sitk.RegionOfInterest(moving.image, size_xyz, start_xyz)
    expected_origin = tuple(
        float(value) for value in fixed.image.TransformIndexToPhysicalPoint(start_xyz)
    )
    _validate_in_memory_crops(
        fixed.image,
        moving.image,
        fixed_crop,
        moving_crop,
        size_xyz,
        expected_origin,
    )

    root = Path(output_root).expanduser().resolve(strict=False)
    experiment_dir = root / request.case_id / request.experiment
    preprocessed_dir = experiment_dir / "preprocessed"
    _reject_raw_output(preprocessed_dir)
    outputs = {
        "fixed": preprocessed_dir / "fixed.nii.gz",
        "moving": preprocessed_dir / "moving.nii.gz",
        "preprocessing_manifest": preprocessed_dir / "preprocessing_manifest.json",
        "shared_crop_manifest": preprocessed_dir / "shared_crop_manifest.json",
    }
    source_paths = {fixed.path, moving.path, bbox_path}
    if any(path.resolve(strict=False) in source_paths for path in outputs.values()):
        raise ValueError("A shared-crop output would overwrite a source file.")
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite and not dry_run:
        raise FileExistsError(
            "Refusing to overwrite existing shared-crop products: "
            + ", ".join(display_path(path) for path in existing)
        )

    boundary_touch = crop["boundary_touch"]
    touched = [name for name, value in boundary_touch.items() if value]
    warnings = []
    if not request.roles_confirmed:
        warnings.append("Fixed/moving role assignment is not researcher-confirmed.")
    if touched:
        warnings.append(
            "The shared bbox touches image boundaries; requested padding may be clipped."
        )
    if original_comparison["anisotropy_detected"]:
        warnings.append(
            "deedsBCV does not account for anisotropic spacing; isotropic resampling is required."
        )

    original_voxels = int(np.prod(fixed.image.GetSize()))
    cropped_voxels = int(np.prod(size_xyz))
    reduction = float((1.0 - cropped_voxels / original_voxels) * 100.0)
    timestamp = datetime.now(timezone.utc).isoformat()
    plan = {
        "bbox_half_open_zyx": crop["bbox_half_open_zyx"],
        "start_index_xyz": list(start_xyz),
        "size_xyz": list(size_xyz),
        "expected_origin_xyz_mm": list(expected_origin),
        "expected_spacing_xyz_mm": list(fixed.image.GetSpacing()),
        "expected_direction": list(fixed.image.GetDirection()),
        "boundary_touch": boundary_touch,
        "planned_outputs": {name: display_path(path) for name, path in outputs.items()},
    }
    if dry_run:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "case_id": request.case_id,
            "source_experiment": request.source_experiment,
            "experiment": request.experiment,
            "crop_strategy": "shared-fixed-image-bbox",
            "original_grid_match": True,
            "isotropic_spacing": not original_comparison["anisotropy_detected"],
            "resampling_required": original_comparison["resampling_required"],
            **plan,
            "warnings": warnings,
            "completion_status": "dry-run",
            "writes_performed": False,
            "registration_executed": False,
        }
    if original_comparison["resampling_required"]:
        raise ValueError(
            "Cannot create registration-ready cropped inputs from anisotropic images; "
            "resampling is not implemented in this checkpoint."
        )

    source_hashes_before = {
        "fixed": sha256_file(fixed.path),
        "moving": sha256_file(moving.path),
        "bbox_json": sha256_file(bbox_path),
    }
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(fixed_crop, str(outputs["fixed"]))
    sitk.WriteImage(moving_crop, str(outputs["moving"]))
    fixed_written = validate_registration_image(
        outputs["fixed"], role="cropped fixed", isotropy_tolerance=isotropy_tolerance
    )
    moving_written = validate_registration_image(
        outputs["moving"], role="cropped moving", isotropy_tolerance=isotropy_tolerance
    )
    cropped_comparison = compare_registration_geometry(fixed_written, moving_written)
    verification = _verify_written_crops(
        fixed,
        moving,
        fixed_written,
        moving_written,
        size_xyz,
        expected_origin,
    )
    source_hashes_after = {
        "fixed": sha256_file(fixed.path),
        "moving": sha256_file(moving.path),
        "bbox_json": sha256_file(bbox_path),
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A shared-crop source checksum changed during preparation.")
    output_hashes = {
        "fixed": sha256_file(outputs["fixed"]),
        "moving": sha256_file(outputs["moving"]),
    }

    preprocessing_manifest = _preprocessing_manifest(
        request,
        fixed,
        moving,
        fixed_written,
        moving_written,
        outputs,
        cropped_comparison,
        source_hashes_before,
        source_hashes_after,
        output_hashes,
        verification,
        warnings,
        timestamp,
    )
    shared_manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": request.case_id,
        "source_experiment": request.source_experiment,
        "output_experiment": request.experiment,
        "crop_strategy": "shared-fixed-image-bbox",
        "bbox_json_path": display_path(bbox_path),
        "bbox_convention": BBOX_CONVENTION,
        "bbox_half_open_zyx": crop["bbox_half_open_zyx"],
        "simpleitk_start_index_xyz": list(start_xyz),
        "simpleitk_size_xyz": list(size_xyz),
        "original_geometry": _geometry(fixed.image),
        "fixed_cropped_geometry": _geometry(fixed_written.image),
        "moving_cropped_geometry": _geometry(moving_written.image),
        "original_grid_match": True,
        "cropped_grid_match": cropped_comparison["classification"] == "same-grid",
        "boundary_touch": boundary_touch,
        "boundary_touch_note": (
            "Requested padding may be clipped where the bbox touches an image boundary."
            if touched
            else None
        ),
        "voxel_counts": {
            "original": original_voxels,
            "cropped": cropped_voxels,
            "reduction_percentage": reduction,
        },
        "checksums_sha256": {
            "sources_before": source_hashes_before,
            "sources_after": source_hashes_after,
            "outputs": output_hashes,
            "source_unchanged": True,
        },
        "geometry_verification": verification,
        "warnings": warnings,
        "completion_status": "complete",
        "timestamp_utc": timestamp,
        "intended_use": "Research-only and non-diagnostic.",
        "registration_executed": False,
    }
    outputs["preprocessing_manifest"].write_text(
        json.dumps(preprocessing_manifest, indent=2) + "\n"
    )
    outputs["shared_crop_manifest"].write_text(
        json.dumps(shared_manifest, indent=2) + "\n"
    )
    return shared_manifest


def _load_bbox_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Bounding-box JSON does not exist.")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Bounding-box JSON is unreadable or malformed.") from exc
    if not isinstance(value, dict):
        raise ValueError("Bounding-box JSON must contain an object.")
    return value


def _validate_bbox_manifest(
    manifest: dict[str, Any], fixed_path: Path, fixed_image: sitk.Image
) -> dict[str, Any]:
    convention = manifest.get("coordinate_conventions", {}).get("bbox_intervals", "")
    if "half-open" not in convention or "[min:max)" not in convention:
        raise ValueError("Bounding-box interval convention must be half-open [min:max).")
    original_size = manifest.get("original_image_size_xyz")
    if list(fixed_image.GetSize()) != original_size:
        raise ValueError("Bounding-box original image size does not match the fixed image.")
    image_path = manifest.get("image_path")
    if image_path and image_path != "[external path]":
        candidate = Path(image_path).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if not _bbox_path_matches_fixed(candidate, fixed_path):
            raise ValueError("Bounding-box image path does not match the supplied fixed image.")
    bbox = manifest.get("voxel_bbox_half_open_zyx")
    if not isinstance(bbox, dict):
        raise ValueError("Bounding-box JSON lacks voxel_bbox_half_open_zyx.")
    try:
        values = {name: int(bbox[name]) for name in (
            "z_min", "z_max", "y_min", "y_max", "x_min", "x_max"
        )}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Bounding-box bounds must be six integer values.") from exc
    x_size, y_size, z_size = fixed_image.GetSize()
    if not (0 <= values["z_min"] < values["z_max"] <= z_size):
        raise ValueError("Invalid bbox z bounds.")
    if not (0 <= values["y_min"] < values["y_max"] <= y_size):
        raise ValueError("Invalid bbox y bounds.")
    if not (0 <= values["x_min"] < values["x_max"] <= x_size):
        raise ValueError("Invalid bbox x bounds.")
    start = (values["x_min"], values["y_min"], values["z_min"])
    size = (
        values["x_max"] - values["x_min"],
        values["y_max"] - values["y_min"],
        values["z_max"] - values["z_min"],
    )
    if manifest.get("cropped_image_size_xyz") != list(size):
        raise ValueError("Bounding-box cropped image size does not match its half-open bounds.")
    expected_origin = fixed_image.TransformIndexToPhysicalPoint(start)
    for key, expected in (
        ("cropped_spacing_xyz_mm", fixed_image.GetSpacing()),
        ("cropped_origin_xyz_mm", expected_origin),
        ("cropped_direction", fixed_image.GetDirection()),
    ):
        recorded = manifest.get(key)
        if recorded is not None and not np.allclose(
            recorded, expected, atol=GEOMETRY_TOLERANCE, rtol=0
        ):
            raise ValueError(f"Bounding-box {key} does not correspond to the fixed image.")
    return {
        "bbox_half_open_zyx": values,
        "start_index_xyz": start,
        "size_xyz": size,
        "boundary_touch": {
            "x_min": values["x_min"] == 0,
            "x_max": values["x_max"] == x_size,
            "y_min": values["y_min"] == 0,
            "y_max": values["y_max"] == y_size,
            "z_min": values["z_min"] == 0,
            "z_max": values["z_max"] == z_size,
        },
    }


def _bbox_path_matches_fixed(bbox_image_path: Path, fixed_path: Path) -> bool:
    if bbox_image_path.resolve(strict=False) == fixed_path.resolve(strict=False):
        return True
    source_manifest = fixed_path.parent / "preprocessing_manifest.json"
    if not source_manifest.is_file():
        return False
    try:
        manifest = json.loads(source_manifest.read_text())
        recorded = manifest.get("role_assignment", {}).get("fixed")
    except (OSError, json.JSONDecodeError, AttributeError):
        return False
    if not recorded or recorded == "[external path]":
        return False
    recorded_path = Path(recorded).expanduser()
    if not recorded_path.is_absolute():
        recorded_path = PROJECT_ROOT / recorded_path
    return recorded_path.resolve(strict=False) == bbox_image_path.resolve(strict=False)


def _validate_in_memory_crops(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_crop: sitk.Image,
    moving_crop: sitk.Image,
    size_xyz: tuple[int, int, int],
    expected_origin: tuple[float, float, float],
) -> None:
    if fixed_crop.GetSize() != size_xyz or moving_crop.GetSize() != size_xyz:
        raise RuntimeError("Shared crop size does not match the bbox.")
    if compare_image_geometry(fixed_crop, moving_crop):
        raise RuntimeError("Fixed and moving shared crops do not share one voxel grid.")
    for role, source, crop in (("fixed", fixed, fixed_crop), ("moving", moving, moving_crop)):
        if crop.GetSpacing() != source.GetSpacing() or crop.GetDirection() != source.GetDirection():
            raise RuntimeError(f"{role.capitalize()} crop did not preserve spacing and direction.")
        if crop.GetPixelID() != source.GetPixelID():
            raise RuntimeError(f"{role.capitalize()} crop did not preserve pixel type.")
        if not np.allclose(crop.GetOrigin(), expected_origin, atol=GEOMETRY_TOLERANCE, rtol=0):
            raise RuntimeError(f"{role.capitalize()} crop origin is incorrect.")


def _verify_written_crops(fixed, moving, fixed_output, moving_output, size_xyz, expected_origin):
    _validate_in_memory_crops(
        fixed.image,
        moving.image,
        fixed_output.image,
        moving_output.image,
        size_xyz,
        expected_origin,
    )
    return {
        "performed": True,
        "passed": True,
        "same_bbox_applied": True,
        "no_resampling": True,
        "size_xyz_matches": True,
        "spacing_preserved": True,
        "direction_preserved": True,
        "origin_matches_source_index": True,
        "cropped_grid_matches": True,
        "fixed_pixel_type_preserved": fixed.image.GetPixelID() == fixed_output.image.GetPixelID(),
        "moving_pixel_type_preserved": moving.image.GetPixelID() == moving_output.image.GetPixelID(),
        "all_voxels_finite": True,
    }


def _preprocessing_manifest(
    request, fixed, moving, fixed_output, moving_output, outputs, comparison,
    source_before, source_after, output_hashes, verification, warnings, timestamp,
):
    actions = [
        "validate-explicit-fixed-and-moving",
        "validate-finite-scalar-3d-nifti",
        "apply-shared-fixed-image-bbox-with-region-of-interest",
        "verify-cropped-same-grid-pair",
    ]
    return {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": request.case_id,
        "experiment": request.experiment,
        "role_assignment": {
            "fixed": display_path(fixed.path),
            "moving": display_path(moving.path),
            "source": request.role_assignment_source,
            "confirmed": request.roles_confirmed,
        },
        "sources": {
            "fixed": {"path": display_path(fixed.path), "extension": fixed.extension},
            "moving": {"path": display_path(moving.path), "extension": moving.extension},
        },
        "working_copies": {
            "fixed": display_path(outputs["fixed"]),
            "moving": display_path(outputs["moving"]),
            "manifest": display_path(outputs["preprocessing_manifest"]),
        },
        "fixed_geometry": _public_summary(fixed_output.summary),
        "moving_geometry": _public_summary(moving_output.summary),
        "geometry_comparison": comparison,
        "checksums_sha256": {
            "source_before": source_before,
            "source_after": source_after,
            "working_copies": output_hashes,
            "source_unchanged": True,
            "note": "Cropped working-copy hashes are not expected to match source hashes.",
        },
        "geometry_verification": verification,
        "preprocessing_actions_planned": actions,
        "preprocessing_actions_performed": actions,
        "warnings": warnings,
        "completion_status": "complete",
        "timestamp_utc": timestamp,
        "intended_use": "Research-only and non-diagnostic.",
    }


def _geometry(image: sitk.Image) -> dict[str, Any]:
    geometry = summarize_geometry(image)
    return {
        "size_xyz": list(geometry.size_xyz),
        "spacing_xyz_mm": list(geometry.spacing_xyz),
        "origin_xyz_mm": list(geometry.origin_xyz),
        "direction": list(geometry.direction),
        "pixel_type": image.GetPixelIDTypeAsString(),
    }


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "pixel_id"}


def _validate_request(request: SharedCropRequest) -> None:
    paths = [Path(request.fixed), Path(request.moving), Path(request.bbox_json)]
    if paths[0].expanduser().resolve(strict=False) == paths[1].expanduser().resolve(strict=False):
        raise ValueError("Fixed and moving inputs must be different files.")
    for label, value in (
        ("case ID", request.case_id),
        ("source experiment", request.source_experiment),
        ("experiment", request.experiment),
    ):
        if not SAFE_COMPONENT.fullmatch(value):
            raise ValueError(f"{label.capitalize()} contains unsafe characters.")
    if request.source_experiment == request.experiment:
        raise ValueError("Source and output experiments must be different.")
    if not request.role_assignment_source.strip():
        raise ValueError("Role assignment source must not be empty.")


def _reject_raw_output(path: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(DEFAULT_RAW_DATA_DIR.resolve(strict=False))
    except ValueError:
        return
    raise ValueError("Refusing to write registration outputs under data/raw/.")
