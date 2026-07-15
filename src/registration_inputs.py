from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import SimpleITK as sitk

from medical_image_io import compare_image_geometry, summarize_geometry
from project_paths import DEFAULT_OUTPUTS_DIR, DEFAULT_RAW_DATA_DIR, display_path
from version import PIPELINE_VERSION


NIFTI_EXTENSIONS = (".nii", ".nii.gz")
DEFAULT_ISOTROPY_TOLERANCE = 1e-5
GEOMETRY_TOLERANCE = 1e-5
INTENSITY_RTOL = 1e-6
INTENSITY_ATOL = 1e-6
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RegistrationInputPair:
    fixed_source: Path
    moving_source: Path
    case_id: str
    experiment: str = "full"
    role_assignment_source: str = "manual"
    roles_confirmed: bool = False


@dataclass(frozen=True)
class ValidatedRegistrationImage:
    path: Path
    image: sitk.Image
    extension: str
    summary: dict[str, Any]


def validate_registration_image(
    path: str | Path,
    *,
    role: str,
    isotropy_tolerance: float = DEFAULT_ISOTROPY_TOLERANCE,
) -> ValidatedRegistrationImage:
    """Load and validate one 3D scalar NIfTI without reading arbitrary metadata."""
    tolerance = _validate_isotropy_tolerance(isotropy_tolerance)
    candidate = Path(path).expanduser().resolve(strict=False)
    if not candidate.exists():
        raise FileNotFoundError(f"{role.capitalize()} NIfTI file does not exist.")
    if not candidate.is_file():
        raise ValueError(f"{role.capitalize()} input must be a NIfTI file, not a directory.")
    extension = _nifti_extension(candidate)
    try:
        image = sitk.ReadImage(str(candidate))
    except RuntimeError as exc:
        raise ValueError(f"{role.capitalize()} NIfTI could not be read by SimpleITK.") from exc
    if image.GetDimension() != 3:
        raise ValueError(
            f"{role.capitalize()} image must be 3D; found {image.GetDimension()}D."
        )
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"{role.capitalize()} image must have scalar pixels, not vector pixels.")
    geometry = summarize_geometry(image)
    if any(not math.isfinite(value) or value <= 0 for value in geometry.spacing_xyz):
        raise ValueError(f"{role.capitalize()} image spacing must be finite and positive.")
    finite_min, finite_max, has_nan, has_inf = _finite_statistics(image)
    if has_nan or has_inf:
        problems = []
        if has_nan:
            problems.append("NaN")
        if has_inf:
            problems.append("infinite")
        raise ValueError(f"{role.capitalize()} image contains {' and '.join(problems)} values.")
    if finite_min is None or finite_max is None:
        raise ValueError(f"{role.capitalize()} image has no finite voxel values.")
    anisotropic = not np.allclose(
        geometry.spacing_xyz,
        (geometry.spacing_xyz[0],) * 3,
        atol=tolerance,
        rtol=0,
    )
    summary = {
        "dimensionality": image.GetDimension(),
        "scalar_pixels": True,
        "pixel_type": image.GetPixelIDTypeAsString(),
        "pixel_id": image.GetPixelID(),
        "size_xyz": list(geometry.size_xyz),
        "numpy_shape_zyx": list(geometry.array_shape_zyx),
        "spacing_xyz_mm": list(geometry.spacing_xyz),
        "origin_xyz_mm": list(geometry.origin_xyz),
        "direction": list(geometry.direction),
        "finite_min": finite_min,
        "finite_max": finite_max,
        "has_nan": has_nan,
        "has_infinite": has_inf,
        "isotropic_spacing": not anisotropic,
        "anisotropic_spacing": anisotropic,
        "isotropy_tolerance_mm": tolerance,
        "physical_voxel_cell_extent_xyz_mm": [
            size * spacing
            for size, spacing in zip(geometry.size_xyz, geometry.spacing_xyz)
        ],
    }
    return ValidatedRegistrationImage(candidate, image, extension, summary)


def compare_registration_geometry(
    fixed: ValidatedRegistrationImage,
    moving: ValidatedRegistrationImage,
) -> dict[str, Any]:
    """Classify deedsBCV compatibility without resampling either image."""
    fixed_geometry = summarize_geometry(fixed.image)
    moving_geometry = summarize_geometry(moving.image)
    dimensions_match = fixed_geometry.size_xyz == moving_geometry.size_xyz
    spacing_matches = bool(
        np.allclose(
            fixed_geometry.spacing_xyz,
            moving_geometry.spacing_xyz,
            atol=GEOMETRY_TOLERANCE,
            rtol=0,
        )
    )
    origin_matches = bool(
        np.allclose(
            fixed_geometry.origin_xyz,
            moving_geometry.origin_xyz,
            atol=GEOMETRY_TOLERANCE,
            rtol=0,
        )
    )
    direction_matches = bool(
        np.allclose(
            fixed_geometry.direction,
            moving_geometry.direction,
            atol=GEOMETRY_TOLERANCE,
            rtol=0,
        )
    )
    fixed_extent = fixed.summary["physical_voxel_cell_extent_xyz_mm"]
    moving_extent = moving.summary["physical_voxel_cell_extent_xyz_mm"]
    extent_matches = bool(
        np.allclose(fixed_extent, moving_extent, atol=GEOMETRY_TOLERANCE, rtol=0)
    )
    grid_matches = not compare_image_geometry(
        fixed.image,
        moving.image,
        spacing_tolerance=GEOMETRY_TOLERANCE,
        origin_tolerance=GEOMETRY_TOLERANCE,
        direction_tolerance=GEOMETRY_TOLERANCE,
    )
    anisotropy_detected = bool(
        fixed.summary["anisotropic_spacing"] or moving.summary["anisotropic_spacing"]
    )
    if anisotropy_detected:
        classification = "anisotropic"
    elif not dimensions_match:
        classification = "different-dimensions"
    elif not grid_matches:
        classification = "same-dimensions-different-geometry"
    else:
        classification = "same-grid"
    resampling_required = classification != "same-grid"
    gzip_required = fixed.extension == ".nii" or moving.extension == ".nii"
    return {
        "classification": classification,
        "dimensions_match": dimensions_match,
        "spacing_matches": spacing_matches,
        "origin_matches": origin_matches,
        "direction_matches": direction_matches,
        "voxel_grid_matches": grid_matches,
        "physical_extent_matches": extent_matches,
        "resampling_required": resampling_required,
        "gzip_preparation_required": gzip_required,
        "anisotropy_detected": anisotropy_detected,
        "geometry_tolerance": GEOMETRY_TOLERANCE,
    }


def prepare_registration_inputs(
    pair: RegistrationInputPair,
    *,
    output_root: str | Path = DEFAULT_OUTPUTS_DIR / "registration",
    isotropy_tolerance: float = DEFAULT_ISOTROPY_TOLERANCE,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate and prepare role-named .nii.gz copies without registration."""
    _validate_pair(pair)
    fixed = validate_registration_image(
        pair.fixed_source, role="fixed", isotropy_tolerance=isotropy_tolerance
    )
    moving = validate_registration_image(
        pair.moving_source, role="moving", isotropy_tolerance=isotropy_tolerance
    )
    comparison = compare_registration_geometry(fixed, moving)
    root = Path(output_root).expanduser().resolve(strict=False)
    experiment_dir = root / pair.case_id / pair.experiment
    preprocessed_dir = experiment_dir / "preprocessed"
    _reject_raw_output(preprocessed_dir)
    outputs = {
        "fixed": preprocessed_dir / "fixed.nii.gz",
        "moving": preprocessed_dir / "moving.nii.gz",
        "manifest": preprocessed_dir / "preprocessing_manifest.json",
    }
    source_paths = {fixed.path.resolve(strict=False), moving.path.resolve(strict=False)}
    if any(path.resolve(strict=False) in source_paths for path in outputs.values()):
        raise ValueError("A preprocessing output path would overwrite a source NIfTI file.")
    if experiment_dir.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"Registration experiment already exists: {display_path(experiment_dir)}. "
            "Use --overwrite to replace only preprocessing products."
        )

    warnings = []
    if not pair.roles_confirmed:
        warnings.append("Fixed/moving role assignment is not researcher-confirmed.")
    if comparison["anisotropy_detected"]:
        warnings.append(
            "deedsBCV does not account for anisotropic spacing; isotropic resampling is required."
        )
    elif comparison["resampling_required"]:
        warnings.append(
            "The images are not on the same voxel grid; a later resampling mode is required."
        )

    actions = ["validate-explicit-fixed-and-moving", "validate-finite-scalar-3d-nifti"]
    if fixed.extension == ".nii":
        actions.append("gzip-fixed-working-copy")
    else:
        actions.append("rewrite-fixed-role-named-working-copy")
    if moving.extension == ".nii":
        actions.append("gzip-moving-working-copy")
    else:
        actions.append("rewrite-moving-role-named-working-copy")

    manifest = _base_manifest(
        pair,
        fixed,
        moving,
        comparison,
        outputs,
        actions,
        warnings,
        completion_status="dry-run" if dry_run else "pending",
    )
    if dry_run:
        manifest["geometry_verification"] = {"performed": False, "passed": None}
        return manifest
    if comparison["resampling_required"]:
        raise ValueError(
            f"Cannot prepare registration-ready inputs: geometry classification is "
            f"{comparison['classification']}; resampling is not implemented in this checkpoint."
        )

    source_hashes_before = {
        "fixed": sha256_file(fixed.path),
        "moving": sha256_file(moving.path),
    }
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    _check_target_overwrite(outputs.values(), overwrite=overwrite)
    sitk.WriteImage(fixed.image, str(outputs["fixed"]))
    sitk.WriteImage(moving.image, str(outputs["moving"]))

    fixed_output = validate_registration_image(
        outputs["fixed"], role="prepared fixed", isotropy_tolerance=isotropy_tolerance
    )
    moving_output = validate_registration_image(
        outputs["moving"], role="prepared moving", isotropy_tolerance=isotropy_tolerance
    )
    verification = _verify_prepared_images(fixed, moving, fixed_output, moving_output)
    source_hashes_after = {
        "fixed": sha256_file(fixed.path),
        "moving": sha256_file(moving.path),
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A source NIfTI checksum changed during preprocessing.")
    manifest["checksums_sha256"] = {
        "source_before": source_hashes_before,
        "source_after": source_hashes_after,
        "working_copies": {
            "fixed": sha256_file(outputs["fixed"]),
            "moving": sha256_file(outputs["moving"]),
        },
        "source_unchanged": True,
        "note": ".nii and .nii.gz byte hashes are not expected to match.",
    }
    manifest["geometry_verification"] = verification
    manifest["output_file_sizes_bytes"] = {
        "fixed": outputs["fixed"].stat().st_size,
        "moving": outputs["moving"].stat().st_size,
    }
    manifest["preprocessing_actions_performed"] = actions
    manifest["completion_status"] = "complete"
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_statistics(
    image: sitk.Image,
) -> tuple[float | None, float | None, bool, bool]:
    array = sitk.GetArrayViewFromImage(image)
    finite_min: float | None = None
    finite_max: float | None = None
    has_nan = False
    has_inf = False
    block_depth = max(1, min(16, array.shape[0]))
    for start in range(0, array.shape[0], block_depth):
        block = np.asarray(array[start : start + block_depth])
        finite = np.isfinite(block)
        has_nan = has_nan or bool(np.isnan(block).any())
        has_inf = has_inf or bool(np.isinf(block).any())
        if finite.any():
            values = block[finite]
            block_min = float(values.min())
            block_max = float(values.max())
            finite_min = block_min if finite_min is None else min(finite_min, block_min)
            finite_max = block_max if finite_max is None else max(finite_max, block_max)
    return finite_min, finite_max, has_nan, has_inf


def _verify_prepared_images(
    fixed_source: ValidatedRegistrationImage,
    moving_source: ValidatedRegistrationImage,
    fixed_output: ValidatedRegistrationImage,
    moving_output: ValidatedRegistrationImage,
) -> dict[str, Any]:
    role_results = {}
    for role, source, output in (
        ("fixed", fixed_source, fixed_output),
        ("moving", moving_source, moving_output),
    ):
        geometry_preserved = not compare_image_geometry(
            source.image,
            output.image,
            spacing_tolerance=GEOMETRY_TOLERANCE,
            origin_tolerance=GEOMETRY_TOLERANCE,
            direction_tolerance=GEOMETRY_TOLERANCE,
        )
        pixel_type_preserved = source.image.GetPixelID() == output.image.GetPixelID()
        intensity_range_preserved = bool(
            np.allclose(
                [source.summary["finite_min"], source.summary["finite_max"]],
                [output.summary["finite_min"], output.summary["finite_max"]],
                rtol=INTENSITY_RTOL,
                atol=INTENSITY_ATOL,
            )
        )
        role_results[role] = {
            "geometry_preserved": geometry_preserved,
            "pixel_type_preserved": pixel_type_preserved,
            "intensity_range_preserved": intensity_range_preserved,
        }
    output_dimensions_match = (
        fixed_output.image.GetSize() == moving_output.image.GetSize()
    )
    passed = output_dimensions_match and all(
        all(values.values()) for values in role_results.values()
    )
    if not passed:
        raise RuntimeError("Prepared registration inputs failed post-write verification.")
    return {
        "performed": True,
        "passed": True,
        "output_dimensions_match": output_dimensions_match,
        "roles": role_results,
    }


def _base_manifest(
    pair: RegistrationInputPair,
    fixed: ValidatedRegistrationImage,
    moving: ValidatedRegistrationImage,
    comparison: dict[str, Any],
    outputs: dict[str, Path],
    actions: list[str],
    warnings: list[str],
    *,
    completion_status: str,
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": pair.case_id,
        "experiment": pair.experiment,
        "role_assignment": {
            "fixed": display_path(fixed.path),
            "moving": display_path(moving.path),
            "source": pair.role_assignment_source,
            "confirmed": pair.roles_confirmed,
        },
        "sources": {
            "fixed": {"path": display_path(fixed.path), "extension": fixed.extension},
            "moving": {"path": display_path(moving.path), "extension": moving.extension},
        },
        "working_copies": {
            name: display_path(path) for name, path in outputs.items()
        },
        "fixed_geometry": _public_geometry(fixed.summary),
        "moving_geometry": _public_geometry(moving.summary),
        "geometry_comparison": comparison,
        "preprocessing_actions_planned": actions,
        "preprocessing_actions_performed": actions[:2],
        "warnings": warnings,
        "completion_status": completion_status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Research-only and non-diagnostic.",
    }


def _public_geometry(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "pixel_id"}


def _validate_pair(pair: RegistrationInputPair) -> None:
    if pair.fixed_source is None or pair.moving_source is None:
        raise ValueError("Both explicit fixed and moving inputs are required.")
    if Path(pair.fixed_source).expanduser().resolve(strict=False) == Path(
        pair.moving_source
    ).expanduser().resolve(strict=False):
        raise ValueError("Fixed and moving inputs must be different files.")
    for label, value in (("case ID", pair.case_id), ("experiment", pair.experiment)):
        if not SAFE_COMPONENT.fullmatch(value):
            raise ValueError(
                f"{label.capitalize()} must contain only letters, numbers, '.', '_' or '-'."
            )
    if not pair.role_assignment_source.strip():
        raise ValueError("Role assignment source must not be empty.")


def _validate_isotropy_tolerance(value: float) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Isotropy tolerance must be finite and non-negative.")
    return tolerance


def _nifti_extension(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    raise ValueError("Registration input must end with .nii or .nii.gz.")


def _reject_raw_output(path: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(DEFAULT_RAW_DATA_DIR.resolve(strict=False))
    except ValueError:
        return
    raise ValueError("Refusing to write registration outputs under data/raw/.")


def _check_target_overwrite(paths: Any, *, overwrite: bool) -> None:
    existing = [path for path in paths if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing preprocessing products: "
            + ", ".join(display_path(path) for path in existing)
        )
