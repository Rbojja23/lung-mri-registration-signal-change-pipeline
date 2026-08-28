from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lung_ct_normalized_map_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from lung_slice_selection import select_lung_visualization_slices
from project_paths import display_path
from version import PIPELINE_VERSION


TERMINOLOGY = "Exploratory normalized registered intensity-change map"
OUTPUT_NAMES = {
    "analytical": "normalized_difference.nii.gz",
    "display": "normalized_difference_display.nii.gz",
    "axial": "normalized_overlay_axial.png",
    "coronal": "normalized_overlay_coronal.png",
    "sagittal": "normalized_overlay_sagittal.png",
    "multi_slice": "normalized_multi_slice.png",
    "report": "normalized_intensity_report.json",
}
MACHINE_EPSILON_FLOAT64 = float(np.finfo(np.float64).eps)
NEAR_ZERO_THRESHOLDS = (1e-12, 1e-9, 1e-6)


def generate_normalized_intensity_map(
    *,
    fixed_path: str | Path,
    registered_path: str | Path,
    lung_mask_path: str | Path,
    output_dir: str | Path,
    denominator_policy: str,
    denominator_threshold: float | None = None,
    epsilon: str | float | None = None,
    left_lung_path: str | Path | None = None,
    right_lung_path: str | Path | None = None,
    display_percentile: float = 99.0,
    overlay_opacity: float = 0.45,
    overwrite: bool = False,
    dry_run: bool = False,
    figures_only: bool = False,
) -> dict[str, Any]:
    """Generate an Exploratory normalized registered intensity-change map."""
    epsilon_value, epsilon_source = _validate_options(
        denominator_policy, denominator_threshold, epsilon, display_percentile, overlay_opacity
    )
    destination = Path(output_dir).expanduser().resolve(strict=False)
    paths = {name: destination / filename for name, filename in OUTPUT_NAMES.items()}
    if figures_only:
        return _regenerate_figures_only(
            fixed_path=fixed_path,
            lung_mask_path=lung_mask_path,
            paths=paths,
            dry_run=dry_run,
        )
    fixed_file, fixed = _read_scalar(fixed_path, "fixed")
    registered_file, registered = _read_scalar(registered_path, "registered")
    _require_same_geometry(fixed, registered, "Fixed/registered")
    combined_file, combined = _read_mask(lung_mask_path, fixed, "combined lung")

    left_file = right_file = None
    if (left_lung_path is None) != (right_lung_path is None):
        raise ValueError("Left and right lung masks must be supplied together or both omitted.")
    if left_lung_path is not None:
        left_file, left = _read_mask(left_lung_path, fixed, "left lung")
        right_file, right = _read_mask(right_lung_path, fixed, "right lung")
        if np.any(left & right):
            raise ValueError("Left and right lung masks overlap.")
        if not np.array_equal(left | right, combined):
            raise ValueError("Left/right mask union must exactly equal the combined lung mask.")
        partition_source = "supplied semantic masks"
    else:
        left, right = _geometric_partitions(combined, fixed)
        partition_source = "derived geometric physical-x partition; not a semantic segmentation"

    fixed_array = sitk.GetArrayFromImage(fixed).astype(np.float64, copy=False)
    registered_array = sitk.GetArrayFromImage(registered).astype(np.float64, copy=False)
    if denominator_policy == "absolute":
        valid = combined & (np.abs(fixed_array) >= float(denominator_threshold))
        invalid_denominator = combined & ~valid
        denominator = fixed_array
    else:
        valid = combined.copy()
        invalid_denominator = np.zeros(combined.shape, dtype=bool)
        denominator = fixed_array + epsilon_value
        if np.any(combined & (denominator == 0)):
            raise ValueError(
                "F + epsilon is exactly zero for at least one lung voxel; the requested formula is undefined "
                "there (this can occur when a negative fixed intensity equals -epsilon)."
            )
    analytical = np.full(fixed_array.shape, np.nan, dtype=np.float32)
    calculated = (fixed_array[valid] - registered_array[valid]) / denominator[valid]
    if not np.isfinite(calculated).all() or np.any(np.abs(calculated) > np.finfo(np.float32).max):
        raise ValueError("The normalized result is not finite and representable as the required float32 analytical map.")
    analytical[valid] = calculated.astype(np.float32)

    regions = {"whole_lung": combined, "left_lung": left, "right_lung": right}
    statistics = {name: _statistics(analytical, mask, valid, invalid_denominator) for name, mask in regions.items()}
    valid_values = analytical[valid].astype(np.float64)
    display_limit = float(np.percentile(np.abs(valid_values), display_percentile)) if valid_values.size else 0.0
    display_limit = max(display_limit, 1e-12)
    display = np.zeros(analytical.shape, dtype=np.float32)
    display[valid] = np.clip(analytical[valid], -display_limit, display_limit)
    clipped = valid & ((analytical < -display_limit) | (analytical > display_limit))
    selected = _selected_slices(combined)

    if not dry_run and any(path.exists() for path in paths.values()) and not overwrite:
        raise FileExistsError("Normalized intensity-map outputs already exist; use --overwrite.")

    fixed_lung = fixed_array[combined]
    fixed_intensity_summary = {
        "minimum": float(np.min(fixed_lung)),
        "maximum": float(np.max(fixed_lung)),
        "exact_zero_voxel_count": int(np.count_nonzero(fixed_lung == 0)),
        "near_zero_absolute_threshold_counts": {
            str(threshold): int(np.count_nonzero(np.abs(fixed_lung) <= threshold))
            for threshold in NEAR_ZERO_THRESHOLDS
        },
        "near_zero_definition": "abs(fixed) <= threshold; descriptive only and never used for exclusion in epsilon mode",
        "negative_voxel_count": int(np.count_nonzero(fixed_lung < 0)),
    }
    if denominator_policy == "absolute":
        formula = {
            "expression": "(fixed - registered) / fixed",
            "valid_domain": "combined fixed-space lung mask and abs(fixed) >= denominator threshold",
            "normalization_epsilon": None,
            "resampling_performed": False,
            "interpolation_performed_by_this_checkpoint": False,
        }
        denominator_report = {
            "policy": "absolute",
            "threshold": float(denominator_threshold),
            "threshold_must_be_positive": True,
            "valid_voxel_count": int(np.count_nonzero(valid)),
            "invalid_denominator_count": int(np.count_nonzero(invalid_denominator)),
            "outside_lung_voxel_count": int(combined.size - np.count_nonzero(combined)),
        }
    else:
        formula = {
            "expression": "(fixed - registered) / (fixed + epsilon)",
            "valid_domain": "combined fixed-space lung mask; no threshold-based voxel exclusion",
            "normalization_epsilon": epsilon_value,
            "resampling_performed": False,
            "interpolation_performed_by_this_checkpoint": False,
        }
        denominator_report = {
            "policy": "epsilon",
            "epsilon_source": epsilon_source,
            "epsilon": epsilon_value,
            "computation_dtype": "float64",
            "threshold": None,
            "threshold_exclusion_performed": False,
            "valid_voxel_count": int(np.count_nonzero(valid)),
            "invalid_denominator_count": 0,
            "outside_lung_voxel_count": int(combined.size - np.count_nonzero(combined)),
            "exact_zero_stabilized_denominator_count": 0,
        }

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "analysis_type": TERMINOLOGY,
        "inputs": {
            "fixed": display_path(fixed_file),
            "registered": display_path(registered_file),
            "combined_lung_mask": display_path(combined_file),
            "left_lung_mask": display_path(left_file) if left_file else None,
            "right_lung_mask": display_path(right_file) if right_file else None,
        },
        "geometry_validation": {"passed": True, **_geometry(fixed)},
        "formula": formula,
        "denominator_policy": denominator_report,
        "fixed_lung_intensity": fixed_intensity_summary,
        "left_right_partition_source": partition_source,
        "statistics": statistics,
        "display_scaling": {
            "colormap": "RdBu_r",
            "center": 0.0,
            "limits": [-display_limit, display_limit],
            "valid_lung_absolute_percentile": display_percentile,
            "clipped_voxel_count": int(np.count_nonzero(clipped)),
            "clipped_fraction_of_valid": float(np.count_nonzero(clipped) / np.count_nonzero(valid)) if valid.any() else 0.0,
            "shared_across_all_figures": True,
            "overlay_opacity": overlay_opacity,
        },
        "selected_slices": selected,
        "interpretation": {
            "terminology": TERMINOLOGY,
            "exploratory": "The ratio is an exploratory registered intensity-change calculation.",
            "not_ventilation": "It is NOT a validated ventilation measurement or biomarker.",
            "positive_values": "A positive value means the executed numerator and selected denominator have the same sign.",
            "negative_values": "A negative value means the executed numerator and selected denominator have opposite signs.",
            "physiology": "No physiological interpretation is assigned.",
        },
        "warnings": [
            f"{TERMINOLOGY} is exploratory and not validated ventilation.",
            "Respiratory phase and inhale/exhale ordering are unknown.",
            "Moving-image modality is unknown.",
            "MR intensity calibration and cross-acquisition comparability are unknown.",
            "Registration and its trilinear interpolation contribute to the registered intensities.",
            "Near-threshold denominators can amplify noise and registration differences." if denominator_policy == "absolute" else None,
            "Machine epsilon protects an exact zero fixed intensity numerically but does not meaningfully stabilize genuinely small denominators." if denominator_policy == "epsilon" and epsilon_source == "machine" else None,
            "An explicit epsilon is a user-selected sensitivity-analysis value, not a medically validated stabilization constant." if denominator_policy == "epsilon" and epsilon_source == "explicit" else None,
            "The requested fixed + epsilon denominator is asymmetric around zero; positive epsilon shifts negative fixed intensities toward zero and can cancel fixed = -epsilon exactly." if denominator_policy == "epsilon" else None,
            "Derived left/right partitions are geometric, not semantic." if left_file is None else None,
        ],
        "outputs": {name: display_path(path) for name, path in paths.items()},
        "completion_status": "dry-run" if dry_run else "pending",
        "writes_performed": False if dry_run else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Research-only and non-diagnostic.",
    }
    report["warnings"] = [warning for warning in report["warnings"] if warning]
    if dry_run:
        return report

    destination.mkdir(parents=True, exist_ok=True)
    _write_float_nifti(paths["analytical"], analytical, fixed)
    _write_float_nifti(paths["display"], display, fixed)
    anatomy = _anatomy_limits(fixed_array)
    title = _normalization_title(denominator_policy)
    for plane in ("axial", "coronal", "sagittal"):
        _render_plane(fixed_array, analytical, valid, combined, fixed.GetSpacing(), selected, plane, display_limit, overlay_opacity, anatomy, paths[plane], title)
    _render_multi_slice(fixed_array, analytical, valid, combined, fixed.GetSpacing(), selected["axial_panel_z"], display_limit, overlay_opacity, anatomy, paths["multi_slice"], title)
    report["completion_status"] = "complete"
    report["writes_performed"] = True
    paths["report"].write_text(json.dumps(report, indent=2) + "\n")
    return report


def _validate_options(policy, threshold, epsilon, percentile, opacity):
    if policy not in {"absolute", "epsilon"}:
        raise ValueError("Denominator policy must be absolute or epsilon.")
    epsilon_value = None
    epsilon_source = None
    if policy == "absolute":
        if threshold is None or not math.isfinite(float(threshold)) or float(threshold) <= 0:
            raise ValueError("Denominator threshold must be finite and greater than zero.")
        if epsilon is not None:
            raise ValueError("--epsilon is only valid with denominator-policy epsilon.")
    else:
        if threshold is not None:
            raise ValueError("--denominator-threshold is only valid with denominator-policy absolute.")
        if isinstance(epsilon, str) and epsilon.strip().lower() == "machine":
            epsilon_value = MACHINE_EPSILON_FLOAT64
            epsilon_source = "machine"
        else:
            try:
                epsilon_value = float(epsilon)
            except (TypeError, ValueError):
                raise ValueError("Epsilon must be 'machine' or a finite number greater than zero.") from None
            epsilon_source = "explicit"
        if not math.isfinite(epsilon_value) or epsilon_value <= 0:
            raise ValueError("Epsilon must be finite and greater than zero.")
    if not 0 < float(percentile) <= 100:
        raise ValueError("Display percentile must be in (0, 100].")
    if not 0 <= float(opacity) <= 1:
        raise ValueError("Overlay opacity must be in [0, 1].")
    return epsilon_value, epsilon_source


def _read_scalar(path, role):
    file = Path(path).expanduser().resolve(strict=False)
    if not file.is_file():
        raise FileNotFoundError(f"{role.capitalize()} image does not exist.")
    if not (file.name.endswith(".nii") or file.name.endswith(".nii.gz")):
        raise ValueError(f"{role.capitalize()} image must use .nii or .nii.gz.")
    image = sitk.ReadImage(str(file))
    if image.GetDimension() != 3:
        raise ValueError(f"{role.capitalize()} image must be 3D.")
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"{role.capitalize()} image must be scalar.")
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError(f"{role.capitalize()} image contains NaN or infinity.")
    return file, image


def _read_mask(path, reference, role):
    file, image = _read_scalar(path, role)
    _require_same_geometry(reference, image, f"Fixed/{role}-mask")
    values = sitk.GetArrayFromImage(image)
    if not np.all(np.isin(np.unique(values), [0, 1])):
        raise ValueError(f"{role.capitalize()} mask must be binary.")
    mask = values.astype(bool)
    if not mask.any():
        raise ValueError(f"{role.capitalize()} mask is empty.")
    return file, mask


def _require_same_geometry(a, b, label):
    fields = (("size", a.GetSize(), b.GetSize()), ("spacing", a.GetSpacing(), b.GetSpacing()), ("origin", a.GetOrigin(), b.GetOrigin()), ("direction", a.GetDirection(), b.GetDirection()))
    mismatches = [name for name, left, right in fields if tuple(left) != tuple(right)]
    if mismatches:
        raise ValueError(f"{label} geometry mismatch: " + ", ".join(mismatches))


def _geometry(image):
    return {"size_xyz": list(image.GetSize()), "spacing_xyz_mm": list(image.GetSpacing()), "origin_xyz_mm": list(image.GetOrigin()), "direction": list(image.GetDirection()), "exact_agreement_required": True}


def _geometric_partitions(mask, image):
    coords = np.argwhere(mask)
    physical_x = np.array([image.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))[0] for z, y, x in coords])
    boundary = float((physical_x.min() + physical_x.max()) / 2)
    left = np.zeros(mask.shape, bool); right = np.zeros(mask.shape, bool)
    for (z, y, x), coordinate in zip(coords, physical_x):
        (left if coordinate > boundary else right)[z, y, x] = True
    if not left.any() or not right.any():
        raise ValueError("Geometric partition produced an empty side; supply semantic left/right masks.")
    return left, right


def _statistics(analytical, region, valid, invalid):
    selected = analytical[region & valid].astype(np.float64)
    result = {
        "valid_voxel_count": int(selected.size),
        "invalid_denominator_count": int(np.count_nonzero(region & invalid)),
    }
    if not selected.size:
        result.update({key: None for key in ("mean", "median", "standard_deviation", "minimum", "maximum", "interquartile_range", "positive_fraction", "negative_fraction")})
        result["percentiles"] = {str(p): None for p in (1, 5, 25, 50, 75, 95, 99)}
        return result
    percentiles = np.percentile(selected, [1, 5, 25, 50, 75, 95, 99])
    result.update({
        "mean": float(np.mean(selected)), "median": float(np.median(selected)),
        "standard_deviation": float(np.std(selected)), "minimum": float(np.min(selected)), "maximum": float(np.max(selected)),
        "interquartile_range": float(percentiles[4] - percentiles[2]),
        "percentiles": {str(p): float(value) for p, value in zip((1, 5, 25, 50, 75, 95, 99), percentiles)},
        "positive_fraction": float(np.count_nonzero(selected > 0) / selected.size),
        "negative_fraction": float(np.count_nonzero(selected < 0) / selected.size),
    })
    return result


def _selected_slices(mask):
    return select_lung_visualization_slices(mask, axial_panel_count=7)


def _regenerate_figures_only(*, fixed_path, lung_mask_path, paths, dry_run):
    """Regenerate normalized PNGs while leaving analytical and report files untouched."""
    if not paths["report"].is_file():
        raise FileNotFoundError("Figures-only mode requires an existing normalized intensity report.")
    report = json.loads(paths["report"].read_text())
    if report.get("completion_status") != "complete":
        raise ValueError("Figures-only mode requires a complete normalized intensity report.")

    _, fixed = _read_scalar(fixed_path, "fixed")
    _, combined = _read_mask(lung_mask_path, fixed, "combined lung")
    analytical = _read_existing_analytical(paths["analytical"], fixed)
    valid = combined & np.isfinite(analytical)
    limits = report["display_scaling"]["limits"]
    if len(limits) != 2 or float(limits[0]) >= 0 or float(limits[1]) <= 0 or not np.isclose(float(limits[0]), -float(limits[1])):
        raise ValueError("Existing normalized display limits are not symmetric around zero.")
    limit = float(limits[1])
    opacity = float(report["display_scaling"]["overlay_opacity"])
    selected = _selected_slices(combined)
    result = dict(report)
    result["selected_slices"] = selected
    result["completion_status"] = "figures-only-dry-run" if dry_run else "figures-only-complete"
    result["writes_performed"] = not dry_run
    if dry_run:
        return result

    fixed_array = sitk.GetArrayFromImage(fixed).astype(np.float64, copy=False)
    anatomy = _anatomy_limits(fixed_array)
    policy = report["denominator_policy"]["policy"]
    title = _normalization_title(policy)
    for plane in ("axial", "coronal", "sagittal"):
        _render_plane(fixed_array, analytical, valid, combined, fixed.GetSpacing(), selected, plane, limit, opacity, anatomy, paths[plane], title)
    _render_multi_slice(fixed_array, analytical, valid, combined, fixed.GetSpacing(), selected["axial_panel_z"], limit, opacity, anatomy, paths["multi_slice"], title)
    return result


def _read_existing_analytical(path, reference):
    if not path.is_file():
        raise FileNotFoundError(f"Figures-only mode requires the existing analytical NIfTI: {path}")
    image = sitk.ReadImage(str(path))
    _require_same_geometry(reference, image, "Fixed/normalized analytical map")
    data_xyz = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32, copy=False)
    data_zyx = np.transpose(data_xyz, (2, 1, 0))
    if data_zyx.shape != tuple(reversed(reference.GetSize())):
        raise ValueError("Existing normalized analytical map shape does not match fixed image.")
    return data_zyx


def _write_float_nifti(path, array, reference):
    # Establish the exact fixed-space NIfTI transform with SimpleITK, then use
    # nibabel for the float payload because the supported SimpleITK writer
    # converts IEEE NaNs to zero.
    placeholder = sitk.GetImageFromArray(np.zeros(array.shape, dtype=np.float32))
    placeholder.CopyInformation(reference); sitk.WriteImage(placeholder, str(path))
    geometry = nib.load(str(path)); header = geometry.header.copy(); header.set_data_dtype(np.float32)
    payload_xyz = np.transpose(np.asarray(array, dtype=np.float32), (2, 1, 0))
    result = nib.Nifti1Image(payload_xyz, geometry.affine, header=header)
    result.set_qform(geometry.get_qform(), int(geometry.header["qform_code"])); result.set_sform(geometry.get_sform(), int(geometry.header["sform_code"]))
    nib.save(result, str(path))


def _anatomy_limits(values):
    low, high = np.percentile(values, [1, 99]); return (float(low), float(high)) if low < high else (float(low - 0.5), float(high + 0.5))


def _plane_view(array, selected, plane):
    central = selected["central"]
    if plane == "axial": return array[central["axial_z"]], central["axial_z"]
    if plane == "coronal": return array[:, central["coronal_y"], :], central["coronal_y"]
    return array[:, :, central["sagittal_x"]], central["sagittal_x"]


def _aspect(spacing, plane):
    if plane == "axial": return spacing[1] / spacing[0]
    if plane == "coronal": return spacing[2] / spacing[0]
    return spacing[2] / spacing[1]


def _normalization_title(policy):
    denominator = "(F + epsilon)" if policy == "epsilon" else "F"
    return f"Normalized registered lung intensity difference: (F - R) / {denominator}"


def _draw_lung_contour(axis, mask_view):
    if np.any(mask_view) and not np.all(mask_view):
        axis.contour(mask_view.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.55, origin="lower")


def _render_plane(fixed, analytical, valid, lung_mask, spacing, selected, plane, limit, opacity, anatomy, path, title):
    background, index = _plane_view(fixed, selected, plane); values, _ = _plane_view(analytical, selected, plane); domain, _ = _plane_view(valid, selected, plane); contour, _ = _plane_view(lung_mask, selected, plane)
    fig, axis = plt.subplots(figsize=(7, 6)); aspect = _aspect(spacing, plane)
    axis.imshow(background, cmap="gray", vmin=anatomy[0], vmax=anatomy[1], origin="lower", aspect=aspect)
    overlay = axis.imshow(np.ma.masked_where(~domain, values), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), alpha=opacity, origin="lower", aspect=aspect)
    _draw_lung_contour(axis, contour)
    fig.colorbar(overlay, ax=axis, label="Normalized registered intensity change")
    axis.set_title(f"{title}\n{plane.capitalize()} index {index}; not validated ventilation"); axis.axis("off")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _render_multi_slice(fixed, analytical, valid, lung_mask, spacing, slices, limit, opacity, anatomy, path, title):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7)); norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit); last = None
    for axis, z in zip(axes.flat, slices):
        axis.imshow(fixed[z], cmap="gray", vmin=anatomy[0], vmax=anatomy[1], origin="lower", aspect=spacing[1]/spacing[0])
        last = axis.imshow(np.ma.masked_where(~valid[z], analytical[z]), cmap="RdBu_r", norm=norm, alpha=opacity, origin="lower", aspect=spacing[1]/spacing[0])
        _draw_lung_contour(axis, lung_mask[z])
        axis.set_title(f"Axial z={z}"); axis.axis("off")
    for axis in axes.flat[len(slices):]: axis.axis("off")
    fig.colorbar(last, ax=axes, shrink=0.75, label="Normalized registered intensity change")
    fig.suptitle(f"{title}\nResearch-only; not validated ventilation")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
