from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lung_ct_intensity_change_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from lung_slice_selection import select_lung_visualization_slices
from project_paths import DEFAULT_OUTPUTS_DIR, display_path
from registration_inputs import sha256_file
from version import PIPELINE_VERSION


TERMINOLOGY = "Preliminary voxelwise registered intensity-change map"
ANALYTICAL_NAMES = {
    "signed": "signed_difference.nii.gz",
    "absolute": "absolute_difference.nii.gz",
    "normalized": "normalized_difference.nii.gz",
}
DISPLAY_NAMES = {
    "signed": "signed_difference_display.nii.gz",
    "absolute": "absolute_difference_display.nii.gz",
    "normalized": "normalized_difference_display.nii.gz",
}
FIGURE_NAMES = {
    "signed_overlay": "signed_difference_overlay.png",
    "absolute_overlay": "absolute_difference_overlay.png",
    "normalized_overlay": "normalized_difference_overlay.png",
    "signed_contact_sheet": "signed_difference_contact_sheet.png",
    "normalized_contact_sheet": "normalized_difference_contact_sheet.png",
    "histogram": "intensity_change_histogram.png",
}


def generate_intensity_change_map(
    *,
    fixed_path: str | Path,
    registered_path: str | Path,
    lung_mask_path: str | Path,
    case_id: str,
    experiment: str = "full",
    output_root: str | Path = DEFAULT_OUTPUTS_DIR / "registration",
    left_lung_mask_path: str | Path | None = None,
    right_lung_mask_path: str | Path | None = None,
    denominator_policy: str | None = None,
    denominator_threshold: float | None = None,
    denominator_percentile: float | None = None,
    skip_normalized: bool = False,
    display_percentile: float = 99.0,
    overlay_opacity: float = 0.45,
    histogram_bins: int = 100,
    overwrite: bool = False,
    dry_run: bool = False,
    figures_only: bool = False,
) -> dict[str, Any]:
    """Create research-only registered intensity-change maps without resampling."""
    _validate_options(
        denominator_policy, denominator_threshold, denominator_percentile,
        skip_normalized, display_percentile, overlay_opacity, histogram_bins,
    )
    output_dir = Path(output_root).expanduser().resolve(strict=False) / case_id / experiment / "intensity_change"
    output_paths = _output_paths(output_dir, include_normalized=not skip_normalized)
    if figures_only:
        return _regenerate_figures_only(
            fixed_path=fixed_path,
            lung_mask_path=lung_mask_path,
            output_paths=output_paths,
            include_normalized=not skip_normalized,
            dry_run=dry_run,
        )
    fixed_file, fixed = _read_scalar_image(fixed_path, "fixed")
    registered_file, registered = _read_scalar_image(registered_path, "registered")
    _require_same_geometry(fixed, registered, "Fixed/registered")
    mask_file, lung_mask = _read_mask(lung_mask_path, fixed, "combined lung")
    optional_masks: dict[str, tuple[Path, np.ndarray]] = {}
    if left_lung_mask_path is not None:
        optional_masks["left"] = _read_mask(left_lung_mask_path, fixed, "left lung")
    if right_lung_mask_path is not None:
        optional_masks["right"] = _read_mask(right_lung_mask_path, fixed, "right lung")
    if "left" in optional_masks and "right" in optional_masks:
        left = optional_masks["left"][1]
        right = optional_masks["right"][1]
        if np.any(left & right):
            raise ValueError("Left and right lung masks overlap.")
        if not np.array_equal(left | right, lung_mask):
            raise ValueError("Left/right mask union must exactly equal the combined lung mask.")

    fixed_array = sitk.GetArrayFromImage(fixed).astype(np.float64, copy=False)
    registered_array = sitk.GetArrayFromImage(registered).astype(np.float64, copy=False)
    difference = fixed_array - registered_array
    analytical = {
        "signed": np.where(lung_mask, difference, np.nan),
        "absolute": np.where(lung_mask, np.abs(difference), np.nan),
    }
    threshold = None
    valid_normalized = None
    invalid_denominator_count = 0
    if not skip_normalized:
        threshold = _resolve_threshold(
            fixed_array, lung_mask, denominator_policy,
            denominator_threshold, denominator_percentile,
        )
        valid_normalized = lung_mask & (np.abs(fixed_array) >= threshold)
        invalid_denominator_count = int(np.count_nonzero(lung_mask & ~valid_normalized))
        normalized = np.full(fixed_array.shape, np.nan, dtype=np.float64)
        normalized[valid_normalized] = difference[valid_normalized] / fixed_array[valid_normalized]
        analytical["normalized"] = normalized

    regions = {"combined_lung": lung_mask}
    regions.update({f"{name}_lung": value[1] for name, value in optional_masks.items()})
    statistics = {
        name: {
            region_name: _statistics(values, region, lung_mask, name, valid_normalized)
            for region_name, region in regions.items()
        }
        for name, values in analytical.items()
    }
    display_arrays: dict[str, np.ndarray] = {}
    display_scaling: dict[str, Any] = {}
    for name, values in analytical.items():
        valid = np.isfinite(values) & lung_mask
        magnitudes = np.abs(values[valid])
        limit = float(np.percentile(magnitudes, display_percentile)) if magnitudes.size else 0.0
        if not math.isfinite(limit) or limit <= 0:
            limit = 1.0
        low, high = (-limit, limit) if name in {"signed", "normalized"} else (0.0, limit)
        clipped_mask = valid & ((values < low) | (values > high))
        display = np.zeros(values.shape, dtype=np.float32)
        display[valid] = np.clip(values[valid], low, high).astype(np.float32)
        display_arrays[name] = display
        clipped_count = int(np.count_nonzero(clipped_mask))
        display_scaling[name] = {
            "analytical_min": float(np.min(values[valid])) if valid.any() else None,
            "analytical_max": float(np.max(values[valid])) if valid.any() else None,
            "display_min": low,
            "display_max": high,
            "clipping_percentile": display_percentile,
            "clipped_voxel_count": clipped_count,
            "clipped_fraction_of_valid": clipped_count / int(np.count_nonzero(valid)) if valid.any() else 0.0,
        }

    selected = _selected_slices(lung_mask)
    if not dry_run:
        existing = [path for path in output_paths.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError("Intensity-change outputs already exist; use --overwrite.")

    mask_paths = {"combined": display_path(mask_file)}
    mask_paths.update({name: display_path(value[0]) for name, value in optional_masks.items()})
    checksums = {
        "fixed": sha256_file(fixed_file),
        "registered": sha256_file(registered_file),
        "combined_lung_mask": sha256_file(mask_file),
    }
    checksums.update({f"{name}_lung_mask": sha256_file(value[0]) for name, value in optional_masks.items()})
    denominator = {
        "normalized_requested": not skip_normalized,
        "policy": denominator_policy if not skip_normalized else None,
        "absolute_threshold": denominator_threshold if denominator_policy == "absolute" and not skip_normalized else None,
        "percentile": denominator_percentile if denominator_policy == "percentile" and not skip_normalized else None,
        "derived_threshold": threshold,
        "rule": "combined_lung_mask and abs(fixed) >= threshold" if not skip_normalized else None,
        "epsilon_added": False,
        "invalid_denominator_voxel_count": invalid_denominator_count,
    }
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": case_id,
        "experiment": experiment,
        "terminology": TERMINOLOGY,
        "inputs": {
            "fixed": display_path(fixed_file),
            "registered_moving": display_path(registered_file),
            "masks": mask_paths,
        },
        "checksums_sha256": checksums,
        "geometry_validation": {"passed": True, **_geometry_dict(fixed)},
        "formulas": {
            "signed_difference": "fixed - registered_moving",
            "absolute_difference": "abs(fixed - registered_moving)",
            "normalized_difference": "(fixed - registered_moving) / fixed on the valid-denominator domain" if not skip_normalized else "not requested",
        },
        "interpolation_context": "Registered moving intensities were produced by deedsBCV using trilinear interpolation.",
        "denominator_policy": denominator,
        "domain_counts": {
            "total_voxels": int(lung_mask.size),
            "combined_lung_voxels": int(np.count_nonzero(lung_mask)),
            "excluded_outside_mask_voxels": int(lung_mask.size - np.count_nonzero(lung_mask)),
        },
        "statistics": statistics,
        "display_scaling": display_scaling,
        "selected_slices": selected,
        "visualization": {
            "anatomy_percentiles": [1.0, 99.0],
            "overlay_opacity": overlay_opacity,
            "histogram_bins": histogram_bins,
        },
        "outputs": {name: display_path(path) for name, path in output_paths.items()},
        "itksnap": {
            "display_maps_are_intensity_overlays": True,
            "automatic_launch": False,
        },
        "warnings": [
            "This is research-only and not a validated ventilation map or clinical biomarker.",
            "Moving modality is unconfirmed.",
            "Respiratory state and inhale/exhale ordering are unconfirmed.",
            "Intensity normalization and calibration are unconfirmed.",
            "Interpolation can smooth and mix registered moving-image intensities.",
            "Percentile denominator policy is exploratory." if denominator_policy == "percentile" and not skip_normalized else None,
        ],
        "unresolved_scientific_assumptions": {
            "moving_modality_confirmed": False,
            "respiratory_state_confirmed": False,
            "intensity_calibration_confirmed": False,
        },
        "completion_status": "dry-run" if dry_run else "pending",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Research-only and non-diagnostic.",
    }
    manifest["warnings"] = [item for item in manifest["warnings"] if item]
    if dry_run:
        manifest["writes_performed"] = False
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in analytical.items():
        _write_array(output_paths[f"{name}_analytical"], array.astype(np.float32), fixed)
        _write_array(output_paths[f"{name}_display"], display_arrays[name], fixed)
    _render_outputs(
        fixed_array, analytical, lung_mask, selected, display_scaling,
        output_paths, overlay_opacity, histogram_bins, fixed.GetSpacing(),
    )
    manifest["completion_status"] = "complete"
    manifest["writes_performed"] = True
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _validate_options(policy, threshold, percentile, skip, display_percentile, opacity, bins):
    if not 0 < float(display_percentile) <= 100:
        raise ValueError("Display percentile must be in (0, 100].")
    if not 0 <= float(opacity) <= 1:
        raise ValueError("Overlay opacity must be in [0, 1].")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("Histogram bins must be an integer of at least 2.")
    if skip:
        return
    if policy not in {"absolute", "percentile"}:
        raise ValueError("Normalized output requires an explicit denominator policy.")
    if policy == "absolute":
        if threshold is None or not math.isfinite(float(threshold)) or float(threshold) <= 0:
            raise ValueError("Absolute denominator policy requires a finite positive threshold.")
        if percentile is not None:
            raise ValueError("Denominator percentile is only valid with percentile policy.")
    else:
        if percentile is None or not 0 <= float(percentile) <= 100:
            raise ValueError("Percentile denominator policy requires a percentile in [0, 100].")
        if threshold is not None:
            raise ValueError("Denominator threshold is only valid with absolute policy.")


def _read_scalar_image(path, role):
    file = _validated_path(path, role)
    image = sitk.ReadImage(str(file))
    if image.GetDimension() != 3:
        raise ValueError(f"{role.capitalize()} image must be 3D.")
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"{role.capitalize()} image must be scalar.")
    array = sitk.GetArrayViewFromImage(image)
    if not np.isfinite(array).all():
        raise ValueError(f"{role.capitalize()} image contains NaN or infinity.")
    return file, image


def _validated_path(path, role):
    file = Path(path).expanduser().resolve(strict=False)
    if not file.is_file():
        raise FileNotFoundError(f"{role.capitalize()} NIfTI does not exist.")
    if not (file.name.endswith(".nii") or file.name.endswith(".nii.gz")):
        raise ValueError(f"{role.capitalize()} input must use .nii or .nii.gz.")
    return file


def _read_mask(path, reference, role):
    file, image = _read_scalar_image(path, role)
    _require_same_geometry(reference, image, f"Fixed/{role} mask")
    array = sitk.GetArrayFromImage(image)
    values = np.unique(array)
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(f"{role.capitalize()} mask must be binary.")
    mask = array.astype(bool)
    if not mask.any():
        raise ValueError(f"{role.capitalize()} mask is empty.")
    return file, mask


def _require_same_geometry(a, b, label):
    fields = (
        ("size", a.GetSize(), b.GetSize()),
        ("spacing", a.GetSpacing(), b.GetSpacing()),
        ("origin", a.GetOrigin(), b.GetOrigin()),
        ("direction", a.GetDirection(), b.GetDirection()),
    )
    mismatches = [name for name, left, right in fields if tuple(left) != tuple(right)]
    if mismatches:
        raise ValueError(f"{label} geometry mismatch: " + ", ".join(mismatches))


def _resolve_threshold(fixed, mask, policy, threshold, percentile):
    if policy == "absolute":
        return float(threshold)
    derived = float(np.percentile(np.abs(fixed[mask]), float(percentile)))
    if not math.isfinite(derived) or derived <= 0:
        raise ValueError(
            "Exploratory percentile policy derived a nonpositive threshold; choose another explicit policy or percentile."
        )
    return derived


def _statistics(values, region, combined, name, valid_normalized):
    valid = region & np.isfinite(values)
    selected = values[valid]
    total_region = int(np.count_nonzero(region))
    if selected.size == 0:
        summary = {key: None for key in ("mean", "median", "standard_deviation", "minimum", "maximum", "interquartile_range", "positive_fraction", "negative_fraction", "zero_fraction")}
        summary["percentiles"] = {str(p): None for p in (1, 5, 25, 50, 75, 95, 99)}
    else:
        percentiles = np.percentile(selected, [1, 5, 25, 50, 75, 95, 99])
        summary = {
            "mean": float(np.mean(selected)),
            "median": float(np.median(selected)),
            "standard_deviation": float(np.std(selected)),
            "minimum": float(np.min(selected)),
            "maximum": float(np.max(selected)),
            "interquartile_range": float(percentiles[4] - percentiles[2]),
            "percentiles": {str(p): float(v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), percentiles)},
            "positive_fraction": float(np.count_nonzero(selected > 0) / selected.size),
            "negative_fraction": float(np.count_nonzero(selected < 0) / selected.size),
            "zero_fraction": float(np.count_nonzero(selected == 0) / selected.size),
        }
    summary.update({
        "valid_voxel_count": int(selected.size),
        "invalid_denominator_count": total_region - int(selected.size) if name == "normalized" else 0,
        "excluded_outside_mask_voxel_count": int(combined.size - np.count_nonzero(combined)),
    })
    return summary


def _selected_slices(mask):
    selected = select_lung_visualization_slices(mask, axial_panel_count=7)
    selected["lung_extent_zyx_inclusive"] = selected.pop("lung_extent_zyx")
    return selected


def _output_paths(directory, include_normalized):
    names = {}
    for name in ("signed", "absolute", "normalized"):
        if name == "normalized" and not include_normalized:
            continue
        names[f"{name}_analytical"] = directory / ANALYTICAL_NAMES[name]
        names[f"{name}_display"] = directory / DISPLAY_NAMES[name]
    for name, filename in FIGURE_NAMES.items():
        if name.startswith("normalized") and not include_normalized:
            continue
        names[name] = directory / filename
    names["manifest"] = directory / "intensity_change_manifest.json"
    return names


def _regenerate_figures_only(*, fixed_path, lung_mask_path, output_paths, include_normalized, dry_run):
    """Regenerate PNG products from existing analytical maps without rewriting analysis."""
    manifest_path = output_paths["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError("Figures-only mode requires an existing intensity-change manifest.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("completion_status") != "complete":
        raise ValueError("Figures-only mode requires a complete intensity-change manifest.")

    _, fixed = _read_scalar_image(fixed_path, "fixed")
    _, lung_mask = _read_mask(lung_mask_path, fixed, "combined lung")
    fixed_array = sitk.GetArrayFromImage(fixed).astype(np.float64, copy=False)
    names = ["signed", "absolute"] + (["normalized"] if include_normalized else [])
    analytical = {
        name: _read_existing_analytical(output_paths[f"{name}_analytical"], fixed, name)
        for name in names
    }
    scaling = {name: manifest["display_scaling"][name] for name in names}
    _validate_display_scaling(scaling)
    selected = _selected_slices(lung_mask)
    result = dict(manifest)
    result["selected_slices"] = selected
    result["completion_status"] = "figures-only-dry-run" if dry_run else "figures-only-complete"
    result["writes_performed"] = not dry_run
    if dry_run:
        return result

    visualization = manifest["visualization"]
    _render_outputs(
        fixed_array,
        analytical,
        lung_mask,
        selected,
        scaling,
        output_paths,
        float(visualization["overlay_opacity"]),
        int(visualization["histogram_bins"]),
        fixed.GetSpacing(),
    )
    return result


def _read_existing_analytical(path, reference, name):
    if not path.is_file():
        raise FileNotFoundError(f"Figures-only mode requires existing {name} analytical NIfTI: {path}")
    image = sitk.ReadImage(str(path))
    _require_same_geometry(reference, image, f"Fixed/{name} analytical map")
    data_xyz = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float64, copy=False)
    data_zyx = np.transpose(data_xyz, (2, 1, 0))
    if data_zyx.shape != tuple(reversed(reference.GetSize())):
        raise ValueError(f"Existing {name} analytical map shape does not match fixed image.")
    return data_zyx


def _validate_display_scaling(scaling):
    for name, values in scaling.items():
        low = float(values["display_min"])
        high = float(values["display_max"])
        if name in {"signed", "normalized"} and (low >= 0 or high <= 0 or not np.isclose(low, -high)):
            raise ValueError(f"Existing {name} display limits are not symmetric around zero.")
        if name == "absolute" and (low < 0 or high <= low):
            raise ValueError("Existing absolute display limits must be nonnegative and increasing.")


def _geometry_dict(image):
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz_mm": list(image.GetSpacing()),
        "origin_xyz_mm": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "exact_agreement_required": True,
        "resampling_performed": False,
        "optional_left_right_union_rule": "exact equality with combined lung mask when both are supplied",
    }


def _write_array(path, array, reference):
    # SimpleITK's NIfTI writer in the supported environment converts IEEE NaN
    # payload values to zero.  Let it establish the fixed-space NIfTI geometry,
    # then replace only the voxel payload through nibabel so analytical NaNs are
    # retained.  NumPy/SimpleITK arrays are z,y,x while nibabel arrays are x,y,z.
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))
    geometry_image = nib.load(str(path))
    header = geometry_image.header.copy()
    header.set_data_dtype(np.float32)
    payload_xyz = np.transpose(np.asarray(array, dtype=np.float32), (2, 1, 0))
    output = nib.Nifti1Image(payload_xyz, geometry_image.affine, header=header)
    output.set_qform(geometry_image.get_qform(), int(geometry_image.header["qform_code"]))
    output.set_sform(geometry_image.get_sform(), int(geometry_image.header["sform_code"]))
    nib.save(output, str(path))


def _anatomy_range(fixed):
    low, high = np.percentile(fixed, [1, 99])
    return (float(low), float(high)) if low < high else (float(low - 0.5), float(high + 0.5))


def _views(array, central, spacing):
    return (
        ("Axial", array[central["axial_z"]], spacing[1] / spacing[0], central["axial_z"]),
        ("Coronal", array[:, central["coronal_y"], :], spacing[2] / spacing[0], central["coronal_y"]),
        ("Sagittal", array[:, :, central["sagittal_x"]], spacing[2] / spacing[1], central["sagittal_x"]),
    )


def _render_outputs(fixed, analytical, mask, selected, scaling, paths, opacity, histogram_bins, spacing_xyz):
    low, high = _anatomy_range(fixed)
    for name, values in analytical.items():
        _render_overlay(fixed, values, mask, selected["central"], spacing_xyz, scaling[name], paths[f"{name}_overlay"], name, low, high, opacity)
    _render_contact_sheet(fixed, analytical["signed"], mask, selected["axial_contact_sheet_z"], scaling["signed"], paths["signed_contact_sheet"], "signed", low, high, opacity, spacing_xyz)
    if "normalized" in analytical:
        _render_contact_sheet(fixed, analytical["normalized"], mask, selected["axial_contact_sheet_z"], scaling["normalized"], paths["normalized_contact_sheet"], "normalized", low, high, opacity, spacing_xyz)
    _render_histogram(analytical, paths["histogram"], histogram_bins)


def _style(name, scaling):
    if name == "absolute":
        return "magma", Normalize(vmin=0, vmax=scaling["display_max"]), "Absolute registered intensity difference"
    label = "Relative registered intensity change" if name == "normalized" else "Fixed − registered intensity"
    return "RdBu_r", TwoSlopeNorm(vmin=scaling["display_min"], vcenter=0, vmax=scaling["display_max"]), label


def _figure_title(name):
    return {
        "signed": "Signed registered lung intensity difference: F - R",
        "absolute": "Absolute registered lung intensity difference: |F - R|",
        "normalized": "Normalized registered lung intensity difference: (F - R) / F",
    }[name]


def _draw_lung_contour(axis, mask_view):
    if np.any(mask_view) and not np.all(mask_view):
        axis.contour(
            mask_view.astype(np.uint8),
            levels=[0.5],
            colors="white",
            linewidths=0.55,
            origin="lower",
        )


def _render_overlay(fixed, values, mask, central, spacing, scaling, path, name, low, high, opacity):
    cmap, norm, label = _style(name, scaling)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    last = None
    for axis, fixed_view, value_view, mask_view in zip(
        axes,
        _views(fixed, central, spacing),
        _views(values, central, spacing),
        _views(mask, central, spacing),
    ):
        title, anatomy, aspect, index = fixed_view
        domain = mask_view[1].astype(bool)
        overlay = np.ma.masked_where(~domain | ~np.isfinite(value_view[1]), value_view[1])
        axis.imshow(anatomy, cmap="gray", vmin=low, vmax=high, origin="lower", aspect=aspect)
        last = axis.imshow(overlay, cmap=cmap, norm=norm, alpha=opacity, origin="lower", aspect=aspect)
        _draw_lung_contour(axis, domain)
        axis.set_title(f"{title} index {index}")
        axis.axis("off")
    fig.colorbar(last, ax=axes, shrink=0.75, label=label)
    fig.suptitle(f"{_figure_title(name)}\nResearch-only; not validated ventilation", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_contact_sheet(fixed, values, mask, slices, scaling, path, name, low, high, opacity, spacing_xyz):
    cmap, norm, label = _style(name, scaling)
    fig, axes = plt.subplots(1, len(slices), figsize=(15, 3.4))
    axes = np.atleast_1d(axes)
    last = None
    for axis, z in zip(axes, slices):
        aspect = spacing_xyz[1] / spacing_xyz[0]
        axis.imshow(fixed[z], cmap="gray", vmin=low, vmax=high, origin="lower", aspect=aspect)
        overlay = np.ma.masked_where(~mask[z] | ~np.isfinite(values[z]), values[z])
        last = axis.imshow(overlay, cmap=cmap, norm=norm, alpha=opacity, origin="lower", aspect=aspect)
        _draw_lung_contour(axis, mask[z])
        axis.set_title(f"Axial z={z}")
        axis.axis("off")
    fig.colorbar(last, ax=axes, shrink=0.72, label=label)
    fig.suptitle(f"{_figure_title(name)}\nResearch-only; not validated ventilation", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_histogram(analytical, path, bins):
    names = ["signed"] + (["normalized"] if "normalized" in analytical else [])
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4), squeeze=False)
    for axis, name in zip(axes[0], names):
        values = analytical[name][np.isfinite(analytical[name])]
        axis.hist(values, bins=bins, color="steelblue", alpha=0.85)
        axis.axvline(0, color="black", linewidth=1, label="zero")
        if values.size:
            axis.axvline(float(np.median(values)), color="darkorange", linewidth=1, label="median")
        axis.set_xlabel("Relative intensity change" if name == "normalized" else "Original intensity units")
        axis.set_ylabel("Lung voxel count")
        axis.set_title(name.capitalize() + " difference")
        axis.legend()
    fig.suptitle("Lung-only intensity-change distributions; research-only")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
