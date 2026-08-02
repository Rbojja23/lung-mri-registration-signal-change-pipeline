from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lung_ct_functional_analysis_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import SimpleITK as sitk

from project_paths import display_path
from version import PIPELINE_VERSION


TERMINOLOGY = "Exploratory registered regional lung intensity-change analysis"
REPORT_NAME = "functional_lung_report.json"
FIGURE_NAMES = (
    "slice_mean.png",
    "slice_std.png",
    "axial_overlay.png",
    "coronal_overlay.png",
    "sagittal_overlay.png",
    "multi_slice_panel.png",
    "lung_histograms.png",
)


def analyze_functional_lung(
    *,
    fixed_path: str | Path,
    registered_path: str | Path,
    lung_mask_path: str | Path,
    output_dir: str | Path,
    left_lung_path: str | Path | None = None,
    right_lung_path: str | Path | None = None,
    display_percentile: float = 99.0,
    overlay_opacity: float = 0.45,
    histogram_bins: int = 100,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Explore spatially coherent registered intensity changes inside fixed lungs."""
    _validate_options(display_percentile, overlay_opacity, histogram_bins)
    fixed_file, fixed = _read_scalar(fixed_path, "fixed")
    registered_file, registered = _read_scalar(registered_path, "registered moving")
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
            raise ValueError("Left/right union must exactly equal the combined lung mask.")
        partition_source = "supplied semantic masks"
    else:
        left, right = _geometric_lung_partitions(combined, fixed)
        partition_source = "derived geometric physical-x partition; not a semantic segmentation"

    fixed_values = sitk.GetArrayFromImage(fixed).astype(np.float64, copy=False)
    registered_values = sitk.GetArrayFromImage(registered).astype(np.float64, copy=False)
    signed = fixed_values - registered_values
    absolute = np.abs(signed)
    regions = {"whole_lung": combined, "left_lung": left, "right_lung": right}
    regional_statistics = {
        name: {"signed_change": _statistics(signed[mask]), "absolute_change": _statistics(absolute[mask])}
        for name, mask in regions.items()
    }
    slice_statistics = _slice_statistics(signed, absolute, combined, fixed)
    hotspots = {
        "largest_positive_change_region": _largest_component(signed > 0, signed, combined, fixed),
        "largest_negative_change_region": _largest_component(signed < 0, signed, combined, fixed),
        "connectivity": "6-connected face adjacency",
        "threshold_rule": "strict sign only (>0 or <0); no magnitude or clinical threshold",
    }
    asymmetry = _asymmetry(signed[left], signed[right])
    voxel_volume = float(np.prod(fixed.GetSpacing()))
    lung_volumes = {
        name: {"voxel_count": int(np.count_nonzero(mask)), "volume_mm3": float(np.count_nonzero(mask) * voxel_volume), "volume_ml": float(np.count_nonzero(mask) * voxel_volume / 1000.0)}
        for name, mask in regions.items()
    }
    selected = _selected_slices(combined)
    display_limit = float(np.percentile(np.abs(signed[combined]), display_percentile))
    display_limit = max(display_limit, 1e-12)
    output = Path(output_dir).expanduser().resolve(strict=False)
    paths = {"report": output / REPORT_NAME}
    paths.update({Path(name).stem: output / name for name in FIGURE_NAMES})
    if not dry_run and any(path.exists() for path in paths.values()) and not overwrite:
        raise FileExistsError("Functional-lung analysis outputs already exist; use --overwrite.")

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "analysis_type": TERMINOLOGY,
        "inputs": {
            "fixed": display_path(fixed_file),
            "registered_moving": display_path(registered_file),
            "combined_lung_mask": display_path(combined_file),
            "left_lung_mask": display_path(left_file) if left_file else None,
            "right_lung_mask": display_path(right_file) if right_file else None,
        },
        "geometry_validation": {"passed": True, **_geometry(fixed)},
        "analysis_domain": {
            "rule": "combined fixed-space lung mask only",
            "outside_lung_contribution": False,
            "outside_lung_voxel_count": int(combined.size - np.count_nonzero(combined)),
            "left_right_partition_source": partition_source,
        },
        "map_definitions": {
            "signed_intensity_change": "fixed - registered_moving",
            "absolute_intensity_change": "abs(fixed - registered_moving)",
            "normalization_performed": False,
        },
        "lung_volumes": lung_volumes,
        "regional_statistics": regional_statistics,
        "slice_statistics": slice_statistics,
        "hotspot_statistics": hotspots,
        "asymmetry": asymmetry,
        "display_scaling": {
            "colormap": "RdBu_r",
            "center": 0.0,
            "limits": [-display_limit, display_limit],
            "lung_only_absolute_percentile": display_percentile,
            "shared_across_all_overlays": True,
            "overlay_opacity": overlay_opacity,
        },
        "selected_slices": selected,
        "histogram_bins": histogram_bins,
        "interpretation": {
            "positive_values": "Fixed intensity is greater than registered moving intensity at that fixed-space lung voxel.",
            "negative_values": "Fixed intensity is less than registered moving intensity at that fixed-space lung voxel.",
            "interpolation": "Registered intensities include contributions from trilinear interpolation, which can smooth and mix neighboring values.",
            "scope": "Spatial coherence is described, not assigned a physiological cause.",
            "conclusion": "This analysis is not a validated ventilation measurement or biomarker.",
        },
        "warnings": [
            "This is exploratory registered regional lung intensity-change analysis, not ventilation.",
            "Respiratory state and inhale/exhale ordering are unknown.",
            "Moving modality and intensity-comparability assumptions remain unresolved.",
            "Registration error and trilinear interpolation can contribute to observed spatial patterns.",
            "Positive and negative connected regions use sign only and must not be interpreted biologically.",
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

    output.mkdir(parents=True, exist_ok=True)
    anatomy_limits = _anatomy_limits(fixed_values)
    _render_slice_plot(slice_statistics, "signed_mean", "Mean fixed − registered intensity", paths["slice_mean"])
    _render_slice_plot(slice_statistics, "signed_standard_deviation", "SD of fixed − registered intensity", paths["slice_std"])
    for plane in ("axial", "coronal", "sagittal"):
        _render_plane_overlay(fixed_values, signed, combined, fixed.GetSpacing(), selected, plane, display_limit, overlay_opacity, anatomy_limits, paths[f"{plane}_overlay"])
    _render_multi_slice(fixed_values, signed, combined, selected["axial_panel_z"], fixed.GetSpacing(), display_limit, overlay_opacity, anatomy_limits, paths["multi_slice_panel"])
    _render_histograms(signed[combined], signed[left], signed[right], histogram_bins, paths["lung_histograms"])
    report["completion_status"] = "complete"
    report["writes_performed"] = True
    paths["report"].write_text(json.dumps(report, indent=2) + "\n")
    return report


def _validate_options(display_percentile, opacity, bins):
    if not 0 < float(display_percentile) <= 100:
        raise ValueError("Display percentile must be in (0, 100].")
    if not 0 <= float(opacity) <= 1:
        raise ValueError("Overlay opacity must be in [0, 1].")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("Histogram bins must be an integer of at least 2.")


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
    array = sitk.GetArrayFromImage(image)
    if not np.all(np.isin(np.unique(array), [0, 1])):
        raise ValueError(f"{role.capitalize()} mask must be binary.")
    mask = array.astype(bool)
    if not mask.any():
        raise ValueError(f"{role.capitalize()} mask is empty.")
    return file, mask


def _require_same_geometry(a, b, label):
    fields = (("size", a.GetSize(), b.GetSize()), ("spacing", a.GetSpacing(), b.GetSpacing()), ("origin", a.GetOrigin(), b.GetOrigin()), ("direction", a.GetDirection(), b.GetDirection()))
    mismatch = [name for name, left, right in fields if tuple(left) != tuple(right)]
    if mismatch:
        raise ValueError(f"{label} geometry mismatch: " + ", ".join(mismatch))


def _geometry(image):
    return {"size_xyz": list(image.GetSize()), "spacing_xyz_mm": list(image.GetSpacing()), "origin_xyz_mm": list(image.GetOrigin()), "direction": list(image.GetDirection()), "exact_agreement_required": True, "resampling_performed": False}


def _geometric_lung_partitions(mask, image):
    coords = np.argwhere(mask)
    physical_x = np.array([image.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))[0] for z, y, x in coords])
    boundary = float((physical_x.min() + physical_x.max()) / 2.0)
    left = np.zeros(mask.shape, dtype=bool)
    right = np.zeros(mask.shape, dtype=bool)
    for (z, y, x), coordinate in zip(coords, physical_x):
        (left if coordinate > boundary else right)[z, y, x] = True
    if not left.any() or not right.any():
        raise ValueError("Geometric physical-x partition produced an empty side; supply semantic left/right masks.")
    return left, right


def _statistics(values):
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "voxel_count": int(values.size), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "standard_deviation": float(np.std(values)), "minimum": float(np.min(values)), "maximum": float(np.max(values)),
        "interquartile_range": float(percentiles[4] - percentiles[2]),
        "percentiles": {str(p): float(v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), percentiles)},
        "positive_fraction": float(np.count_nonzero(values > 0) / values.size),
        "negative_fraction": float(np.count_nonzero(values < 0) / values.size),
    }


def _slice_statistics(signed, absolute, mask, image):
    result = []
    for z in range(mask.shape[0]):
        selected = mask[z]
        count = int(np.count_nonzero(selected))
        physical_z = float(image.TransformIndexToPhysicalPoint((0, 0, z))[2])
        if count:
            signed_values, absolute_values = signed[z][selected], absolute[z][selected]
            item = {
                "axial_index_z": z, "physical_position_z_mm": physical_z, "lung_voxel_count": count,
                "signed_mean": float(np.mean(signed_values)), "signed_median": float(np.median(signed_values)),
                "signed_standard_deviation": float(np.std(signed_values)),
                "absolute_mean": float(np.mean(absolute_values)), "absolute_standard_deviation": float(np.std(absolute_values)),
            }
        else:
            item = {"axial_index_z": z, "physical_position_z_mm": physical_z, "lung_voxel_count": 0, "signed_mean": None, "signed_median": None, "signed_standard_deviation": None, "absolute_mean": None, "absolute_standard_deviation": None}
        result.append(item)
    return result


def _largest_component(condition, signed, combined, image):
    active = condition & combined
    visited = np.zeros(active.shape, dtype=bool)
    best: list[tuple[int, int, int]] = []
    shape = active.shape
    for seed in np.argwhere(active):
        start = tuple(int(value) for value in seed)
        if visited[start]:
            continue
        component = []
        queue = deque([start]); visited[start] = True
        while queue:
            z, y, x = queue.popleft(); component.append((z, y, x))
            for nz, ny, nx in ((z-1,y,x),(z+1,y,x),(z,y-1,x),(z,y+1,x),(z,y,x-1),(z,y,x+1)):
                if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2] and active[nz, ny, nx] and not visited[nz, ny, nx]:
                    visited[nz, ny, nx] = True; queue.append((nz, ny, nx))
        if len(component) > len(best):
            best = component
    if not best:
        return {"available": False, "voxel_count": 0, "reason": "No voxels satisfy the sign rule."}
    coords = np.asarray(best)
    centroid_zyx = np.mean(coords, axis=0)
    centroid_xyz_mm = image.TransformContinuousIndexToPhysicalPoint((float(centroid_zyx[2]), float(centroid_zyx[1]), float(centroid_zyx[0])))
    lower, upper = coords.min(axis=0), coords.max(axis=0) + 1
    values = signed[tuple(coords.T)]
    return {
        "available": True, "voxel_count": len(best),
        "volume_mm3": float(len(best) * np.prod(image.GetSpacing())),
        "centroid_index_zyx": centroid_zyx.tolist(), "centroid_physical_xyz_mm": list(centroid_xyz_mm),
        "bounding_box_half_open_zyx": {"minimum": lower.tolist(), "maximum_exclusive": upper.tolist()},
        "signed_change_mean": float(np.mean(values)), "signed_change_minimum": float(np.min(values)), "signed_change_maximum": float(np.max(values)),
    }


def _asymmetry(left, right):
    left_mean, right_mean = float(np.mean(left)), float(np.mean(right))
    left_var, right_var = float(np.var(left)), float(np.var(right))
    denominator = math.sqrt(((left.size - 1) * left_var + (right.size - 1) * right_var) / (left.size + right.size - 2)) if left.size + right.size > 2 else 0.0
    return {
        "left_mean_signed_change": left_mean, "right_mean_signed_change": right_mean,
        "left_minus_right_mean_difference": left_mean - right_mean,
        "pooled_standardized_effect_size": None if denominator == 0 else float((left_mean - right_mean) / denominator),
        "effect_size_definition": "descriptive pooled-SD standardized mean difference; no biological interpretation",
    }


def _selected_slices(mask):
    coords = np.argwhere(mask); lower, upper = coords.min(axis=0), coords.max(axis=0)
    center = np.rint((lower + upper) / 2).astype(int)
    panel = np.rint(np.linspace(lower[0], upper[0], 7)).astype(int)
    return {"central": {"axial_z": int(center[0]), "coronal_y": int(center[1]), "sagittal_x": int(center[2])}, "axial_panel_z": [int(value) for value in panel]}


def _anatomy_limits(values):
    low, high = np.percentile(values, [1, 99]); return (float(low), float(high)) if low < high else (float(low - 0.5), float(high + 0.5))


def _render_slice_plot(statistics, key, ylabel, path):
    selected = [item for item in statistics if item[key] is not None]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([item["physical_position_z_mm"] for item in selected], [item[key] for item in selected], marker="o", markersize=2)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Axial physical position z (mm)"); axis.set_ylabel(ylabel)
    axis.set_title("Lung-only axial slice trend — descriptive, not ventilation")
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _plane_view(array, selected, plane):
    central = selected["central"]
    if plane == "axial": return array[central["axial_z"]], central["axial_z"]
    if plane == "coronal": return array[:, central["coronal_y"], :], central["coronal_y"]
    return array[:, :, central["sagittal_x"]], central["sagittal_x"]


def _plane_aspect(spacing, plane):
    if plane == "axial": return spacing[1] / spacing[0]
    if plane == "coronal": return spacing[2] / spacing[0]
    return spacing[2] / spacing[1]


def _render_plane_overlay(fixed, signed, mask, spacing, selected, plane, limit, opacity, anatomy_limits, path):
    anatomy, index = _plane_view(fixed, selected, plane); values, _ = _plane_view(signed, selected, plane); domain, _ = _plane_view(mask, selected, plane)
    fig, axis = plt.subplots(figsize=(7, 6)); aspect = _plane_aspect(spacing, plane)
    axis.imshow(anatomy, cmap="gray", vmin=anatomy_limits[0], vmax=anatomy_limits[1], origin="lower", aspect=aspect)
    overlay = axis.imshow(np.ma.masked_where(~domain, values), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), alpha=opacity, origin="lower", aspect=aspect)
    fig.colorbar(overlay, ax=axis, label="Fixed − registered intensity")
    axis.set_title(f"{plane.capitalize()} index {index}\nExploratory intensity change; not ventilation"); axis.axis("off")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _render_multi_slice(fixed, signed, mask, slices, spacing, limit, opacity, anatomy_limits, path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7)); norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit); last = None
    for axis, z in zip(axes.flat, slices):
        axis.imshow(fixed[z], cmap="gray", vmin=anatomy_limits[0], vmax=anatomy_limits[1], origin="lower", aspect=spacing[1]/spacing[0])
        last = axis.imshow(np.ma.masked_where(~mask[z], signed[z]), cmap="RdBu_r", norm=norm, alpha=opacity, origin="lower", aspect=spacing[1]/spacing[0])
        axis.set_title(f"Axial z={z}"); axis.axis("off")
    for axis in axes.flat[len(slices):]: axis.axis("off")
    fig.colorbar(last, ax=axes, shrink=0.75, label="Fixed − registered intensity")
    fig.suptitle("Lung-only regional intensity change — research-only, not ventilation")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _render_histograms(whole, left, right, bins, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(whole, bins=bins, color="steelblue"); axes[0].axvline(0, color="black"); axes[0].set_title("Whole-lung signed change")
    edges = np.histogram_bin_edges(whole, bins=bins)
    axes[1].hist(left, bins=edges, alpha=0.55, label="left"); axes[1].hist(right, bins=edges, alpha=0.55, label="right"); axes[1].axvline(0, color="black"); axes[1].set_title("Left versus right"); axes[1].legend()
    ordered = np.sort(whole); axes[2].plot(ordered, np.arange(1, ordered.size + 1) / ordered.size); axes[2].axvline(0, color="black"); axes[2].set_title("Signed cumulative distribution"); axes[2].set_ylabel("Cumulative fraction")
    for axis in axes: axis.set_xlabel("Fixed − registered intensity")
    fig.suptitle("Descriptive lung-only distributions; no physiological thresholds")
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
