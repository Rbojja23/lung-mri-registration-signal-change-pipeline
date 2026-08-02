from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lung_ct_metric_analysis_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import SimpleITK as sitk

from project_paths import display_path
from version import PIPELINE_VERSION


REPORT_NAME = "registration_metric_report.json"
FIGURE_NAMES = ("metric_summary.png", "error_histograms.png", "difference_statistics.png")
METRIC_DEFINITIONS = {
    "mse": "Mean of squared voxelwise errors; lower is better.",
    "mae": "Mean of absolute voxelwise errors; lower is better.",
    "rmse": "Square root of MSE, expressed in the original intensity units; lower is better.",
    "ncc": "Global Pearson correlation of selected voxel intensities; higher is better.",
    "mutual_information": "Statistical dependence from a joint histogram using natural logarithms; higher is generally better.",
    "ssd": "Sum of squared voxelwise errors over the selected domain; lower is better.",
}


def analyze_registration_metrics(
    *,
    fixed_path: str | Path,
    moving_path: str | Path,
    registered_path: str | Path,
    output_dir: str | Path,
    lung_mask_path: str | Path | None = None,
    histogram_bins: int = 100,
    mi_bins: int = 64,
    display_percentile: float = 99.0,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Analyze existing registration metrics and errors without registration."""
    _validate_options(histogram_bins, mi_bins, display_percentile)
    fixed_file, fixed = _read_image(fixed_path, "fixed")
    moving_file, moving = _read_image(moving_path, "moving")
    registered_file, registered = _read_image(registered_path, "registered moving")
    _require_same_geometry(fixed, moving, "Fixed/moving")
    _require_same_geometry(fixed, registered, "Fixed/registered")

    fixed_array = sitk.GetArrayFromImage(fixed).astype(np.float64, copy=False)
    moving_array = sitk.GetArrayFromImage(moving).astype(np.float64, copy=False)
    registered_array = sitk.GetArrayFromImage(registered).astype(np.float64, copy=False)
    lung_file = None
    lung_mask = None
    if lung_mask_path is not None:
        lung_file, lung_mask = _read_mask(lung_mask_path, fixed)

    domains = {"whole_image": np.ones(fixed_array.shape, dtype=bool)}
    if lung_mask is not None:
        domains["lung_only"] = lung_mask
        domains["outside_lung"] = ~lung_mask
    common_range = _common_range(fixed_array, moving_array, registered_array)
    metrics = {
        name: _metric_comparison(
            fixed_array[mask], moving_array[mask], registered_array[mask], mi_bins, common_range
        )
        for name, mask in domains.items()
    }

    signed_before = fixed_array - moving_array
    signed_after = fixed_array - registered_array
    absolute_before = np.abs(signed_before)
    absolute_after = np.abs(signed_after)
    distributions = {
        name: {
            "signed_error_before": _distribution_summary(signed_before[mask]),
            "signed_error_after": _distribution_summary(signed_after[mask]),
            "absolute_error_before": _distribution_summary(absolute_before[mask]),
            "absolute_error_after": _distribution_summary(absolute_after[mask]),
        }
        for name, mask in domains.items()
    }
    interpolation = _interpolation_analysis(
        moving_array, registered_array, domains, histogram_bins, common_range,
        moving.GetPixelIDTypeAsString(), registered.GetPixelIDTypeAsString(),
    )
    selected_mask = lung_mask if lung_mask is not None else domains["whole_image"]
    selected = _representative_slice(selected_mask)
    scales = _display_scales(
        signed_before, signed_after, absolute_before, absolute_after,
        selected_mask, display_percentile,
    )

    destination = Path(output_dir).expanduser().resolve(strict=False)
    paths = {"report": destination / REPORT_NAME}
    paths.update({Path(name).stem: destination / name for name in FIGURE_NAMES})
    if not dry_run and any(path.exists() for path in paths.values()) and not overwrite:
        raise FileExistsError("Registration metric analysis outputs already exist; use --overwrite.")

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "analysis_type": "Standalone registration metric and interpolation analysis",
        "inputs": {
            "fixed": display_path(fixed_file),
            "moving": display_path(moving_file),
            "registered_moving": display_path(registered_file),
            "lung_mask": display_path(lung_file) if lung_file else None,
        },
        "geometry_validation": {"passed": True, **_geometry(fixed)},
        "metric_definitions": METRIC_DEFINITIONS,
        "metrics": metrics,
        "voxelwise_error_definitions": {
            "signed_before": "fixed - moving",
            "signed_after": "fixed - registered_moving",
            "absolute_before": "abs(fixed - moving)",
            "absolute_after": "abs(fixed - registered_moving)",
        },
        "error_distributions": distributions,
        "interpolation_analysis": interpolation,
        "analysis_domains": {name: int(np.count_nonzero(mask)) for name, mask in domains.items()},
        "histogram_configuration": {
            "error_histogram_bins": histogram_bins,
            "mutual_information_bins": mi_bins,
            "mutual_information_range": list(common_range),
            "natural_logarithms": True,
        },
        "display_scaling": {**scales, "percentile": display_percentile, "domain": "lung_only" if lung_mask is not None else "whole_image"},
        "representative_slice": selected,
        "interpretation": _interpretation(metrics),
        "warnings": [
            "No metric is universally best; each measures a different property.",
            "Intensity metrics and error reductions do not establish anatomical correctness.",
            "NCC measures centered pattern agreement and may change less than absolute-error metrics when brightness or scale differs.",
            "Mutual information depends on histogram bins, range, sample size, and background composition.",
            "MAE and MSE can respond to interpolation-induced smoothing and tissue-boundary intensity mixing.",
            "Trilinear interpolation can create floating-point intensity values not present in an integer moving image.",
        ],
        "outputs": {name: display_path(path) for name, path in paths.items()},
        "completion_status": "dry-run" if dry_run else "pending",
        "writes_performed": False if dry_run else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Research-only and non-diagnostic.",
    }
    if dry_run:
        return report

    destination.mkdir(parents=True, exist_ok=True)
    plot_mask = lung_mask if lung_mask is not None else domains["whole_image"]
    _render_metric_summary(metrics, paths["metric_summary"])
    _render_error_histograms(
        signed_before[plot_mask], signed_after[plot_mask], absolute_before[plot_mask],
        absolute_after[plot_mask], histogram_bins, paths["error_histograms"],
    )
    _render_difference_statistics(
        fixed_array, signed_before, signed_after, absolute_before, absolute_after,
        plot_mask, selected, scales, fixed.GetSpacing(), paths["difference_statistics"],
    )
    report["completion_status"] = "complete"
    report["writes_performed"] = True
    paths["report"].write_text(json.dumps(report, indent=2) + "\n")
    return report


def _validate_options(histogram_bins, mi_bins, display_percentile):
    if not isinstance(histogram_bins, int) or histogram_bins < 2:
        raise ValueError("Histogram bins must be an integer of at least 2.")
    if not isinstance(mi_bins, int) or mi_bins < 2:
        raise ValueError("MI bins must be an integer of at least 2.")
    if not 0 < float(display_percentile) <= 100:
        raise ValueError("Display percentile must be in (0, 100].")


def _read_image(path, role):
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


def _read_mask(path, reference):
    file, mask_image = _read_image(path, "lung mask")
    _require_same_geometry(reference, mask_image, "Fixed/lung-mask")
    array = sitk.GetArrayFromImage(mask_image)
    if not np.all(np.isin(np.unique(array), [0, 1])):
        raise ValueError("Lung mask must be binary.")
    mask = array.astype(bool)
    if not mask.any():
        raise ValueError("Lung mask is empty.")
    if mask.all():
        raise ValueError("Lung mask leaves no outside-lung domain.")
    return file, mask


def _require_same_geometry(a, b, label):
    fields = (("size", a.GetSize(), b.GetSize()), ("spacing", a.GetSpacing(), b.GetSpacing()), ("origin", a.GetOrigin(), b.GetOrigin()), ("direction", a.GetDirection(), b.GetDirection()))
    mismatch = [name for name, left, right in fields if tuple(left) != tuple(right)]
    if mismatch:
        raise ValueError(f"{label} geometry mismatch: " + ", ".join(mismatch))


def _geometry(image):
    return {"size_xyz": list(image.GetSize()), "spacing_xyz_mm": list(image.GetSpacing()), "origin_xyz_mm": list(image.GetOrigin()), "direction": list(image.GetDirection()), "exact_agreement_required": True, "resampling_performed": False}


def _common_range(*arrays):
    low = min(float(np.min(array)) for array in arrays)
    high = max(float(np.max(array)) for array in arrays)
    return (low - 0.5, high + 0.5) if low == high else (low, high)


def _metrics(fixed, candidate, bins, value_range):
    if fixed.size == 0:
        raise ValueError("Metric domain is empty.")
    error = fixed - candidate
    ssd = float(np.dot(error, error))
    mse = ssd / fixed.size
    fixed_centered = fixed - np.mean(fixed)
    candidate_centered = candidate - np.mean(candidate)
    denominator = math.sqrt(float(np.dot(fixed_centered, fixed_centered) * np.dot(candidate_centered, candidate_centered)))
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    histogram = np.histogram2d(fixed, candidate, bins=(edges, edges))[0]
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(error))),
        "rmse": math.sqrt(mse),
        "ncc": None if denominator == 0 else float(np.dot(fixed_centered, candidate_centered) / denominator),
        "mutual_information": _mutual_information(histogram),
        "ssd": ssd,
        "voxel_count": int(fixed.size),
    }


def _mutual_information(histogram):
    total = float(histogram.sum())
    if total == 0:
        return 0.0
    joint = histogram / total
    independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    valid = joint > 0
    return float(np.sum(joint[valid] * np.log(joint[valid] / independent[valid])))


def _metric_comparison(fixed, moving, registered, bins, value_range):
    before = _metrics(fixed, moving, bins, value_range)
    after = _metrics(fixed, registered, bins, value_range)
    comparisons = {}
    for name in METRIC_DEFINITIONS:
        old, new = before[name], after[name]
        direction = "lower-is-better" if name in {"mse", "mae", "rmse", "ssd"} else "higher-is-better"
        if old is None or new is None:
            improvement = percent = None
        else:
            improvement = float(old - new) if direction == "lower-is-better" else float(new - old)
            percent = None if old == 0 else float(improvement / abs(old) * 100.0)
        comparisons[name] = {"before": old, "after": new, "absolute_improvement": improvement, "percent_improvement": percent, "direction": direction}
    return {"voxel_count": int(fixed.size), "metrics": comparisons}


def _distribution_summary(values):
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(values.size), "mean": float(np.mean(values)), "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)), "maximum": float(np.max(values)), "median": float(np.median(values)),
        "interquartile_range": float(percentiles[4] - percentiles[2]),
        "percentiles": {str(p): float(v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), percentiles)},
    }


def _interpolation_analysis(moving, registered, domains, bins, value_range, moving_type, registered_type):
    result = {
        "moving_pixel_type": moving_type,
        "registered_pixel_type": registered_type,
        "moving_is_uint16": "16-bit unsigned integer" in moving_type,
        "registered_is_float": "float" in registered_type.lower(),
        "interpolation_method_context": "deedsBCV final intensity warp uses trilinear interpolation.",
        "new_value_explanation": "Trilinear interpolation forms weighted combinations of neighboring intensities and can create floating-point values absent from a uint16 moving image.",
        "domains": {},
    }
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    for name, mask in domains.items():
        moving_values, registered_values = moving[mask], registered[mask]
        moving_hist = np.histogram(moving_values, bins=edges)[0].astype(float)
        registered_hist = np.histogram(registered_values, bins=edges)[0].astype(float)
        moving_prob = moving_hist / moving_hist.sum()
        registered_prob = registered_hist / registered_hist.sum()
        moving_summary = _distribution_summary(moving_values)
        registered_summary = _distribution_summary(registered_values)
        result["domains"][name] = {
            "moving": moving_summary,
            "registered": registered_summary,
            "difference_in_mean_registered_minus_moving": registered_summary["mean"] - moving_summary["mean"],
            "difference_in_variance_registered_minus_moving": float(np.var(registered_values) - np.var(moving_values)),
            "histogram_difference_l1_probability": float(np.sum(np.abs(registered_prob - moving_prob))),
            "histogram_bins": bins,
            "histogram_range": list(value_range),
        }
    return result


def _representative_slice(mask):
    coords = np.argwhere(mask)
    center = np.rint((coords.min(axis=0) + coords.max(axis=0)) / 2).astype(int)
    return {"axial_z": int(center[0]), "coronal_y": int(center[1]), "sagittal_x": int(center[2])}


def _display_scales(signed_before, signed_after, absolute_before, absolute_after, mask, percentile):
    signed_limit = float(np.percentile(np.concatenate((np.abs(signed_before[mask]), np.abs(signed_after[mask]))), percentile))
    absolute_limit = float(np.percentile(np.concatenate((absolute_before[mask], absolute_after[mask])), percentile))
    return {"signed": [-max(signed_limit, 1e-12), max(signed_limit, 1e-12)], "absolute": [0.0, max(absolute_limit, 1e-12)]}


def _interpretation(metrics):
    whole = metrics["whole_image"]["metrics"]
    return {
        "mse": "MSE squares errors, so large mismatches and outliers receive disproportionate weight.",
        "mae": "MAE measures average absolute voxelwise error and responds directly to interpolation and intensity-scale changes.",
        "rmse": "RMSE is the square root of MSE and remains in the original intensity units.",
        "ncc": "NCC measures centered structural or pattern agreement rather than absolute equality; it may change less when brightness or scale differs.",
        "mutual_information": "MI measures statistical dependence and is often useful for multimodality, but depends on histogram construction and sampled background.",
        "observed_whole_image": {name: values["absolute_improvement"] for name, values in whole.items()},
        "overall": "Metric changes should be interpreted together; no single metric is universally superior.",
    }


def _render_metric_summary(metrics, path):
    names = list(METRIC_DEFINITIONS)
    values = [metrics["whole_image"]["metrics"][name]["percent_improvement"] for name in names]
    plotted = [0 if value is None else value for value in values]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    colors = ["#4472C4" if value >= 0 else "#C44E52" for value in plotted]
    axis.bar(names, plotted, color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Improvement relative to before (%)")
    axis.set_title("Registration metric summary — no metric is universally best")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_error_histograms(signed_before, signed_after, absolute_before, absolute_after, bins, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(signed_before, bins=bins, alpha=0.55, label="before")
    axes[0, 0].hist(signed_after, bins=bins, alpha=0.55, label="after")
    axes[0, 0].axvline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Signed error histogram")
    axes[0, 0].legend()
    axes[0, 1].hist(absolute_before, bins=bins, alpha=0.55, label="before")
    axes[0, 1].hist(absolute_after, bins=bins, alpha=0.55, label="after")
    axes[0, 1].set_title("Absolute error histogram")
    axes[0, 1].legend()
    for values, label in ((absolute_before, "before"), (absolute_after, "after")):
        ordered = np.sort(values)
        axes[1, 0].plot(ordered, np.arange(1, ordered.size + 1) / ordered.size, label=label)
    axes[1, 0].set_title("Absolute-error cumulative distribution")
    axes[1, 0].set_ylabel("Cumulative fraction")
    axes[1, 0].legend()
    axes[1, 1].boxplot([absolute_before, absolute_after], tick_labels=["before", "after"], showfliers=False)
    axes[1, 1].set_title("Absolute-error boxplot")
    for axis in axes.flat:
        axis.set_xlabel("Original intensity units")
    fig.suptitle("Voxelwise registration error distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_difference_statistics(fixed, signed_before, signed_after, absolute_before, absolute_after, mask, selected, scales, spacing, path):
    z = selected["axial_z"]
    coords = np.argwhere(mask[z])
    if coords.size:
        y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0) + 1
    else:
        y0, x0, y1, x1 = 0, 0, mask.shape[1], mask.shape[2]
    anatomy_low, anatomy_high = np.percentile(fixed, [1, 99])
    signed_norm = TwoSlopeNorm(vmin=scales["signed"][0], vcenter=0, vmax=scales["signed"][1])
    absolute_norm = Normalize(vmin=0, vmax=scales["absolute"][1])
    panels = ((signed_before, "Signed before", "RdBu_r", signed_norm), (signed_after, "Signed after", "RdBu_r", signed_norm), (absolute_before, "Absolute before", "magma", absolute_norm), (absolute_after, "Absolute after", "magma", absolute_norm))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    aspect = spacing[1] / spacing[0]
    for column, (values, title, cmap, norm) in enumerate(panels):
        overlay = np.ma.masked_where(~mask[z], values[z])
        axes[0, column].imshow(fixed[z], cmap="gray", vmin=anatomy_low, vmax=anatomy_high, origin="lower", aspect=aspect)
        axes[0, column].imshow(overlay, cmap=cmap, norm=norm, alpha=0.45, origin="lower", aspect=aspect)
        axes[0, column].set_title(f"{title}; axial z={z}")
        axes[1, column].imshow(fixed[z, y0:y1, x0:x1], cmap="gray", vmin=anatomy_low, vmax=anatomy_high, origin="lower", aspect=aspect)
        axes[1, column].imshow(overlay[y0:y1, x0:x1], cmap=cmap, norm=norm, alpha=0.45, origin="lower", aspect=aspect)
        axes[1, column].set_title("Lung zoom")
        axes[0, column].axis("off"); axes[1, column].axis("off")
    fig.suptitle("Fixed anatomy with matched-scale voxelwise error overlays")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
