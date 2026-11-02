from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lung_ct_fractional_signal_proxy_mpl"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import nibabel as nib
import numpy as np
from scipy.ndimage import convolve1d
import SimpleITK as sitk

from lung_slice_selection import select_lung_visualization_slices
from medical_image_io import summarize_geometry
from project_paths import DEFAULT_OUTPUTS_DIR, display_path
from registration_inputs import sha256_file, validate_registration_image
from version import PIPELINE_VERSION


ANALYSIS_TYPE = "Exploratory registered-space fractional MRI signal-change proxy"
FORMULA = "(registered_moving - fixed) / registered_moving"
KERNEL_SIZE_ZYX = (7, 7, 7)
KERNEL_RADIUS_XYZ = (3, 3, 3)
DEFAULT_GAUSSIAN_SIGMA_MM = 2.5
DEFAULT_GAUSSIAN_TRUNCATE = 1.5
CASE_KEYS = ("case_1", "case_2")
VARIANTS = ("native", "mean_filtered", "gaussian_filtered")
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OUTPUTS_DIR
    / "registration"
    / "case_comparison"
    / "fractional_mri_signal_change_proxy_gaussian_mm_comparison"
)
DEFAULT_PRIOR_PROXY_DIR = (
    DEFAULT_OUTPUTS_DIR
    / "registration"
    / "case_comparison"
    / "fractional_mri_signal_change_proxy"
)
FIGURE_NAMES = {
    "axial": "fractional_proxy_axial_comparison.png",
    "multiplanar": "fractional_proxy_multiplanar_comparison.png",
    "report": "fractional_proxy_report.json",
}
CASE_OUTPUT_NAMES = {
    "native": "native_fractional_proxy.nii.gz",
    "native_validity": "native_validity.nii.gz",
    "mean_filtered": "mean_filtered_fractional_proxy.nii.gz",
    "mean_filtered_validity": "mean_filtered_validity.nii.gz",
    "gaussian_filtered": "gaussian_filtered_fractional_proxy.nii.gz",
    "gaussian_filtered_validity": "gaussian_filtered_validity.nii.gz",
}


def generate_fractional_signal_change_comparison(
    *,
    case_1_dir: str | Path,
    case_2_dir: str | Path,
    case_1_lung_mask: str | Path,
    case_2_lung_mask: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    denominator_threshold: float | None = None,
    denominator_threshold_fraction: float | None = 0.01,
    display_percentile: float = 99.0,
    overlay_opacity: float = 0.62,
    gaussian_sigma_mm: float = DEFAULT_GAUSSIAN_SIGMA_MM,
    gaussian_truncate: float = DEFAULT_GAUSSIAN_TRUNCATE,
    gaussian_support_radius_mm: float | None = None,
    prior_proxy_dir: str | Path = DEFAULT_PRIOR_PROXY_DIR,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate a physical-space Gaussian proxy and a three-way comparison."""
    _validate_options(
        denominator_threshold,
        denominator_threshold_fraction,
        display_percentile,
        overlay_opacity,
        gaussian_sigma_mm,
        gaussian_truncate,
        gaussian_support_radius_mm,
    )
    destination = Path(output_dir).expanduser().resolve(strict=False)
    output_paths = _output_paths(destination)
    existing = [path for path in _all_output_paths(output_paths) if path.exists()]
    if existing and not overwrite and not dry_run:
        raise FileExistsError(
            "Fractional MRI signal-change proxy outputs already exist; use --overwrite."
        )

    context = _load_context(
        case_1_dir=case_1_dir,
        case_2_dir=case_2_dir,
        case_1_lung_mask=case_1_lung_mask,
        case_2_lung_mask=case_2_lung_mask,
    )
    prior = _load_prior_product_record(prior_proxy_dir)
    context["source_paths"].update(prior["protected_paths"])
    hashes_before = _source_hashes(context["source_paths"])
    slice_selection = select_lung_visualization_slices(
        context["cases"]["case_1"]["mask_array"], axial_panel_count=7
    )
    slice_indices = {
        case: dict(slice_selection["central"]) for case in CASE_KEYS
    }
    physical_points = {
        case: list(
            context["cases"][case]["fixed"].image.TransformIndexToPhysicalPoint(
                (
                    slice_indices[case]["sagittal_x"],
                    slice_indices[case]["coronal_y"],
                    slice_indices[case]["axial_z"],
                )
            )
        )
        for case in CASE_KEYS
    }

    results: dict[str, dict[str, Any]] = {}
    pooled_values: list[np.ndarray] = []
    for case in CASE_KEYS:
        item = context["cases"][case]
        fixed_array = sitk.GetArrayFromImage(item["fixed"].image).astype(np.float32)
        registered_array = sitk.GetArrayFromImage(item["registered"].image).astype(np.float32)
        mask = item["mask_array"]
        threshold, threshold_report = _resolve_threshold(
            registered_array,
            mask,
            absolute=denominator_threshold,
            fraction=denominator_threshold_fraction,
        )
        computed = compute_case_proxies(
            fixed_array,
            registered_array,
            mask,
            threshold=threshold,
            kernel_radius_xyz=KERNEL_RADIUS_XYZ,
            spacing_xyz_mm=item["fixed"].image.GetSpacing(),
            gaussian_sigma_mm=gaussian_sigma_mm,
            gaussian_truncate=gaussian_truncate,
            gaussian_support_radius_mm=gaussian_support_radius_mm,
        )
        results[case] = {
            **computed,
            "threshold_report": threshold_report,
        }
        for variant in VARIANTS:
            pooled_values.append(computed[variant][computed[f"{variant}_valid"]])

        if not dry_run:
            case_dir = output_paths[case]["directory"]
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_image(
                output_paths[case]["native"], computed["native"], item["fixed"].image
            )
            _write_image(
                output_paths[case]["native_validity"],
                computed["native_valid"].astype(np.uint8),
                item["fixed"].image,
            )
            _write_image(
                output_paths[case]["mean_filtered"],
                computed["mean_filtered"],
                item["fixed"].image,
            )
            _write_image(
                output_paths[case]["mean_filtered_validity"],
                computed["mean_filtered_valid"].astype(np.uint8),
                item["fixed"].image,
            )
            _write_image(
                output_paths[case]["gaussian_filtered"],
                computed["gaussian_filtered"],
                item["fixed"].image,
            )
            _write_image(
                output_paths[case]["gaussian_filtered_validity"],
                computed["gaussian_filtered_valid"].astype(np.uint8),
                item["fixed"].image,
            )

    shared_limit = compute_shared_symmetric_limit(
        pooled_values, percentile=display_percentile
    )
    anatomy_limits = _pooled_anatomy_limits(
        [context["cases"][case]["fixed"].image for case in CASE_KEYS]
    )
    spacing = context["cases"]["case_1"]["fixed"].image.GetSpacing()
    kernel_report = _kernel_report(spacing)
    gaussian_report = _gaussian_kernel_report(
        spacing,
        sigma_mm=gaussian_sigma_mm,
        truncate=gaussian_truncate,
        support_radius_mm=gaussian_support_radius_mm,
    )
    case_reports = {
        case: _case_report(
            context["cases"][case],
            results[case],
            shared_limit,
            output_paths[case],
        )
        for case in CASE_KEYS
    }
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "analysis_type": ANALYSIS_TYPE,
        "terminology": {
            "recommended_name": ANALYSIS_TYPE,
            "not_validated_fractional_ventilation": True,
            "not_perfusion": True,
            "not_ct_density_change": True,
            "not_clinical_biomarker": True,
        },
        "intended_use": "Research-only, exploratory, non-diagnostic visualization and analysis.",
        "formula": {
            "expression": FORMULA,
            "symbols": {
                "registered_moving": "moving MRI transformed into fixed-image correspondence",
                "fixed": "fixed MRI signal",
            },
            "valid_domain": "exact fixed-space lung mask, finite inputs, and abs(registered-moving denominator) >= configured threshold",
            "machine_epsilon_used": False,
        },
        "filtering": {
            "primary_filtered_experiment": "mask-aware normalized mean filtering of fixed and registered-moving inputs separately, followed by ratio calculation",
            "post_ratio_filtering_performed": False,
            "normalization": "BoxMean(image * native_validity) / BoxMean(native_validity)",
            "invalid_native_voxels_contribute_to_filter": False,
            "filtered_result_restricted_to_native_valid_center_voxels": True,
            "kernel": kernel_report,
        },
        "gaussian_filtering": {
            "experiment": "mask-aware normalized Gaussian filtering of fixed and registered-moving inputs separately, followed by ratio calculation",
            "post_ratio_filtering_performed": False,
            "normalization": "Gaussian(image * native_validity) / Gaussian(native_validity)",
            "invalid_native_voxels_contribute_to_filter": False,
            "filtered_result_restricted_to_native_valid_center_voxels": True,
            "kernel": gaussian_report,
            "comparison_intent": "Physical-mm Gaussian chosen for support comparable to the 7x7x7 box on the real grid; the weights are not mathematically identical to a box filter.",
        },
        "mask_policy": {
            "calculation_domain": "exact binary fixed-space combined-lung mask",
            "outside_lung_analytical_representation": "NaN",
            "invalid_denominator_analytical_representation": "NaN",
            "validity_maps_saved": True,
            "display_outside_lung": "grayscale fixed anatomy without colored proxy",
            "mask_changed_or_resampled": False,
        },
        "inputs": {
            case: context["cases"][case]["input_report"] for case in CASE_KEYS
        },
        "geometry_validation": context["geometry_report"],
        "role_assignments": {
            case: context["cases"][case]["role_assignment"] for case in CASE_KEYS
        },
        "denominator_policy": {
            "definition": "valid iff voxel is in the exact lung mask, F and M_R are finite, and abs(M_R) >= threshold",
            "threshold_mode": (
                "absolute" if denominator_threshold is not None else "fraction_of_native_registered_lung_median_absolute_signal"
            ),
            "configured_absolute_threshold": denominator_threshold,
            "configured_fraction": denominator_threshold_fraction,
            "thresholds_by_case": {
                case: results[case]["threshold_report"] for case in CASE_KEYS
            },
        },
        "cases": case_reports,
        "slice_selection": {
            "policy": slice_selection["selection"],
            "indices_numpy_zyx": slice_indices,
            "physical_points_xyz_mm": physical_points,
            "identical_indices_across_cases": slice_indices["case_1"] == slice_indices["case_2"],
        },
        "display": {
            "anatomical_background": "fixed MRI grayscale everywhere",
            "colored_domain": "valid proxy voxels inside exact fixed-space lung mask only",
            "colormap": "RdBu_r",
            "zero_centered_diverging": True,
            "overlay_opacity": float(overlay_opacity),
            "anatomy_limits": list(anatomy_limits),
            "shared_symmetric_window": {
                "method": "linear percentile of pooled absolute valid values from both cases and all three variants",
                "pooled_order": [
                    "case_1_native",
                    "case_1_mean_filtered",
                    "case_1_gaussian_filtered",
                    "case_2_native",
                    "case_2_mean_filtered",
                    "case_2_gaussian_filtered",
                ],
                "percentile": float(display_percentile),
                "pooled_valid_voxel_count": int(sum(values.size for values in pooled_values)),
                "vmin": -shared_limit,
                "vcenter": 0.0,
                "vmax": shared_limit,
                "identical_for_all_proxy_panels": True,
            },
            "previous_four_map_window_for_reference": prior["old_window"],
            "physical_aspect_ratios_preserved": True,
            "lung_mask_contour": True,
        },
        "outputs": _display_output_paths(output_paths),
        "input_immutability": {
            "checksums_verified_before_and_after": False if dry_run else None,
            "unchanged": None,
            "sha256_before": hashes_before,
        },
        "output_checksums_sha256": None,
        "warnings": [
            "This is an exploratory MRI signal-change proxy, not validated fractional ventilation, perfusion, CT density change, or a clinical biomarker.",
            "Fixed/moving assignments were inferred and are not researcher-confirmed.",
            "MRI intensity calibration, respiratory phase ordering, and cross-acquisition intensity comparability remain unconfirmed.",
            "Registration and trilinear interpolation contribute to registered-moving intensities.",
            "The relative denominator threshold is scale-aware but is not physiologically validated.",
            "Mean filtering changes effective spatial resolution and may suppress localized registration errors.",
            "Gaussian filtering changes effective spatial resolution and may suppress localized registration errors.",
            "The default Gaussian sigma of 2.5 mm is an exploratory weighting choice selected for support comparability, not physiological validation.",
        ],
        "completion_status": "dry-run" if dry_run else "pending",
        "writes_performed": False if dry_run else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        return report

    destination.mkdir(parents=True, exist_ok=True)
    _render_axial(
        context,
        results,
        slice_indices,
        anatomy_limits,
        shared_limit,
        overlay_opacity,
        kernel_report,
        gaussian_report,
        report["denominator_policy"],
        output_paths["figures"]["axial"],
    )
    _render_multiplanar(
        context,
        results,
        slice_indices,
        anatomy_limits,
        shared_limit,
        overlay_opacity,
        kernel_report,
        gaussian_report,
        report["denominator_policy"],
        output_paths["figures"]["multiplanar"],
    )
    hashes_after = _source_hashes(context["source_paths"])
    if hashes_after != hashes_before:
        raise RuntimeError("A source NIfTI or provenance manifest changed during analysis.")
    report["input_immutability"] = {
        "checksums_verified_before_and_after": True,
        "unchanged": True,
        "sha256_before": hashes_before,
        "sha256_after": hashes_after,
    }
    report["output_checksums_sha256"] = {
        display_path(path): sha256_file(path)
        for path in _all_output_paths(output_paths)
        if path != output_paths["figures"]["report"]
    }
    report["completion_status"] = "complete"
    report["writes_performed"] = True
    output_paths["figures"]["report"].write_text(json.dumps(report, indent=2) + "\n")
    return report


def compute_case_proxies(
    fixed: np.ndarray,
    registered: np.ndarray,
    lung_mask: np.ndarray,
    *,
    threshold: float,
    kernel_radius_xyz: tuple[int, int, int] = KERNEL_RADIUS_XYZ,
    spacing_xyz_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    gaussian_sigma_mm: float = DEFAULT_GAUSSIAN_SIGMA_MM,
    gaussian_truncate: float = DEFAULT_GAUSSIAN_TRUNCATE,
    gaussian_support_radius_mm: float | None = None,
) -> dict[str, Any]:
    fixed_array = np.asarray(fixed, dtype=np.float32)
    registered_array = np.asarray(registered, dtype=np.float32)
    mask = np.asarray(lung_mask, dtype=bool)
    if fixed_array.shape != registered_array.shape or fixed_array.shape != mask.shape:
        raise ValueError("Fixed, registered-moving, and lung-mask arrays must have identical shapes.")
    if fixed_array.ndim != 3:
        raise ValueError("Fractional proxy inputs must be 3D.")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Denominator threshold must be finite and greater than zero.")
    finite = np.isfinite(fixed_array) & np.isfinite(registered_array)
    native_valid = mask & finite & (np.abs(registered_array) >= threshold)
    native = _ratio(fixed_array, registered_array, native_valid)

    filtered_fixed, support = masked_normalized_box_mean(
        fixed_array, native_valid, kernel_radius_xyz=kernel_radius_xyz
    )
    filtered_registered, registered_support = masked_normalized_box_mean(
        registered_array, native_valid, kernel_radius_xyz=kernel_radius_xyz
    )
    if not np.array_equal(support, registered_support):
        raise RuntimeError("Fixed and registered filtering support unexpectedly differs.")
    mean_filtered_valid = (
        native_valid
        & (support > 0)
        & np.isfinite(filtered_fixed)
        & np.isfinite(filtered_registered)
        & (np.abs(filtered_registered) >= threshold)
    )
    mean_filtered = _ratio(filtered_fixed, filtered_registered, mean_filtered_valid)
    valid_support = support[mean_filtered_valid]
    mean_support_report = {
        "minimum": int(valid_support.min()) if valid_support.size else None,
        "median": float(np.median(valid_support)) if valid_support.size else None,
        "maximum": int(valid_support.max()) if valid_support.size else None,
        "full_kernel_voxel_count": int(np.prod(KERNEL_SIZE_ZYX)),
    }
    del filtered_fixed, filtered_registered, support, registered_support, valid_support

    sigma_voxels_xyz = sigma_mm_to_voxels(gaussian_sigma_mm, spacing_xyz_mm)
    gaussian_radius_xyz = gaussian_kernel_radius_xyz(
        sigma_voxels_xyz,
        truncate=gaussian_truncate,
        support_radius_mm=gaussian_support_radius_mm,
        spacing_xyz_mm=spacing_xyz_mm,
    )
    gaussian_fixed, gaussian_weight = masked_normalized_gaussian_mean(
        fixed_array,
        native_valid,
        sigma_voxels_xyz=sigma_voxels_xyz,
        kernel_radius_xyz=gaussian_radius_xyz,
    )
    gaussian_registered, registered_gaussian_weight = masked_normalized_gaussian_mean(
        registered_array,
        native_valid,
        sigma_voxels_xyz=sigma_voxels_xyz,
        kernel_radius_xyz=gaussian_radius_xyz,
    )
    if not np.array_equal(gaussian_weight, registered_gaussian_weight):
        raise RuntimeError("Fixed and registered Gaussian support unexpectedly differs.")
    gaussian_filtered_valid = (
        native_valid
        & (gaussian_weight > 0)
        & np.isfinite(gaussian_fixed)
        & np.isfinite(gaussian_registered)
        & (np.abs(gaussian_registered) >= threshold)
    )
    gaussian_filtered = _ratio(
        gaussian_fixed, gaussian_registered, gaussian_filtered_valid
    )
    valid_gaussian_weight = gaussian_weight[gaussian_filtered_valid]
    gaussian_support_report = {
        "minimum": float(valid_gaussian_weight.min()) if valid_gaussian_weight.size else None,
        "median": float(np.median(valid_gaussian_weight)) if valid_gaussian_weight.size else None,
        "maximum": float(valid_gaussian_weight.max()) if valid_gaussian_weight.size else None,
        "full_kernel_weight": 1.0,
    }
    native_invalid = mask & ~native_valid
    gaussian_insufficient_support = native_valid & ~(gaussian_weight > 0)
    gaussian_filtered_invalid = (
        native_valid
        & (gaussian_weight > 0)
        & (
            ~np.isfinite(gaussian_fixed)
            | ~np.isfinite(gaussian_registered)
            | (np.abs(gaussian_registered) < threshold)
        )
    )
    return {
        "native": native,
        "native_valid": native_valid,
        "mean_filtered": mean_filtered,
        "mean_filtered_valid": mean_filtered_valid,
        "mean_support_report": mean_support_report,
        "gaussian_filtered": gaussian_filtered,
        "gaussian_filtered_valid": gaussian_filtered_valid,
        "gaussian_support_report": gaussian_support_report,
        "invalid_counts": {
            "native_invalid_lung_voxels": int(np.count_nonzero(native_invalid)),
            "gaussian_filtered_invalid_after_native_validity": int(
                np.count_nonzero(gaussian_filtered_invalid)
            ),
            "gaussian_insufficient_positive_support": int(
                np.count_nonzero(gaussian_insufficient_support)
            ),
            "gaussian_final_valid": int(np.count_nonzero(gaussian_filtered_valid)),
        },
    }


def masked_normalized_box_mean(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    kernel_radius_xyz: tuple[int, int, int] = KERNEL_RADIUS_XYZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalized box filtering without contributions from invalid voxels."""
    array = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if array.shape != valid.shape or array.ndim != 3:
        raise ValueError("Box-filter values and validity mask must be identical 3D arrays.")
    if len(kernel_radius_xyz) != 3 or any(int(radius) < 0 for radius in kernel_radius_xyz):
        raise ValueError("Kernel radii must be three nonnegative integers in x,y,z order.")
    reference = sitk.GetImageFromArray(np.where(valid, array, 0.0).astype(np.float32))
    mask_image = sitk.GetImageFromArray(valid.astype(np.float32))
    radius = [int(value) for value in kernel_radius_xyz]
    # Explicit constant padding avoids SimpleITK's zero-flux boundary
    # condition duplicating edge voxels. The same padding is applied to the
    # weighted signal and support, so normalized convolution remains exact.
    padded_reference = sitk.ConstantPad(reference, radius, radius, 0.0)
    padded_mask = sitk.ConstantPad(mask_image, radius, radius, 0.0)
    weighted_mean_padded = sitk.BoxMean(padded_reference, radius)
    support_fraction_padded = sitk.BoxMean(padded_mask, radius)
    size = list(reference.GetSize())
    weighted_mean = sitk.RegionOfInterest(weighted_mean_padded, size, radius)
    support_fraction = sitk.RegionOfInterest(support_fraction_padded, size, radius)
    numerator = sitk.GetArrayFromImage(weighted_mean).astype(np.float32)
    support_fraction_array = sitk.GetArrayFromImage(support_fraction).astype(np.float32)
    result = np.full(array.shape, np.nan, dtype=np.float32)
    supported = support_fraction_array > 0
    result[supported] = numerator[supported] / support_fraction_array[supported]
    kernel_voxels = int(np.prod([2 * value + 1 for value in radius]))
    support_count = np.rint(support_fraction_array * kernel_voxels).astype(np.uint16)
    return result, support_count


def masked_normalized_gaussian_mean(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sigma_voxels_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    kernel_radius_xyz: tuple[int, int, int] = KERNEL_RADIUS_XYZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a finite, separable normalized Gaussian without invalid contributions."""
    array = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if array.shape != valid.shape or array.ndim != 3:
        raise ValueError("Gaussian-filter values and validity mask must be identical 3D arrays.")
    if len(sigma_voxels_xyz) != 3 or any(
        not math.isfinite(float(value)) or float(value) <= 0
        for value in sigma_voxels_xyz
    ):
        raise ValueError("Gaussian voxel sigmas must be three finite positive values.")
    if len(kernel_radius_xyz) != 3 or any(int(radius) < 0 for radius in kernel_radius_xyz):
        raise ValueError("Kernel radii must be three nonnegative integers in x,y,z order.")

    numerator = np.where(valid, array, 0.0).astype(np.float32)
    denominator = valid.astype(np.float32)
    # NumPy arrays are z,y,x, while the public radii are x,y,z.
    for axis, (radius, sigma_voxels) in enumerate(
        zip(reversed(kernel_radius_xyz), reversed(sigma_voxels_xyz))
    ):
        offsets = np.arange(-int(radius), int(radius) + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (offsets / float(sigma_voxels)) ** 2)
        weights /= weights.sum()
        numerator = convolve1d(
            numerator, weights, axis=axis, mode="constant", cval=0.0
        )
        denominator = convolve1d(
            denominator, weights, axis=axis, mode="constant", cval=0.0
        )
    result = np.full(array.shape, np.nan, dtype=np.float32)
    supported = denominator > 0
    result[supported] = numerator[supported] / denominator[supported]
    return result, denominator.astype(np.float32, copy=False)


def sigma_mm_to_voxels(
    sigma_mm: float, spacing_xyz_mm: Sequence[float]
) -> tuple[float, float, float]:
    """Convert one isotropic physical sigma to axis-specific voxel sigmas."""
    if not math.isfinite(float(sigma_mm)) or float(sigma_mm) <= 0:
        raise ValueError("Gaussian sigma in millimeters must be finite and positive.")
    if len(spacing_xyz_mm) != 3 or any(
        not math.isfinite(float(value)) or float(value) <= 0
        for value in spacing_xyz_mm
    ):
        raise ValueError("Voxel spacing must contain three finite positive values.")
    return tuple(float(sigma_mm) / float(spacing) for spacing in spacing_xyz_mm)


def gaussian_kernel_radius_xyz(
    sigma_voxels_xyz: Sequence[float],
    *,
    truncate: float,
    support_radius_mm: float | None,
    spacing_xyz_mm: Sequence[float],
) -> tuple[int, int, int]:
    """Return discrete axis radii from truncation or an explicit physical radius."""
    if not math.isfinite(float(truncate)) or float(truncate) <= 0:
        raise ValueError("Gaussian truncation must be finite and positive.")
    if support_radius_mm is not None:
        if not math.isfinite(float(support_radius_mm)) or float(support_radius_mm) <= 0:
            raise ValueError("Gaussian support radius in millimeters must be finite and positive.")
        return tuple(
            max(1, int(math.ceil(float(support_radius_mm) / float(spacing))))
            for spacing in spacing_xyz_mm
        )
    return tuple(max(1, int(math.ceil(float(truncate) * float(sigma)))) for sigma in sigma_voxels_xyz)


def compute_shared_symmetric_limit(
    value_sets: Sequence[np.ndarray], *, percentile: float = 99.0
) -> float:
    if not 0 < percentile <= 100:
        raise ValueError("Display percentile must be in (0, 100].")
    if not value_sets:
        raise ValueError("At least one valid value set is required.")
    flattened = []
    for values in value_sets:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("Shared-window inputs must be non-empty and finite.")
        flattened.append(np.abs(array))
    limit = float(np.percentile(np.concatenate(flattened), percentile, method="linear"))
    return limit if limit > 0 else 1.0


def _ratio(fixed, registered, valid):
    result = np.full(fixed.shape, np.nan, dtype=np.float32)
    calculated = (
        registered[valid].astype(np.float64) - fixed[valid].astype(np.float64)
    ) / registered[valid].astype(np.float64)
    if not np.isfinite(calculated).all():
        raise ValueError("Fractional proxy produced nonfinite values on the valid domain.")
    result[valid] = calculated.astype(np.float32)
    return result


def _resolve_threshold(registered, mask, *, absolute, fraction):
    finite_values = np.abs(registered[mask & np.isfinite(registered)].astype(np.float64))
    if finite_values.size == 0:
        raise ValueError("No finite registered-moving lung voxels are available.")
    median = float(np.median(finite_values))
    if absolute is not None:
        threshold = float(absolute)
        mode = "absolute"
        definition = "user-supplied absolute MRI signal threshold"
    else:
        threshold = float(fraction) * median
        mode = "fraction_of_native_registered_lung_median_absolute_signal"
        definition = "configured fraction multiplied by median(abs(M_R)) over finite voxels in the exact fixed-space lung mask"
    return threshold, {
        "mode": mode,
        "definition": definition,
        "value": threshold,
        "configured_fraction": fraction if absolute is None else None,
        "native_registered_lung_median_absolute_signal": median,
        "comparison": "abs(M_R) >= threshold",
    }


def _load_context(*, case_1_dir, case_2_dir, case_1_lung_mask, case_2_lung_mask):
    case_dirs = {
        "case_1": Path(case_1_dir).expanduser().resolve(strict=False),
        "case_2": Path(case_2_dir).expanduser().resolve(strict=False),
    }
    mask_paths = {
        "case_1": Path(case_1_lung_mask).expanduser().resolve(strict=False),
        "case_2": Path(case_2_lung_mask).expanduser().resolve(strict=False),
    }
    cases = {}
    source_paths = {}
    for case in CASE_KEYS:
        directory = case_dirs[case]
        manifest_path = directory / "full/registration_manifest.json"
        preprocessing_path = directory / "full/preprocessed/preprocessing_manifest.json"
        mask_manifest_path = mask_paths[case].parents[1] / "manifest.json"
        manifest = _read_json(manifest_path, f"{case} registration manifest")
        preprocessing = _read_json(preprocessing_path, f"{case} preprocessing manifest")
        mask_manifest = _read_json(mask_manifest_path, f"{case} lung-mask manifest")
        if manifest.get("completion_status") != "complete" or manifest.get("experiment") != "full":
            raise ValueError(f"{case} registration manifest is not a completed full registration.")
        if preprocessing.get("completion_status") != "complete":
            raise ValueError(f"{case} preprocessing manifest is not complete.")
        if mask_manifest.get("completion_status") != "success":
            raise ValueError(f"{case} lung-mask manifest is not successful.")
        paths = {
            "fixed": directory / "full/preprocessed/fixed.nii.gz",
            "moving": directory / "full/preprocessed/moving.nii.gz",
            "registered": directory / "full/deformable/moving_registered.nii.gz",
            "lung_mask": mask_paths[case],
        }
        loaded = {
            role: validate_registration_image(path, role=f"{case} {role}")
            for role, path in paths.items()
        }
        geometry_mismatches = {}
        for role in ("registered", "lung_mask"):
            mismatches = _exact_geometry_mismatches(loaded["fixed"].image, loaded[role].image)
            geometry_mismatches[f"fixed_vs_{role}"] = mismatches
            if mismatches:
                raise ValueError(
                    f"{case} fixed/{role} exact geometry mismatch: " + "; ".join(mismatches)
                )
        mask_array = sitk.GetArrayFromImage(loaded["lung_mask"].image)
        unique = set(float(value) for value in np.unique(mask_array))
        if not unique.issubset({0.0, 1.0}) or not np.any(mask_array):
            raise ValueError(f"{case} lung mask must be nonempty and binary.")
        expected_fixed = str(mask_manifest.get("input_path", ""))
        if expected_fixed:
            if Path(expected_fixed).is_absolute():
                provenance_matches = Path(expected_fixed).resolve(strict=False) == paths["fixed"].resolve(strict=False)
            else:
                provenance_matches = display_path(paths["fixed"]) == expected_fixed
            if not provenance_matches:
                raise ValueError(f"{case} lung-mask provenance does not point to the selected fixed image.")
        input_report = {
            "registration_manifest": {"path": display_path(manifest_path), "sha256": sha256_file(manifest_path)},
            "preprocessing_manifest": {"path": display_path(preprocessing_path), "sha256": sha256_file(preprocessing_path)},
            "lung_mask_manifest": {"path": display_path(mask_manifest_path), "sha256": sha256_file(mask_manifest_path)},
            "fixed": {"path": display_path(paths["fixed"]), "sha256": sha256_file(paths["fixed"])},
            "original_moving_not_used_for_voxelwise_proxy": {"path": display_path(paths["moving"]), "sha256": sha256_file(paths["moving"])},
            "registered_moving": {"path": display_path(paths["registered"]), "sha256": sha256_file(paths["registered"])},
            "exact_fixed_space_lung_mask": {"path": display_path(paths["lung_mask"]), "sha256": sha256_file(paths["lung_mask"])},
            "lung_mask_provenance": {
                "modality": mask_manifest.get("modality"),
                "task": mask_manifest.get("task"),
                "totalsegmentator_version": mask_manifest.get("totalsegmentator_version"),
                "selected_lung_labels": mask_manifest.get("selected_lung_labels"),
            },
        }
        cases[case] = {
            **loaded,
            "paths": paths,
            "mask_array": mask_array.astype(bool),
            "input_report": input_report,
            "role_assignment": preprocessing.get("role_assignment"),
            "geometry_mismatches": geometry_mismatches,
        }
        for name, path in {
            "registration_manifest": manifest_path,
            "preprocessing_manifest": preprocessing_path,
            "lung_mask_manifest": mask_manifest_path,
            **paths,
        }.items():
            source_paths[f"{case}:{name}"] = path

    cross = _exact_geometry_mismatches(cases["case_1"]["fixed"].image, cases["case_2"]["fixed"].image)
    if cross:
        raise ValueError("Case 1/Case 2 fixed grids must agree exactly: " + "; ".join(cross))
    geometry_report = {
        case: {
            "exact_fixed_registered_mask_grid": True,
            "mismatches": cases[case]["geometry_mismatches"],
            "fixed": _geometry_dict(cases[case]["fixed"].image),
            "registered_moving": _geometry_dict(cases[case]["registered"].image),
            "lung_mask": _geometry_dict(cases[case]["lung_mask"].image),
        }
        for case in CASE_KEYS
    }
    geometry_report["cross_case_fixed_grid"] = {
        "exact_agreement": True,
        "mismatches": [],
    }
    return {
        "cases": cases,
        "source_paths": source_paths,
        "geometry_report": geometry_report,
    }


def _case_report(item, result, shared_limit, paths):
    mask = item["mask_array"]
    variants = {}
    for variant in VARIANTS:
        valid = result[f"{variant}_valid"]
        values = result[variant][valid]
        negative_clipped = int(np.count_nonzero(values < -shared_limit))
        positive_clipped = int(np.count_nonzero(values > shared_limit))
        clipped = negative_clipped + positive_clipped
        variants[variant] = {
            "statistics": _statistics(values),
            "counts": {
                "lung_voxels": int(np.count_nonzero(mask)),
                "valid_voxels": int(np.count_nonzero(valid)),
                "invalid_lung_voxels": int(np.count_nonzero(mask & ~valid)),
                "outside_lung_voxels": int(mask.size - np.count_nonzero(mask)),
            },
            "clipping": {
                "definition": "valid proxy voxels with abs(value) strictly greater than shared vmax",
                "count": clipped,
                "negative_count": negative_clipped,
                "positive_count": positive_clipped,
                "fraction_of_valid": float(clipped / values.size) if values.size else 0.0,
            },
            "output": display_path(paths[variant]),
            "validity_output": display_path(paths[f"{variant}_validity"]),
        }
    variants["mean_filtered"]["filter_support_count"] = result["mean_support_report"]
    variants["gaussian_filtered"]["normalized_gaussian_support_weight"] = result[
        "gaussian_support_report"
    ]
    return {
        "threshold": result["threshold_report"],
        "invalid_data_counts": result["invalid_counts"],
        "variants": variants,
    }


def _load_prior_product_record(prior_proxy_dir):
    directory = Path(prior_proxy_dir).expanduser().resolve(strict=False)
    report_path = directory / FIGURE_NAMES["report"]
    report = _read_json(report_path, "prior native/mean proxy report")
    if report.get("completion_status") != "complete":
        raise ValueError("Prior native/mean proxy report is not complete.")
    old_window = report.get("display", {}).get("shared_symmetric_window")
    if not isinstance(old_window, dict):
        raise ValueError("Prior proxy report does not contain its four-map display window.")
    protected = {"prior_proxy_report": report_path}
    for case in CASE_KEYS:
        outputs = report.get("outputs", {}).get("cases", {}).get(case, {})
        for key in ("native", "native_validity", "mean_filtered", "mean_filtered_validity"):
            raw_path = outputs.get(key)
            if not raw_path:
                raise ValueError(f"Prior proxy report is missing {case} {key} output.")
            path = Path(raw_path)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve(strict=False)
            if not path.is_file():
                raise FileNotFoundError(f"Prior analytical product does not exist: {path}")
            protected[f"prior:{case}:{key}"] = path
    return {
        "directory": directory,
        "old_window": old_window,
        "protected_paths": protected,
    }


def _statistics(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("mean", "median", "standard_deviation", "minimum", "maximum", "positive_fraction", "negative_fraction", "zero_fraction", "percentiles")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "positive_fraction": float(np.count_nonzero(array > 0) / array.size),
        "negative_fraction": float(np.count_nonzero(array < 0) / array.size),
        "zero_fraction": float(np.count_nonzero(array == 0) / array.size),
        "percentiles": {
            str(value): float(np.percentile(array, value)) for value in (1, 5, 25, 50, 75, 95, 99)
        },
    }


def _render_axial(
    context, results, indices, anatomy_limits, limit, opacity, kernel,
    gaussian, policy, path
):
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), layout="constrained")
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    last = None
    for row, case in enumerate(CASE_KEYS):
        item = context["cases"][case]
        fixed = sitk.GetArrayViewFromImage(item["fixed"].image)
        z = indices[case]["axial_z"]
        aspect = _aspect(item["fixed"].image, "axial")
        mask_slice = item["mask_array"][z]
        axes[row, 0].imshow(fixed[z], cmap="gray", vmin=anatomy_limits[0], vmax=anatomy_limits[1], origin="lower", aspect=aspect)
        _draw_contour(axes[row, 0], mask_slice)
        axes[row, 0].set_title(f"Case {row + 1} — fixed anatomy\naxial z={z}")
        axes[row, 0].axis("off")
        for column, variant in enumerate(VARIANTS, start=1):
            axis = axes[row, column]
            axis.imshow(fixed[z], cmap="gray", vmin=anatomy_limits[0], vmax=anatomy_limits[1], origin="lower", aspect=aspect)
            valid_slice = results[case][f"{variant}_valid"][z]
            value_slice = results[case][variant][z]
            last = axis.imshow(np.ma.masked_where(~valid_slice, value_slice), cmap="RdBu_r", norm=norm, alpha=opacity, origin="lower", aspect=aspect)
            _draw_contour(axis, mask_slice)
            label = {
                "native": "native",
                "mean_filtered": "7×7×7 mean-filtered inputs → ratio",
                "gaussian_filtered": (
                    f"Gaussian inputs → ratio (σ={gaussian['sigma_mm']:g} mm; "
                    f"{'×'.join(map(str, gaussian['size_voxels_xyz']))})"
                ),
            }[variant]
            axis.set_title(f"Case {row + 1} — {label}\naxial z={z}")
            axis.axis("off")
    threshold_text = _threshold_caption(policy)
    figure.suptitle(
        "Research-only fractional MRI signal-change proxy — not validated ventilation\n"
        f"P=(M_R−F)/M_R; {threshold_text}\n"
        f"7×7×7 mask-aware input filters ({_kernel_caption(kernel)}); "
        f"Gaussian σ={gaussian['sigma_mm']:g} mm, trunc={gaussian['truncation_sigma']:g}σ, "
        f"shared limits [−{limit:.6g}, +{limit:.6g}]",
        fontsize=11,
    )
    figure.colorbar(last, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.78, pad=0.025, label="Fractional MRI signal-change proxy (signed; not validated ventilation)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_multiplanar(
    context, results, indices, anatomy_limits, limit, opacity, kernel,
    gaussian, policy, path
):
    views = ("axial", "coronal", "sagittal")
    figure, axes = plt.subplots(6, 4, figsize=(17, 22), layout="constrained")
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    last = None
    for case_index, case in enumerate(CASE_KEYS):
        item = context["cases"][case]
        fixed = sitk.GetArrayViewFromImage(item["fixed"].image)
        for view_index, view in enumerate(views):
            row = case_index * 3 + view_index
            aspect = _aspect(item["fixed"].image, view)
            fixed_slice = _extract_slice(fixed, indices[case], view)
            mask_slice = _extract_slice(item["mask_array"], indices[case], view)
            axes[row, 0].imshow(fixed_slice, cmap="gray", vmin=anatomy_limits[0], vmax=anatomy_limits[1], origin="lower", aspect=aspect)
            _draw_contour(axes[row, 0], mask_slice)
            axes[row, 0].set_title(f"Case {case_index + 1} — {view} fixed")
            axes[row, 0].axis("off")
            for column, variant in enumerate(VARIANTS, start=1):
                axis = axes[row, column]
                axis.imshow(fixed_slice, cmap="gray", vmin=anatomy_limits[0], vmax=anatomy_limits[1], origin="lower", aspect=aspect)
                values = _extract_slice(results[case][variant], indices[case], view)
                valid = _extract_slice(results[case][f"{variant}_valid"], indices[case], view)
                last = axis.imshow(np.ma.masked_where(~valid, values), cmap="RdBu_r", norm=norm, alpha=opacity, origin="lower", aspect=aspect)
                _draw_contour(axis, mask_slice)
                label = {
                    "native": "native",
                    "mean_filtered": "mean inputs → ratio",
                    "gaussian_filtered": (
                        f"Gaussian inputs → ratio (σ={gaussian['sigma_mm']:g} mm)"
                    ),
                }[variant]
                axis.set_title(f"Case {case_index + 1} — {view} {label}")
                axis.axis("off")
    figure.suptitle(
        "Research-only fractional MRI signal-change proxy — not validated ventilation\n"
        f"P=(M_R−F)/M_R; {_threshold_caption(policy)}\n"
        f"7×7×7 mask-aware input filters ({_kernel_caption(kernel)}); "
        f"Gaussian σ={gaussian['sigma_mm']:g} mm, trunc={gaussian['truncation_sigma']:g}σ, "
        f"shared limits [−{limit:.6g}, +{limit:.6g}]",
        fontsize=11,
    )
    figure.colorbar(last, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.75, pad=0.015, label="Fractional MRI signal-change proxy (signed; not validated ventilation)")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _draw_contour(axis, mask):
    if np.any(mask) and not np.all(mask):
        axis.contour(mask.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.55, origin="lower")


def _threshold_caption(policy):
    if policy["threshold_mode"] == "absolute":
        return f"|M_R| ≥ {policy['configured_absolute_threshold']:g}"
    return f"|M_R| ≥ {policy['configured_fraction']:g}×case median(|M_R|)"


def _kernel_caption(kernel):
    return "×".join(f"{value:.3f}" for value in kernel["voxel_cell_support_mm_zyx"]) + " mm"


def _kernel_report(spacing_xyz):
    spacing_zyx = tuple(reversed(tuple(float(value) for value in spacing_xyz)))
    support = [float(size * spacing) for size, spacing in zip(KERNEL_SIZE_ZYX, spacing_zyx)]
    center_span = [float((size - 1) * spacing) for size, spacing in zip(KERNEL_SIZE_ZYX, spacing_zyx)]
    return {
        "type": "uniform box mean",
        "size_voxels_zyx": list(KERNEL_SIZE_ZYX),
        "radius_voxels_xyz": list(KERNEL_RADIUS_XYZ),
        "voxel_count": int(np.prod(KERNEL_SIZE_ZYX)),
        "voxel_cell_support_mm_zyx": support,
        "first_to_last_center_span_mm_zyx": center_span,
        "spacing_mm_zyx": list(spacing_zyx),
    }


def _gaussian_kernel_report(
    spacing_xyz, *, sigma_mm, truncate, support_radius_mm
):
    spacing_xyz = tuple(float(value) for value in spacing_xyz)
    spacing_zyx = tuple(reversed(spacing_xyz))
    sigma_voxels_xyz = sigma_mm_to_voxels(sigma_mm, spacing_xyz)
    radius_xyz = gaussian_kernel_radius_xyz(
        sigma_voxels_xyz,
        truncate=truncate,
        support_radius_mm=support_radius_mm,
        spacing_xyz_mm=spacing_xyz,
    )
    weights_xyz = []
    for radius, sigma_voxels in zip(radius_xyz, sigma_voxels_xyz):
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (offsets / sigma_voxels) ** 2)
        weights /= weights.sum()
        weights_xyz.append([float(value) for value in weights])
    size_xyz = tuple(2 * radius + 1 for radius in radius_xyz)
    support_xyz = tuple(size * spacing for size, spacing in zip(size_xyz, spacing_xyz))
    center_span_xyz = tuple((size - 1) * spacing for size, spacing in zip(size_xyz, spacing_xyz))
    return {
        "type": "finite separable Gaussian",
        "sigma_mm": float(sigma_mm),
        "sigma_voxels_xyz": list(sigma_voxels_xyz),
        "truncation_sigma": float(truncate),
        "support_policy": (
            "explicit_physical_radius"
            if support_radius_mm is not None
            else "ceil(truncation_sigma * sigma_voxels) independently per axis"
        ),
        "configured_support_radius_mm": support_radius_mm,
        "radius_voxels_xyz": list(radius_xyz),
        "size_voxels_xyz": list(size_xyz),
        "size_voxels_zyx": list(reversed(size_xyz)),
        "spacing_mm_xyz": list(spacing_xyz),
        "one_dimensional_normalized_weights_xyz": weights_xyz,
        "voxel_cell_support_mm_xyz": list(support_xyz),
        "voxel_cell_support_mm_zyx": list(reversed(support_xyz)),
        "first_to_last_center_span_mm_xyz": list(center_span_xyz),
        "first_to_last_center_span_mm_zyx": list(reversed(center_span_xyz)),
        "boundary_policy": "constant zero outside the array followed by validity-weight normalization",
        "reproducible_definition": "For each axis, w(d)=exp(-0.5*(d/sigma_voxels_axis)^2), sampled at integer offsets through the reported radius and normalized to unit 1-D sum; separable 3-D convolution is normalized by the identically filtered native-validity mask.",
    }


def _pooled_anatomy_limits(images):
    samples = []
    for image in images:
        array = sitk.GetArrayViewFromImage(image)
        samples.append(np.asarray(array).reshape(-1)[:: max(1, array.size // 500_000)])
    low, high = np.percentile(np.concatenate(samples), [1, 99])
    if low == high:
        return float(low - 0.5), float(high + 0.5)
    return float(low), float(high)


def _output_paths(destination):
    result = {"figures": {name: destination / filename for name, filename in FIGURE_NAMES.items()}}
    for case in CASE_KEYS:
        case_dir = destination / case
        result[case] = {"directory": case_dir, **{name: case_dir / filename for name, filename in CASE_OUTPUT_NAMES.items()}}
    return result


def _all_output_paths(paths):
    result = list(paths["figures"].values())
    for case in CASE_KEYS:
        result.extend(path for key, path in paths[case].items() if key != "directory")
    return result


def _display_output_paths(paths):
    return {
        "figures": {name: display_path(path) for name, path in paths["figures"].items()},
        "cases": {
            case: {name: display_path(path) for name, path in paths[case].items() if name != "directory"}
            for case in CASE_KEYS
        },
    }


def _write_image(path, array, reference):
    values = np.asarray(array)
    if np.issubdtype(values.dtype, np.floating):
        # The SimpleITK NIfTI writer can convert IEEE NaNs to zero. Establish
        # the exact reference transform first, then replace only the payload.
        placeholder = sitk.GetImageFromArray(np.zeros(values.shape, dtype=np.float32))
        placeholder.CopyInformation(reference)
        sitk.WriteImage(placeholder, str(path), True)
        geometry = nib.load(str(path))
        header = geometry.header.copy()
        header.set_data_dtype(np.float32)
        payload_xyz = np.transpose(values.astype(np.float32, copy=False), (2, 1, 0))
        result = nib.Nifti1Image(payload_xyz, geometry.affine, header=header)
        result.set_qform(geometry.get_qform(), int(geometry.header["qform_code"]))
        result.set_sform(geometry.get_sform(), int(geometry.header["sform_code"]))
        nib.save(result, str(path))
    else:
        image = sitk.GetImageFromArray(values)
        image.CopyInformation(reference)
        sitk.WriteImage(image, str(path), True)


def _source_hashes(paths):
    return {name: sha256_file(path) for name, path in paths.items()}


def _read_json(path, role):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{role} does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is unreadable or malformed: {path}") from exc


def _exact_geometry_mismatches(first, second):
    mismatches = []
    for name, left, right in (
        ("size", tuple(first.GetSize()), tuple(second.GetSize())),
        ("spacing", tuple(first.GetSpacing()), tuple(second.GetSpacing())),
        ("origin", tuple(first.GetOrigin()), tuple(second.GetOrigin())),
        ("direction", tuple(first.GetDirection()), tuple(second.GetDirection())),
    ):
        if left != right:
            mismatches.append(f"{name} mismatch: {left} != {right}")
    return mismatches


def _geometry_dict(image):
    geometry = summarize_geometry(image)
    return {
        "size_xyz": list(geometry.size_xyz),
        "spacing_xyz_mm": list(geometry.spacing_xyz),
        "origin_xyz_mm": list(geometry.origin_xyz),
        "direction": list(geometry.direction),
        "exact_agreement_required": True,
    }


def _extract_slice(array, indices, view):
    if view == "axial":
        return array[indices["axial_z"]]
    if view == "coronal":
        return array[:, indices["coronal_y"], :]
    if view == "sagittal":
        return array[:, :, indices["sagittal_x"]]
    raise ValueError(f"Unknown view: {view}")


def _aspect(image, view):
    x, y, z = image.GetSpacing()
    return {"axial": y / x, "coronal": z / x, "sagittal": z / y}[view]


def _validate_options(
    absolute,
    fraction,
    percentile,
    opacity,
    gaussian_sigma_mm,
    gaussian_truncate,
    gaussian_support_radius_mm,
):
    if absolute is not None and fraction is not None:
        raise ValueError("Choose either an absolute threshold or a threshold fraction, not both.")
    if absolute is None and fraction is None:
        raise ValueError("A denominator threshold policy is required.")
    selected = absolute if absolute is not None else fraction
    if not math.isfinite(float(selected)) or float(selected) <= 0:
        raise ValueError("Denominator threshold or fraction must be finite and greater than zero.")
    if not 0 < float(percentile) <= 100:
        raise ValueError("Display percentile must be in (0, 100].")
    if not 0 <= float(opacity) <= 1:
        raise ValueError("Overlay opacity must be between 0 and 1.")
    if not math.isfinite(float(gaussian_sigma_mm)) or float(gaussian_sigma_mm) <= 0:
        raise ValueError("Gaussian sigma in millimeters must be finite and greater than zero.")
    if not math.isfinite(float(gaussian_truncate)) or float(gaussian_truncate) <= 0:
        raise ValueError("Gaussian truncation must be finite and greater than zero.")
    if gaussian_support_radius_mm is not None and (
        not math.isfinite(float(gaussian_support_radius_mm))
        or float(gaussian_support_radius_mm) <= 0
    ):
        raise ValueError("Gaussian support radius must be finite and greater than zero.")
