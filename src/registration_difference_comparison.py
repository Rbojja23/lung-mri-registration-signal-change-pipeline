from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lung_ct_registration_difference_comparison_mpl"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import SimpleITK as sitk

from medical_image_io import summarize_geometry
from project_paths import DEFAULT_OUTPUTS_DIR, display_path
from registration_inputs import sha256_file, validate_registration_image
from version import PIPELINE_VERSION


ANALYSIS_TYPE = "Case 1-versus-Case 2 equal-window registration difference visualization"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OUTPUTS_DIR
    / "registration"
    / "case_comparison"
    / "equal_window_registration_differences"
)
OUTPUT_NAMES = {
    "grayscale": "matched_window_grayscale_axial.png",
    "turbo": "matched_window_turbo_axial.png",
    "overlay": "matched_window_turbo_overlay_axial.png",
    "multiplanar": "matched_window_turbo_multiplanar.png",
    "report": "equal_window_registration_difference_report.json",
}
PANEL_KEYS = (
    "case_1_before",
    "case_1_after",
    "case_2_before",
    "case_2_after",
)
PANEL_LABELS = {
    "case_1_before": "Case 1 — before registration",
    "case_1_after": "Case 1 — after registration",
    "case_2_before": "Case 2 — before registration",
    "case_2_after": "Case 2 — after registration",
}
FORMULAS = {
    "before": "abs(fixed - moving)",
    "after": "abs(fixed - registered)",
}
COLORBAR_LABEL = "Absolute intensity difference |F − candidate|"


def compare_registration_differences(
    *,
    case_1_dir: str | Path,
    case_2_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    percentile: float = 99.0,
    opacity: float = 0.45,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create research-only, cross-case window-matched difference figures.

    Inputs are read from each completed full-registration directory. They are
    validated and hashed but are never resampled, reoriented, normalized, or
    written by this function.
    """
    _validate_options(percentile, opacity)
    case_dirs = {
        "case_1": Path(case_1_dir).expanduser().resolve(strict=False),
        "case_2": Path(case_2_dir).expanduser().resolve(strict=False),
    }
    destination = Path(output_dir).expanduser().resolve(strict=False)
    output_paths = {name: destination / filename for name, filename in OUTPUT_NAMES.items()}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite and not dry_run:
        raise FileExistsError(
            "Equal-window comparison outputs already exist; use --overwrite."
        )

    registrations = {
        case: _load_complete_registration(case, directory)
        for case, directory in case_dirs.items()
    }
    cases = {case: _load_case(case, directory) for case, directory in case_dirs.items()}
    hashes_before = {
        case: {role: sha256_file(item.path) for role, item in inputs.items()}
        for case, inputs in cases.items()
    }
    fixed_grid_mismatches = _exact_geometry_mismatches(
        cases["case_1"]["fixed"].image,
        cases["case_2"]["fixed"].image,
    )
    slice_indices, physical_points = _matched_slice_indices(cases, fixed_grid_mismatches)
    arrays = {
        case: {
            role: sitk.GetArrayViewFromImage(validated.image)
            for role, validated in inputs.items()
        }
        for case, inputs in cases.items()
    }
    difference_pairs = _difference_pairs(arrays)
    shared_vmax = compute_shared_difference_percentile(
        [(fixed, candidate) for _, fixed, candidate in difference_pairs],
        percentile=percentile,
    )
    legacy_windows = {
        case: _legacy_per_case_window(arrays[case]) for case in ("case_1", "case_2")
    }
    clipping = _clipping_counts(difference_pairs, shared_vmax)
    anatomy_limits = _pooled_anatomy_limits(
        [arrays["case_1"]["fixed"], arrays["case_2"]["fixed"]]
    )
    panel_limits = {
        name: {key: {"vmin": 0.0, "vmax": shared_vmax} for key in PANEL_KEYS}
        for name in ("grayscale", "turbo", "overlay")
    }
    panel_limits["multiplanar"] = {
        view: {key: {"vmin": 0.0, "vmax": shared_vmax} for key in PANEL_KEYS}
        for view in ("axial", "coronal", "sagittal")
    }

    geometry_report = {
        case: {
            "within_case_exact_grid": True,
            "fixed_vs_original_moving_mismatches": [],
            "fixed_vs_registered_moving_mismatches": [],
            "fixed": _geometry_dict(inputs["fixed"].image),
        }
        for case, inputs in cases.items()
    }
    geometry_report["cross_case_fixed_grids"] = {
        "agree": not fixed_grid_mismatches,
        "mismatches": fixed_grid_mismatches,
        "slice_policy": (
            "identical voxel indices because the fixed grids agree"
            if not fixed_grid_mismatches
            else "same physical center point mapped independently to each fixed grid"
        ),
        "comparison": "exact equality of size, spacing, origin, and direction",
    }
    warnings = [
        "Research-only and non-diagnostic; intensity difference does not establish anatomical correctness.",
        "Residual differences may reflect anatomy, respiratory state, MRI intensity variation, interpolation, or registration error.",
        "Existing registration_qc behavior shares one sampled 99th-percentile window between before and after within each case, but calculates it independently for each case; those existing figures are therefore not equally windowed across cases.",
        "The new shared display window is calculated from every voxel in all four difference volumes and is used for display only; analytical NIfTI inputs are unchanged.",
    ]
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "analysis_type": ANALYSIS_TYPE,
        "intended_use": "Research-only, descriptive, and non-diagnostic.",
        "inputs": {
            case: {
                "registration_manifest": {
                    "path": display_path(registrations[case]["path"]),
                    "sha256": registrations[case]["sha256"],
                },
                "analytical_images": {
                    role: {
                        "path": display_path(item.path),
                        "sha256": hashes_before[case][role],
                    }
                    for role, item in inputs.items()
                },
            }
            for case, inputs in cases.items()
        },
        "geometry_validation": geometry_report,
        "formulas": FORMULAS,
        "slice_selection": {
            "indices_numpy_zyx": slice_indices,
            "physical_points_xyz_mm": physical_points,
        },
        "existing_qc_window_behavior": {
            "before_and_after_share_a_window_within_each_case": True,
            "window_is_calculated_independently_for_each_case": True,
            "equal_across_cases": legacy_windows["case_1"] == legacy_windows["case_2"],
            "sampled_99th_percentile_vmax_by_case": legacy_windows,
            "implementation": "registration_qc._difference_percentile; stride-sampled pooled before/after differences per case",
        },
        "shared_difference_window": {
            "method": "linear percentile over the pooled voxels of all four full-resolution difference volumes",
            "percentile": float(percentile),
            "pooled_volume_order": list(PANEL_KEYS),
            "pooled_voxel_count": int(sum(fixed.size for _, fixed, _ in difference_pairs)),
            "vmin": 0.0,
            "vmax": shared_vmax,
            "identical_for_every_difference_panel": True,
            "panel_limits": panel_limits,
        },
        "display": {
            "colormaps": {
                "grayscale": "gray",
                "turbo": "turbo",
                "overlay_difference": "turbo",
                "overlay_fixed_anatomy": "gray",
                "multiplanar": "turbo",
            },
            "overlay_opacity": float(opacity),
            "fixed_anatomy_limits": {"vmin": anatomy_limits[0], "vmax": anatomy_limits[1]},
            "colorbar_label": COLORBAR_LABEL,
            "physical_aspect_ratios_preserved": True,
        },
        "clipping_counts": clipping,
        "outputs": {name: display_path(path) for name, path in output_paths.items()},
        "input_immutability": {
            "checksums_verified_before_and_after": None,
            "unchanged": None,
        },
        "warnings": warnings,
        "completion_status": "dry-run" if dry_run else "pending",
        "writes_performed": False if dry_run else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        return report

    destination.mkdir(parents=True, exist_ok=True)
    difference_slices = _panel_slices(arrays, slice_indices, "axial")
    fixed_slices = _fixed_panel_slices(arrays, slice_indices, "axial")
    axial_aspects = _panel_aspects(cases, "axial")
    title_suffix = (
        f"Absolute difference |F − candidate|; shared {percentile:g}th-percentile window "
        f"[0, {shared_vmax:.6g}] — RESEARCH ONLY"
    )
    _render_four_panel(
        difference_slices,
        output_paths["grayscale"],
        cmap="gray",
        shared_vmax=shared_vmax,
        aspects=axial_aspects,
        title=f"Case 1 vs Case 2 registration differences — axial grayscale\n{title_suffix}",
    )
    _render_four_panel(
        difference_slices,
        output_paths["turbo"],
        cmap="turbo",
        shared_vmax=shared_vmax,
        aspects=axial_aspects,
        title=f"Case 1 vs Case 2 registration differences — axial turbo\n{title_suffix}",
    )
    _render_overlay_four_panel(
        fixed_slices,
        difference_slices,
        output_paths["overlay"],
        shared_vmax=shared_vmax,
        anatomy_limits=anatomy_limits,
        opacity=opacity,
        aspects=axial_aspects,
        title=f"Case 1 vs Case 2 fixed-anatomy difference overlays — axial turbo\n{title_suffix}",
    )
    _render_multiplanar(
        arrays,
        cases,
        slice_indices,
        output_paths["multiplanar"],
        percentile=percentile,
        shared_vmax=shared_vmax,
    )

    hashes_after = {
        case: {role: sha256_file(item.path) for role, item in inputs.items()}
        for case, inputs in cases.items()
    }
    unchanged = hashes_before == hashes_after
    if not unchanged:
        raise RuntimeError("An analytical input checksum changed while creating figures.")
    report["input_immutability"] = {
        "checksums_verified_before_and_after": True,
        "unchanged": True,
        "sha256_after": hashes_after,
    }
    report["completion_status"] = "complete"
    report["writes_performed"] = True
    output_paths["report"].write_text(json.dumps(report, indent=2) + "\n")
    return report


def compute_shared_difference_percentile(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    percentile: float = 99.0,
    block_depth: int = 8,
) -> float:
    """Return an order-invariant percentile over all pairwise absolute differences.

    A temporary float32 memmap keeps the full pooled distribution exact at the
    analytical inputs' effective precision without retaining four 3D difference
    arrays in RAM. No subsampling is performed.
    """
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be greater than 0 and at most 100.")
    if not pairs:
        raise ValueError("At least one difference pair is required.")
    normalized = []
    total = 0
    for fixed, candidate in pairs:
        fixed_array = np.asarray(fixed)
        candidate_array = np.asarray(candidate)
        if fixed_array.shape != candidate_array.shape:
            raise ValueError("Difference arrays must have identical shapes.")
        if fixed_array.ndim != 3 or fixed_array.size == 0:
            raise ValueError("Difference arrays must be non-empty 3D volumes.")
        if not np.isfinite(fixed_array).all() or not np.isfinite(candidate_array).all():
            raise ValueError("Difference inputs must contain only finite values.")
        normalized.append((fixed_array, candidate_array))
        total += fixed_array.size

    with tempfile.TemporaryDirectory(prefix="registration_difference_percentile_") as tmp:
        path = Path(tmp) / "pooled.float32"
        pool = np.memmap(path, dtype=np.float32, mode="w+", shape=(total,))
        offset = 0
        for fixed, candidate in normalized:
            for start in range(0, fixed.shape[0], block_depth):
                stop = min(start + block_depth, fixed.shape[0])
                difference = np.abs(
                    fixed[start:stop].astype(np.float32)
                    - candidate[start:stop].astype(np.float32)
                ).reshape(-1)
                pool[offset : offset + difference.size] = difference
                offset += difference.size
        pool.flush()
        value = float(np.percentile(pool, percentile, method="linear", overwrite_input=True))
        del pool
    return value if value > 0 else 1.0


def _load_case(case: str, directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError(f"{PANEL_LABELS[case + '_before'].split(' — ')[0]} directory does not exist.")
    paths = {
        "fixed": directory / "full/preprocessed/fixed.nii.gz",
        "moving": directory / "full/preprocessed/moving.nii.gz",
        "registered": directory / "full/deformable/moving_registered.nii.gz",
    }
    loaded = {
        role: validate_registration_image(path, role=f"{case} {role}")
        for role, path in paths.items()
    }
    for role in ("moving", "registered"):
        mismatches = _exact_geometry_mismatches(
            loaded["fixed"].image,
            loaded[role].image,
        )
        if mismatches:
            raise ValueError(
                f"{case.replace('_', ' ').title()} fixed/{role} exact geometry mismatch: "
                + "; ".join(mismatches)
                + ". Analytical inputs were not resampled or altered."
            )
    return loaded


def _load_complete_registration(case: str, directory: Path) -> dict[str, Any]:
    path = directory / "full/registration_manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{case.replace('_', ' ').title()} full registration manifest does not exist."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{case.replace('_', ' ').title()} full registration manifest is unreadable or malformed."
        ) from exc
    if manifest.get("completion_status") != "complete" or manifest.get("experiment") != "full":
        raise ValueError(
            f"{case.replace('_', ' ').title()} registration manifest is not a completed full registration."
        )
    if not manifest.get("case_id"):
        raise ValueError(f"{case.replace('_', ' ').title()} registration manifest lacks case_id.")
    return {"path": path, "sha256": sha256_file(path), "case_id": manifest["case_id"]}


def _exact_geometry_mismatches(first: sitk.Image, second: sitk.Image) -> list[str]:
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


def _difference_pairs(arrays):
    return [
        ("case_1_before", arrays["case_1"]["fixed"], arrays["case_1"]["moving"]),
        ("case_1_after", arrays["case_1"]["fixed"], arrays["case_1"]["registered"]),
        ("case_2_before", arrays["case_2"]["fixed"], arrays["case_2"]["moving"]),
        ("case_2_after", arrays["case_2"]["fixed"], arrays["case_2"]["registered"]),
    ]


def _legacy_per_case_window(arrays):
    step = max(1, arrays["fixed"].size // 500_000)
    fixed = np.asarray(arrays["fixed"]).reshape(-1)[::step].astype(float)
    before = np.abs(fixed - np.asarray(arrays["moving"]).reshape(-1)[::step])
    after = np.abs(fixed - np.asarray(arrays["registered"]).reshape(-1)[::step])
    value = float(np.percentile(np.concatenate((before, after)), 99))
    return value if value > 0 else 1.0


def _clipping_counts(difference_pairs, vmax):
    volumes = {}
    total = 0
    for key, fixed, candidate in difference_pairs:
        count = 0
        difference_sum = 0.0
        difference_max = 0.0
        for start in range(0, fixed.shape[0], 8):
            difference = np.abs(
                fixed[start : start + 8].astype(np.float32)
                - candidate[start : start + 8].astype(np.float32)
            )
            count += int(np.count_nonzero(difference > vmax))
            difference_sum += float(difference.sum(dtype=np.float64))
            difference_max = max(difference_max, float(difference.max()))
        volumes[key] = {
            "above_vmax": count,
            "clipped_fraction": float(count / fixed.size),
            "voxel_count": int(fixed.size),
            "mean_absolute_difference": float(difference_sum / fixed.size),
            "maximum_absolute_difference": difference_max,
        }
        total += count
    return {
        "definition": "voxels with absolute difference strictly greater than shared vmax",
        "by_volume": volumes,
        "pooled_above_vmax": total,
        "pooled_voxel_count": int(sum(item[1].size for item in difference_pairs)),
    }


def _matched_slice_indices(cases, cross_case_mismatches):
    fixed_1 = cases["case_1"]["fixed"].image
    center_1 = tuple(int(size // 2) for size in fixed_1.GetSize())
    point = tuple(float(value) for value in fixed_1.TransformIndexToPhysicalPoint(center_1))
    if cross_case_mismatches:
        fixed_2 = cases["case_2"]["fixed"].image
        continuous = fixed_2.TransformPhysicalPointToContinuousIndex(point)
        if any(value < -0.5 or value > size - 0.5 for value, size in zip(continuous, fixed_2.GetSize())):
            raise ValueError("Case fixed grids do not share the Case 1 center physical point.")
        center_2 = tuple(int(value) for value in fixed_2.TransformPhysicalPointToIndex(point))
    else:
        center_2 = center_1
    indices = {
        "case_1": {"axial_z": center_1[2], "coronal_y": center_1[1], "sagittal_x": center_1[0]},
        "case_2": {"axial_z": center_2[2], "coronal_y": center_2[1], "sagittal_x": center_2[0]},
    }
    points = {
        case: list(cases[case]["fixed"].image.TransformIndexToPhysicalPoint(
            (values["sagittal_x"], values["coronal_y"], values["axial_z"])
        ))
        for case, values in indices.items()
    }
    return indices, points


def _geometry_dict(image):
    geometry = summarize_geometry(image)
    return {
        "size_xyz": list(geometry.size_xyz),
        "spacing_xyz_mm": list(geometry.spacing_xyz),
        "origin_xyz_mm": list(geometry.origin_xyz),
        "direction": list(geometry.direction),
    }


def _extract_slice(array, indices, view):
    if view == "axial":
        return array[indices["axial_z"]]
    if view == "coronal":
        return array[:, indices["coronal_y"], :]
    if view == "sagittal":
        return array[:, :, indices["sagittal_x"]]
    raise ValueError(f"Unknown view: {view}")


def _panel_slices(arrays, indices, view):
    slices = {}
    for key in PANEL_KEYS:
        case, state = key.rsplit("_", 1)
        candidate = "moving" if state == "before" else "registered"
        fixed_slice = _extract_slice(arrays[case]["fixed"], indices[case], view)
        candidate_slice = _extract_slice(arrays[case][candidate], indices[case], view)
        slices[key] = np.abs(
            fixed_slice.astype(np.float32) - candidate_slice.astype(np.float32)
        )
    return slices


def _fixed_panel_slices(arrays, indices, view):
    return {
        key: _extract_slice(arrays[key.rsplit("_", 1)[0]]["fixed"], indices[key.rsplit("_", 1)[0]], view)
        for key in PANEL_KEYS
    }


def _aspect(image, view):
    x, y, z = image.GetSpacing()
    return {"axial": y / x, "coronal": z / x, "sagittal": z / y}[view]


def _panel_aspects(cases, view):
    return {key: _aspect(cases[key.rsplit("_", 1)[0]]["fixed"].image, view) for key in PANEL_KEYS}


def _pooled_anatomy_limits(arrays):
    samples = [array.reshape(-1)[:: max(1, array.size // 500_000)] for array in arrays]
    low, high = np.percentile(np.concatenate(samples), [1, 99])
    if low == high:
        return float(low - 0.5), float(high + 0.5)
    return float(low), float(high)


def _render_four_panel(slices, path, *, cmap, shared_vmax, aspects, title):
    norm = Normalize(vmin=0.0, vmax=shared_vmax)
    figure, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    image = None
    for axis, key in zip(axes.flat, PANEL_KEYS):
        image = axis.imshow(slices[key], cmap=cmap, norm=norm, aspect=aspects[key], origin="lower")
        axis.set_title(f"{PANEL_LABELS[key]}\n{FORMULAS[key.rsplit('_', 1)[1]]}")
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.colorbar(
        image, ax=axes.ravel().tolist(), label=COLORBAR_LABEL,
        orientation="horizontal", shrink=0.82, pad=0.04, aspect=42,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_overlay_four_panel(
    fixed_slices, difference_slices, path, *, shared_vmax, anatomy_limits, opacity, aspects, title
):
    norm = Normalize(vmin=0.0, vmax=shared_vmax)
    figure, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    image = None
    for axis, key in zip(axes.flat, PANEL_KEYS):
        axis.imshow(
            fixed_slices[key], cmap="gray", vmin=anatomy_limits[0], vmax=anatomy_limits[1],
            aspect=aspects[key], origin="lower",
        )
        image = axis.imshow(
            difference_slices[key], cmap="turbo", norm=norm, alpha=opacity,
            aspect=aspects[key], origin="lower",
        )
        axis.set_title(f"{PANEL_LABELS[key]}\n{FORMULAS[key.rsplit('_', 1)[1]]}; α={opacity:g}")
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.colorbar(
        image, ax=axes.ravel().tolist(), label=COLORBAR_LABEL,
        orientation="horizontal", shrink=0.82, pad=0.04, aspect=42,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_multiplanar(arrays, cases, indices, path, *, percentile, shared_vmax):
    norm = Normalize(vmin=0.0, vmax=shared_vmax)
    views = ("axial", "coronal", "sagittal")
    figure, axes = plt.subplots(3, 4, figsize=(14, 11), layout="constrained")
    image = None
    for row, view in enumerate(views):
        slices = _panel_slices(arrays, indices, view)
        aspects = _panel_aspects(cases, view)
        for column, key in enumerate(PANEL_KEYS):
            axis = axes[row, column]
            image = axis.imshow(
                slices[key], cmap="turbo", norm=norm, aspect=aspects[key], origin="lower"
            )
            state = key.rsplit("_", 1)[1]
            axis.set_title(f"{view.title()} — {PANEL_LABELS[key]}\n{FORMULAS[state]}", fontsize=9)
            axis.axis("off")
    figure.suptitle(
        "Case 1 vs Case 2 multiplanar registration differences\n"
        f"Absolute difference |F − candidate|; shared {percentile:g}th-percentile window "
        f"[0, {shared_vmax:.6g}] — RESEARCH ONLY",
        fontsize=12,
    )
    figure.colorbar(
        image, ax=axes.ravel().tolist(), label=COLORBAR_LABEL,
        orientation="horizontal", shrink=0.8, pad=0.035, aspect=52,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _validate_options(percentile, opacity):
    if not 0 < float(percentile) <= 100:
        raise ValueError("Percentile must be greater than 0 and at most 100.")
    if not 0 <= float(opacity) <= 1:
        raise ValueError("Opacity must be between 0 and 1.")
