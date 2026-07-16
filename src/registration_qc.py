from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lung_ct_qc_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from medical_image_io import compare_image_geometry, summarize_geometry
from project_paths import DEFAULT_OUTPUTS_DIR, display_path
from registration_inputs import GEOMETRY_TOLERANCE, validate_registration_image
from version import PIPELINE_VERSION
from viewer_launcher import build_itksnap_launch_plan, launch_plan


SSD_PATTERN = re.compile(
    r"SSD\s+before\s+registration:\s*([-+0-9.eE]+)\s+and\s+after\s+([-+0-9.eE]+)",
    re.IGNORECASE,
)
FIGURE_NAMES = (
    "before_overlay.png",
    "after_overlay.png",
    "before_checkerboard.png",
    "after_checkerboard.png",
    "difference_before.png",
    "difference_after.png",
)


def compute_similarity_metrics(
    fixed: sitk.Image,
    candidate: sitk.Image,
    *,
    histogram_bins: int = 64,
    mask: np.ndarray | None = None,
    histogram_range: tuple[float, float] | None = None,
    block_depth: int = 8,
) -> dict[str, Any]:
    """Compute deterministic 3D intensity metrics in z-blocks.

    NCC is the global Pearson correlation over selected voxels. MI uses one
    deterministic square joint histogram and natural logarithms.
    """
    _validate_bins(histogram_bins)
    mismatches = compare_image_geometry(fixed, candidate)
    if mismatches:
        raise ValueError("Metric image grid mismatch: " + "; ".join(mismatches))
    fixed_array = sitk.GetArrayViewFromImage(fixed)
    candidate_array = sitk.GetArrayViewFromImage(candidate)
    if mask is not None and mask.shape != fixed_array.shape:
        raise ValueError("Metric mask shape does not match the image grid.")
    if histogram_range is None:
        histogram_range = _common_range((fixed_array, candidate_array))
    low, high = histogram_range
    edges = np.linspace(low, high, histogram_bins + 1, dtype=np.float64)
    histogram = np.zeros((histogram_bins, histogram_bins), dtype=np.int64)
    count = 0
    sum_fixed = sum_candidate = 0.0
    sum_fixed_sq = sum_candidate_sq = sum_cross = 0.0
    sum_squared_error = sum_absolute_error = 0.0
    depth = fixed_array.shape[0]
    for start in range(0, depth, block_depth):
        stop = min(start + block_depth, depth)
        x = np.asarray(fixed_array[start:stop], dtype=np.float64)
        y = np.asarray(candidate_array[start:stop], dtype=np.float64)
        if mask is not None:
            selected = mask[start:stop]
            x = x[selected]
            y = y[selected]
        else:
            x = x.reshape(-1)
            y = y.reshape(-1)
        if x.size == 0:
            continue
        difference = x - y
        count += int(x.size)
        sum_fixed += float(x.sum(dtype=np.float64))
        sum_candidate += float(y.sum(dtype=np.float64))
        sum_fixed_sq += float(np.dot(x, x))
        sum_candidate_sq += float(np.dot(y, y))
        sum_cross += float(np.dot(x, y))
        sum_squared_error += float(np.dot(difference, difference))
        sum_absolute_error += float(np.abs(difference).sum(dtype=np.float64))
        histogram += np.histogram2d(x, y, bins=(edges, edges))[0].astype(np.int64)
    if count == 0:
        raise ValueError("Metric domain is empty.")
    variance_fixed = sum_fixed_sq - (sum_fixed * sum_fixed / count)
    variance_candidate = sum_candidate_sq - (sum_candidate * sum_candidate / count)
    denominator = math.sqrt(max(variance_fixed, 0.0) * max(variance_candidate, 0.0))
    ncc = None if denominator == 0 else (sum_cross - sum_fixed * sum_candidate / count) / denominator
    return {
        "voxel_count": count,
        "mean_squared_error": sum_squared_error / count,
        "mean_absolute_error": sum_absolute_error / count,
        "normalized_cross_correlation": float(ncc) if ncc is not None else None,
        "mutual_information": _mutual_information(histogram),
        "ncc_note": (
            "Undefined because at least one selected image has zero variance."
            if ncc is None
            else None
        ),
    }


def metric_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changes = {}
    for name in (
        "mean_squared_error",
        "mean_absolute_error",
        "normalized_cross_correlation",
        "mutual_information",
    ):
        old, new = before.get(name), after.get(name)
        if old is None or new is None:
            changes[name] = {"absolute_change": None, "percentage_change": None, "improvement": None}
            continue
        absolute = float(new - old)
        percentage = None if old == 0 else float(absolute / abs(old) * 100.0)
        lower_is_better = name in {"mean_squared_error", "mean_absolute_error"}
        improvement = float(old - new) if lower_is_better else float(new - old)
        changes[name] = {
            "absolute_change": absolute,
            "percentage_change": percentage,
            "improvement": improvement,
            "direction": "lower-is-better" if lower_is_better else "higher-is-better",
        }
    return changes


def parse_deeds_ssd(log_path: str | Path | None) -> dict[str, Any] | None:
    if log_path is None or not Path(log_path).is_file():
        return None
    try:
        matches = SSD_PATTERN.findall(Path(log_path).read_text(errors="replace"))
    except OSError:
        return None
    if not matches:
        return None
    try:
        before, after = (float(value) for value in matches[-1])
    except ValueError:
        return None
    if not math.isfinite(before) or not math.isfinite(after):
        return None
    return {
        "before": before,
        "after": after,
        "absolute_change": after - before,
        "source": "final safely parsed deedsBCV stdout diagnostic",
    }


def extract_runtime_summary(manifest_path: str | Path) -> dict[str, float | None]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError("Registration manifest does not exist.")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Registration manifest is unreadable or malformed.") from exc
    stages = manifest.get("stages", {})
    affine = _finite_runtime(stages.get("affine", {}).get("runtime_seconds"))
    deformable = _finite_runtime(stages.get("deformable", {}).get("runtime_seconds"))
    total = affine + deformable if affine is not None and deformable is not None else None
    return {"affine_seconds": affine, "deformable_seconds": deformable, "total_seconds": total}


def select_representative_slices(
    image: sitk.Image, foreground_mask: np.ndarray | None = None
) -> dict[str, int]:
    shape = sitk.GetArrayViewFromImage(image).shape
    if foreground_mask is None:
        return {"axial_z": shape[0] // 2, "coronal_y": shape[1] // 2, "sagittal_x": shape[2] // 2}
    coordinates = np.argwhere(foreground_mask)
    if coordinates.size == 0:
        raise ValueError("Cannot select representative slices from an empty mask.")
    center = np.rint((coordinates.min(axis=0) + coordinates.max(axis=0)) / 2).astype(int)
    return {"axial_z": int(center[0]), "coronal_y": int(center[1]), "sagittal_x": int(center[2])}


def build_registration_itksnap_plan(
    fixed: str | Path,
    moving: str | Path,
    registered: str | Path,
    *,
    installation=None,
):
    return build_itksnap_launch_plan(
        image_path=fixed,
        overlay_paths=[moving, registered],
        installation=installation,
    )


def evaluate_registration(
    *,
    case_id: str,
    experiment: str,
    experiment_dir: str | Path,
    fixed_path: str | Path,
    moving_path: str | Path,
    registered_path: str | Path,
    registration_manifest_path: str | Path,
    fixed_mask_path: str | Path | None = None,
    moving_mask_path: str | Path | None = None,
    registered_mask_path: str | Path | None = None,
    histogram_bins: int = 64,
    overwrite: bool = False,
    dry_run: bool = False,
    launch_itksnap: bool = False,
    dry_run_launch: bool = False,
    launch_builder: Callable[..., Any] = build_registration_itksnap_plan,
    launcher: Callable[..., dict] = launch_plan,
) -> dict[str, Any]:
    _validate_bins(histogram_bins)
    experiment_root = Path(experiment_dir).expanduser().resolve(strict=False)
    output_dir = experiment_root / "qc"
    outputs = {name: output_dir / name for name in FIGURE_NAMES}
    outputs["manifest"] = output_dir / "registration_qc.json"
    existing = [path for path in outputs.values() if path.exists()]
    reuse_existing_qc = (
        launch_itksnap
        and not overwrite
        and not dry_run
        and len(existing) == len(outputs)
    )
    if existing and not overwrite and not dry_run and not reuse_existing_qc:
        raise FileExistsError("Registration QC outputs already exist; use --overwrite.")

    fixed = validate_registration_image(fixed_path, role="QC fixed")
    moving = validate_registration_image(moving_path, role="QC original moving")
    registered = validate_registration_image(registered_path, role="QC registered moving")
    original_mismatches = compare_image_geometry(fixed.image, moving.image)
    registered_mismatches = compare_image_geometry(fixed.image, registered.image)
    if original_mismatches:
        raise ValueError("Fixed/original-moving grid mismatch: " + "; ".join(original_mismatches))
    if registered_mismatches:
        raise ValueError("Fixed/registered-moving grid mismatch: " + "; ".join(registered_mismatches))
    registration_manifest = Path(registration_manifest_path).expanduser().resolve(strict=False)
    runtime = extract_runtime_summary(registration_manifest)
    masks = _load_masks(
        fixed.image,
        moving.image,
        registered.image,
        fixed_mask_path,
        moving_mask_path,
        registered_mask_path,
    )
    arrays = tuple(sitk.GetArrayViewFromImage(item.image) for item in (fixed, moving, registered))
    common_range = _common_range(arrays)
    before = compute_similarity_metrics(
        fixed.image, moving.image, histogram_bins=histogram_bins, histogram_range=common_range
    )
    after = compute_similarity_metrics(
        fixed.image, registered.image, histogram_bins=histogram_bins, histogram_range=common_range
    )
    masked_metrics = _masked_metrics(
        fixed.image, moving.image, registered.image, masks, histogram_bins, common_range
    )
    mask_similarity = _mask_similarity(masks)
    selected = select_representative_slices(fixed.image, masks.get("fixed"))
    geometry = summarize_geometry(fixed.image)
    voxel_count = int(np.prod(geometry.size_xyz))
    physical_volume = float(voxel_count * np.prod(geometry.spacing_xyz))
    log_path = experiment_root / "deformable" / "deeds_stdout.log"
    ssd = parse_deeds_ssd(log_path)
    warnings = [
        "Intensity similarity changes do not establish anatomical correctness; expert visual review is required.",
        "3D structural similarity is omitted because no validated memory-efficient 3D implementation is provided.",
        "Dice is only reported when corresponding binary anatomical masks are supplied.",
    ]
    if before["normalized_cross_correlation"] is None or after["normalized_cross_correlation"] is None:
        warnings.append("NCC is undefined for a zero-variance metric domain.")
    if ssd is None:
        warnings.append("Final deedsBCV SSD could not be parsed; QC continues without it.")
    figures = {name.removesuffix(".png"): display_path(path) for name, path in outputs.items() if name != "manifest"}
    result = {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": case_id,
        "experiment": experiment,
        "images": {
            "fixed": display_path(fixed.path),
            "original_moving": display_path(moving.path),
            "registered_moving": display_path(registered.path),
        },
        "geometry_validation": {
            "fixed_original_moving_same_grid": True,
            "fixed_registered_moving_same_grid": True,
            "size_xyz": list(geometry.size_xyz),
            "spacing_xyz_mm": list(geometry.spacing_xyz),
            "origin_xyz_mm": list(geometry.origin_xyz),
            "direction": list(geometry.direction),
        },
        "metrics": {"before": before, "after": after, "changes": metric_changes(before, after)},
        "masked_metrics": masked_metrics,
        "mask_usage": {
            "fixed": fixed_mask_path is not None,
            "moving": moving_mask_path is not None,
            "registered": registered_mask_path is not None,
            "dice": mask_similarity,
        },
        "histogram_bins": histogram_bins,
        "mutual_information_implementation": "deterministic 3D joint histogram with shared linear edges and natural logarithms",
        "runtime": runtime,
        "voxel_count": voxel_count,
        "physical_volume_mm3": physical_volume,
        "physical_volume_ml": physical_volume / 1000.0,
        "deedsbcv_ssd": ssd,
        "representative_slices": selected,
        "figures": figures,
        "warnings": warnings,
        "planned_outputs": {name: display_path(path) for name, path in outputs.items()},
        "completion_status": "dry-run" if dry_run else "pending",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Research-only and non-diagnostic.",
    }
    launch_result = None
    if launch_itksnap or dry_run_launch:
        plan = launch_builder(fixed.path, moving.path, registered.path)
        launch_result = launcher(plan, dry_run=dry_run_launch or dry_run)
        result["itksnap"] = launch_result
    if dry_run:
        result["writes_performed"] = False
        return result
    if reuse_existing_qc:
        result["completion_status"] = "reused"
        result["writes_performed"] = False
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    _render_figures(fixed.image, moving.image, registered.image, selected, outputs)
    result["completion_status"] = "complete"
    outputs["manifest"].write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_registration_experiments(
    *,
    case_id: str,
    full_qc_path: str | Path,
    cropped_qc_path: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    full = _load_complete_qc(full_qc_path, case_id, "full")
    cropped = _load_complete_qc(cropped_qc_path, case_id, "cropped")
    directory = Path(output_dir or DEFAULT_OUTPUTS_DIR / "registration" / case_id / "comparison").expanduser().resolve(strict=False)
    manifest_path = directory / "registration_comparison.json"
    figure_path = directory / "registration_comparison.png"
    if not overwrite and not dry_run and (manifest_path.exists() or figure_path.exists()):
        raise FileExistsError("Registration comparison outputs already exist; use --overwrite.")
    metric_names = (
        "mean_squared_error",
        "mean_absolute_error",
        "normalized_cross_correlation",
        "mutual_information",
    )
    winners = {}
    for name in metric_names:
        full_after = full["metrics"]["after"].get(name)
        cropped_after = cropped["metrics"]["after"].get(name)
        winners[name] = _metric_winner(name, full_after, cropped_after)
    full_runtime = full.get("runtime", {}).get("total_seconds")
    cropped_runtime = cropped.get("runtime", {}).get("total_seconds")
    runtime_winner = _lower_winner(full_runtime, cropped_runtime)
    result = {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": case_id,
        "experiments": {
            "full": _comparison_values(full),
            "cropped": _comparison_values(cropped),
        },
        "metric_winners": winners,
        "runtime_winner": runtime_winner,
        "overall_anatomical_winner": None,
        "interpretation": "No overall anatomical winner is declared without expert visual review.",
        "outputs": {"json": display_path(manifest_path), "figure": display_path(figure_path)},
        "completion_status": "dry-run" if dry_run else "pending",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Research-only and non-diagnostic.",
    }
    if dry_run:
        result["writes_performed"] = False
        return result
    directory.mkdir(parents=True, exist_ok=True)
    _render_comparison(full, cropped, figure_path)
    result["completion_status"] = "complete"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _load_masks(fixed, moving, registered, fixed_path, moving_path, registered_path):
    loaded = {}
    for name, path, image in (
        ("fixed", fixed_path, fixed),
        ("moving", moving_path, moving),
        ("registered", registered_path, registered),
    ):
        if path is None:
            continue
        mask = sitk.ReadImage(str(path))
        mismatches = compare_image_geometry(image, mask)
        if mismatches:
            raise ValueError(f"{name.capitalize()} mask grid mismatch: " + "; ".join(mismatches))
        array = sitk.GetArrayFromImage(mask)
        values = np.unique(array)
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError(f"{name.capitalize()} mask must be binary.")
        boolean = array.astype(bool)
        if not boolean.any():
            raise ValueError(f"{name.capitalize()} mask is empty.")
        loaded[name] = boolean
    return loaded


def _masked_metrics(fixed, moving, registered, masks, bins, value_range):
    results = {}
    if "fixed" in masks:
        results["fixed_mask_domain"] = {
            "before": compute_similarity_metrics(fixed, moving, histogram_bins=bins, mask=masks["fixed"], histogram_range=value_range),
            "after": compute_similarity_metrics(fixed, registered, histogram_bins=bins, mask=masks["fixed"], histogram_range=value_range),
        }
    if "fixed" in masks and "moving" in masks:
        intersection = masks["fixed"] & masks["moving"]
        results.setdefault("intersection_domains", {})["before"] = (
            compute_similarity_metrics(
                fixed, moving, histogram_bins=bins, mask=intersection, histogram_range=value_range
            )
            if intersection.any()
            else {"status": "unavailable", "reason": "fixed/moving mask intersection is empty"}
        )
    if "fixed" in masks and "registered" in masks:
        intersection = masks["fixed"] & masks["registered"]
        results.setdefault("intersection_domains", {})["after"] = (
            compute_similarity_metrics(
                fixed, registered, histogram_bins=bins, mask=intersection, histogram_range=value_range
            )
            if intersection.any()
            else {"status": "unavailable", "reason": "fixed/registered mask intersection is empty"}
        )
    return results


def _mask_similarity(masks):
    result = {"before": None, "after": None, "note": "Dice requires corresponding binary anatomical masks."}
    if "fixed" in masks and "moving" in masks:
        result["before"] = _dice(masks["fixed"], masks["moving"])
    if "fixed" in masks and "registered" in masks:
        result["after"] = _dice(masks["fixed"], masks["registered"])
    return result


def _dice(a, b):
    return float(2 * np.count_nonzero(a & b) / (np.count_nonzero(a) + np.count_nonzero(b)))


def _render_figures(fixed, moving, registered, selected, outputs):
    arrays = [sitk.GetArrayViewFromImage(image) for image in (fixed, moving, registered)]
    low, high = _percentile_range(arrays)
    diff_high = _difference_percentile(arrays)
    for state, candidate in (("before", arrays[1]), ("after", arrays[2])):
        _render_triptych(arrays[0], candidate, fixed, selected, outputs[f"{state}_overlay.png"], "overlay", low, high, diff_high)
        _render_triptych(arrays[0], candidate, fixed, selected, outputs[f"{state}_checkerboard.png"], "checkerboard", low, high, diff_high)
        _render_triptych(arrays[0], candidate, fixed, selected, outputs[f"difference_{state}.png"], "difference", low, high, diff_high)


def _render_triptych(fixed, candidate, image, selected, path, kind, low, high, diff_high):
    views = (
        ("Axial", fixed[selected["axial_z"]], candidate[selected["axial_z"]], image.GetSpacing()[1] / image.GetSpacing()[0]),
        ("Coronal", fixed[:, selected["coronal_y"], :], candidate[:, selected["coronal_y"], :], image.GetSpacing()[2] / image.GetSpacing()[0]),
        ("Sagittal", fixed[:, :, selected["sagittal_x"]], candidate[:, :, selected["sagittal_x"]], image.GetSpacing()[2] / image.GetSpacing()[1]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (title, a, b, aspect) in zip(axes, views):
        if kind == "overlay":
            axis.imshow(a, cmap="gray", vmin=low, vmax=high, aspect=aspect, origin="lower")
            axis.imshow(b, cmap="magma", vmin=low, vmax=high, alpha=0.4, aspect=aspect, origin="lower")
        elif kind == "checkerboard":
            tile = max(4, min(a.shape) // 8)
            grid = np.indices(a.shape).sum(axis=0) // tile
            checker = np.where(grid % 2 == 0, a, b)
            axis.imshow(checker, cmap="gray", vmin=low, vmax=high, aspect=aspect, origin="lower")
        else:
            axis.imshow(np.abs(a.astype(np.float64) - b.astype(np.float64)), cmap="gray", vmin=0, vmax=diff_high, aspect=aspect, origin="lower")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _render_comparison(full, cropped, path):
    labels = ["Affine", "Deformable", "Total"]
    full_values = [full["runtime"].get("affine_seconds"), full["runtime"].get("deformable_seconds"), full["runtime"].get("total_seconds")]
    cropped_values = [cropped["runtime"].get("affine_seconds"), cropped["runtime"].get("deformable_seconds"), cropped["runtime"].get("total_seconds")]
    x = np.arange(len(labels))
    width = 0.35
    figure = plt.figure(figsize=(7, 4))
    axis = figure.add_axes([0.12, 0.18, 0.84, 0.72])
    axis.bar(x - width / 2, [value or 0 for value in full_values], width, label="full")
    axis.bar(x + width / 2, [value or 0 for value in cropped_values], width, label="cropped")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title("Registration runtime comparison")
    axis.legend()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _common_range(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    low = min(float(np.min(array)) for array in arrays)
    high = max(float(np.max(array)) for array in arrays)
    if low == high:
        return low - 0.5, high + 0.5
    return low, high


def _percentile_range(arrays):
    samples = [np.asarray(array).reshape(-1)[:: max(1, array.size // 500_000)] for array in arrays]
    combined = np.concatenate(samples)
    low, high = np.percentile(combined, [1, 99])
    return (float(low), float(high)) if low < high else (float(low - 0.5), float(high + 0.5))


def _difference_percentile(arrays):
    step = max(1, arrays[0].size // 500_000)
    before = np.abs(np.asarray(arrays[0]).reshape(-1)[::step].astype(float) - np.asarray(arrays[1]).reshape(-1)[::step])
    after = np.abs(np.asarray(arrays[0]).reshape(-1)[::step].astype(float) - np.asarray(arrays[2]).reshape(-1)[::step])
    value = float(np.percentile(np.concatenate((before, after)), 99))
    return value if value > 0 else 1.0


def _mutual_information(histogram):
    total = histogram.sum()
    if total == 0:
        return 0.0
    joint = histogram.astype(np.float64) / total
    product = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    valid = joint > 0
    return float(np.sum(joint[valid] * np.log(joint[valid] / product[valid])))


def _finite_runtime(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _validate_bins(value):
    if not isinstance(value, int) or value < 2:
        raise ValueError("Histogram bins must be an integer of at least 2.")


def _load_complete_qc(path, case_id, experiment):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{experiment.capitalize()} QC manifest is unreadable.") from exc
    if data.get("completion_status") != "complete" or data.get("case_id") != case_id or data.get("experiment") != experiment:
        raise ValueError(f"{experiment.capitalize()} QC manifest does not match the requested complete experiment.")
    return data


def _metric_winner(name, full, cropped):
    if full is None or cropped is None or full == cropped:
        return "tie-or-unavailable"
    if name in {"mean_squared_error", "mean_absolute_error"}:
        return "full" if full < cropped else "cropped"
    return "full" if full > cropped else "cropped"


def _lower_winner(full, cropped):
    if full is None or cropped is None or full == cropped:
        return "tie-or-unavailable"
    return "full" if full < cropped else "cropped"


def _comparison_values(data):
    return {
        "runtime": data.get("runtime"),
        "voxel_count": data.get("voxel_count"),
        "physical_volume_mm3": data.get("physical_volume_mm3"),
        "metrics_before": data.get("metrics", {}).get("before"),
        "metrics_after": data.get("metrics", {}).get("after"),
        "metric_changes": data.get("metrics", {}).get("changes"),
        "deedsbcv_ssd": data.get("deedsbcv_ssd"),
    }
