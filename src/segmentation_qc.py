from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lung_ct_qc_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import SimpleITK as sitk

from medical_image_io import (
    compare_image_geometry,
    normalize_modality,
    summarize_geometry,
    voxel_index_xyz_to_physical_point,
)
from project_paths import DEFAULT_OUTPUTS_DIR, display_path


DEFAULT_CT_WINDOW_CENTER = -500.0
DEFAULT_CT_WINDOW_WIDTH = 1500.0
DEFAULT_MR_LOWER_PERCENTILE = 1.0
DEFAULT_MR_UPPER_PERCENTILE = 99.0


@dataclass(frozen=True)
class IndividualMask:
    label: str
    path: Path


@dataclass(frozen=True)
class QCSliceSelection:
    axial_zyx: list[int]
    coronal_y: int
    sagittal_x_by_label: dict[str, int]


@dataclass(frozen=True)
class QCMetrics:
    voxel_count: int
    volume_mm3: float
    volume_ml: float
    extent_zyx: dict[str, list[int]]
    physical_dimensions_mm: dict[str, float]
    centroid_zyx: list[float]
    centroid_xyz: list[float]
    centroid_physical_xyz_mm: list[float]
    occupied_axial_slices: int
    image_fraction: float
    connected_component_count: int
    largest_connected_component_fraction: float
    touches_boundaries: dict[str, bool]


@dataclass(frozen=True)
class QCResult:
    metrics: QCMetrics
    selected_slices: QCSliceSelection
    warnings: list[str]
    individual_metrics: dict[str, dict[str, Any]]
    individual_overlap_voxels: int
    individual_union_equals_combined: bool | None


def load_image_and_masks(
    image_path: str | Path,
    mask_path: str | Path,
    individual_masks: Sequence[IndividualMask] = (),
) -> tuple[sitk.Image, sitk.Image, list[tuple[IndividualMask, sitk.Image]]]:
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    validate_geometry(image, mask, "combined mask")

    loaded_individuals = []
    for item in individual_masks:
        individual_image = sitk.ReadImage(str(item.path))
        validate_geometry(image, individual_image, f"individual mask {item.label}")
        loaded_individuals.append((item, individual_image))

    return image, mask, loaded_individuals


def validate_geometry(reference: sitk.Image, candidate: sitk.Image, label: str) -> None:
    mismatches = compare_image_geometry(reference, candidate)
    if mismatches:
        raise ValueError(f"Geometry mismatch for {label}: " + "; ".join(mismatches))


def ensure_binary_mask(array: np.ndarray, label: str) -> np.ndarray:
    values = np.unique(array)
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(f"{label} must be binary with values 0 and 1. Found {values.tolist()}")
    return array.astype(bool)


def compute_qc(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    individual_masks: Sequence[tuple[IndividualMask, sitk.Image]] = (),
) -> QCResult:
    validate_geometry(image, mask, "combined mask")
    mask_array = ensure_binary_mask(sitk.GetArrayFromImage(mask), "combined mask")
    if not np.any(mask_array):
        raise ValueError("Combined lung mask is empty.")

    metrics = compute_mask_metrics(mask_array, image)
    selected_slices = select_representative_slices(mask_array, individual_masks)
    warnings = qc_warnings(metrics)

    individual_metrics: dict[str, dict[str, Any]] = {}
    union = np.zeros_like(mask_array, dtype=bool)
    overlap = np.zeros_like(mask_array, dtype=bool)
    individual_union_equals_combined: bool | None = None
    for item, individual_image in individual_masks:
        validate_geometry(image, individual_image, f"individual mask {item.label}")
        individual_array = ensure_binary_mask(
            sitk.GetArrayFromImage(individual_image),
            f"individual mask {item.label}",
        )
        if not np.any(individual_array):
            warnings.append(f"Individual mask {item.label} is empty.")
        overlap |= union & individual_array
        union |= individual_array
        individual_stats = compute_mask_metrics(individual_array, image) if np.any(individual_array) else None
        individual_metrics[item.label] = {
            "path": display_path(item.path),
            "voxel_count": int(np.count_nonzero(individual_array)),
            "volume_mm3": (
                float(individual_stats.volume_mm3) if individual_stats is not None else 0.0
            ),
            "volume_ml": (
                float(individual_stats.volume_ml) if individual_stats is not None else 0.0
            ),
        }

    overlap_voxels = int(np.count_nonzero(overlap))
    if individual_masks:
        individual_union_equals_combined = bool(np.array_equal(union, mask_array))
        if overlap_voxels > 0:
            warnings.append(
                f"Individual masks overlap unexpectedly in {overlap_voxels} voxels."
            )
        if not individual_union_equals_combined:
            warnings.append("Union of individual masks does not equal the combined mask.")

    return QCResult(
        metrics=metrics,
        selected_slices=selected_slices,
        warnings=warnings,
        individual_metrics=individual_metrics,
        individual_overlap_voxels=overlap_voxels,
        individual_union_equals_combined=individual_union_equals_combined,
    )


def compute_mask_metrics(mask_array: np.ndarray, reference_image: sitk.Image) -> QCMetrics:
    coords = np.argwhere(mask_array)
    voxel_count = int(coords.shape[0])
    if voxel_count == 0:
        raise ValueError("Mask is empty.")

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    spacing_x, spacing_y, spacing_z = summarize_geometry(reference_image).spacing_xyz
    voxel_volume = float(spacing_x * spacing_y * spacing_z)
    volume_mm3 = float(voxel_count * voxel_volume)
    centroid_zyx = coords.mean(axis=0)
    centroid_xyz = [
        float(centroid_zyx[2]),
        float(centroid_zyx[1]),
        float(centroid_zyx[0]),
    ]
    centroid_physical = voxel_index_xyz_to_physical_point(
        reference_image, tuple(int(round(v)) for v in centroid_xyz)
    )

    component_image = sitk.ConnectedComponent(sitk.GetImageFromArray(mask_array.astype(np.uint8)))
    component_stats = sitk.LabelShapeStatisticsImageFilter()
    component_stats.Execute(component_image)
    component_labels = list(component_stats.GetLabels())
    component_sizes = [component_stats.GetNumberOfPixels(label) for label in component_labels]
    largest_fraction = float(max(component_sizes) / voxel_count) if component_sizes else 0.0

    z_size, y_size, x_size = mask_array.shape
    return QCMetrics(
        voxel_count=voxel_count,
        volume_mm3=volume_mm3,
        volume_ml=volume_mm3 / 1000.0,
        extent_zyx={
            "z": [int(z_min), int(z_max)],
            "y": [int(y_min), int(y_max)],
            "x": [int(x_min), int(x_max)],
        },
        physical_dimensions_mm={
            "z": float((z_max - z_min + 1) * spacing_z),
            "y": float((y_max - y_min + 1) * spacing_y),
            "x": float((x_max - x_min + 1) * spacing_x),
        },
        centroid_zyx=[float(v) for v in centroid_zyx],
        centroid_xyz=centroid_xyz,
        centroid_physical_xyz_mm=[float(v) for v in centroid_physical],
        occupied_axial_slices=int(np.count_nonzero(mask_array.any(axis=(1, 2)))),
        image_fraction=float(voxel_count / mask_array.size),
        connected_component_count=len(component_labels),
        largest_connected_component_fraction=largest_fraction,
        touches_boundaries={
            "x_min": bool(x_min == 0),
            "x_max": bool(x_max == x_size - 1),
            "y_min": bool(y_min == 0),
            "y_max": bool(y_max == y_size - 1),
            "z_min": bool(z_min == 0),
            "z_max": bool(z_max == z_size - 1),
        },
    )


def qc_warnings(metrics: QCMetrics) -> list[str]:
    warnings = []
    if metrics.voxel_count < 1000:
        warnings.append("Mask is extremely small; verify segmentation coverage.")
    if metrics.image_fraction > 0.5:
        warnings.append("Mask occupies an unusually large fraction of the image.")
    if metrics.connected_component_count > 8:
        warnings.append("Mask has many disconnected components.")
    if metrics.largest_connected_component_fraction < 0.8:
        warnings.append("Largest connected component is a low fraction of total mask.")
    touched = [name for name, value in metrics.touches_boundaries.items() if value]
    if touched:
        warnings.append(f"Mask touches image boundaries: {', '.join(touched)}.")
    return warnings


def select_representative_slices(
    mask_array: np.ndarray,
    individual_masks: Sequence[tuple[IndividualMask, sitk.Image]] = (),
) -> QCSliceSelection:
    axial_counts = mask_array.sum(axis=(1, 2))
    occupied = np.flatnonzero(axial_counts)
    if occupied.size == 0:
        raise ValueError("Cannot select slices from an empty mask.")

    robust_axial = _trim_sparse_edges(occupied, axial_counts)
    axial = _quantile_indices(robust_axial, [0.10, 0.30, 0.50, 0.70, 0.90])

    coronal_counts = mask_array.sum(axis=(0, 2))
    coronal_y = int(np.argmax(coronal_counts))

    sagittal_x_by_label: dict[str, int] = {}
    if individual_masks:
        for item, individual_image in individual_masks:
            individual = sitk.GetArrayFromImage(individual_image) > 0
            counts = individual.sum(axis=(0, 1))
            if counts.max() > 0:
                sagittal_x_by_label[item.label] = int(np.argmax(counts))
    if not sagittal_x_by_label:
        counts = mask_array.sum(axis=(0, 1))
        sagittal_x_by_label["combined_lungs"] = int(np.argmax(counts))

    return QCSliceSelection(
        axial_zyx=axial,
        coronal_y=coronal_y,
        sagittal_x_by_label=sagittal_x_by_label,
    )


def _trim_sparse_edges(occupied: np.ndarray, axial_counts: np.ndarray) -> np.ndarray:
    max_count = axial_counts[occupied].max()
    threshold = max(1, int(round(max_count * 0.05)))
    robust = occupied[axial_counts[occupied] >= threshold]
    return robust if robust.size >= 5 else occupied


def _quantile_indices(indices: np.ndarray, quantiles: Sequence[float]) -> list[int]:
    selected = []
    for q in quantiles:
        position = int(round(q * (len(indices) - 1)))
        selected.append(int(indices[position]))
    return selected


def ct_display_limits(center: float = DEFAULT_CT_WINDOW_CENTER, width: float = DEFAULT_CT_WINDOW_WIDTH) -> tuple[float, float]:
    return center - width / 2.0, center + width / 2.0


def apply_ct_window(array: np.ndarray, center: float = DEFAULT_CT_WINDOW_CENTER, width: float = DEFAULT_CT_WINDOW_WIDTH) -> np.ndarray:
    lower, upper = ct_display_limits(center, width)
    return np.clip(array, lower, upper)


def mr_display_limits(
    array: np.ndarray,
    lower_percentile: float = DEFAULT_MR_LOWER_PERCENTILE,
    upper_percentile: float = DEFAULT_MR_UPPER_PERCENTILE,
) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("MRI display scaling requires finite voxel values.")
    lower, upper = np.percentile(finite, [lower_percentile, upper_percentile])
    if lower == upper:
        upper = lower + 1.0
    return float(lower), float(upper)


def binary_contour(mask_2d: np.ndarray) -> np.ndarray:
    mask = mask_2d.astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors_all = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~neighbors_all


def render_qc_figures(
    *,
    image: sitk.Image,
    combined_mask: sitk.Image,
    qc: QCResult,
    modality: str,
    case_id: str,
    output_dir: Path,
    individual_masks: Sequence[tuple[IndividualMask, sitk.Image]] = (),
    show: bool = False,
    overwrite: bool = False,
    mr_lower_percentile: float = DEFAULT_MR_LOWER_PERCENTILE,
    mr_upper_percentile: float = DEFAULT_MR_UPPER_PERCENTILE,
    ct_window_center: float = DEFAULT_CT_WINDOW_CENTER,
    ct_window_width: float = DEFAULT_CT_WINDOW_WIDTH,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "segmentation_overlay.png"
    contour_path = output_dir / "segmentation_contours.png"
    for path in [overlay_path, contour_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing figure: {display_path(path)}")

    _render_contact_sheet(
        image=image,
        combined_mask=combined_mask,
        qc=qc,
        modality=modality,
        case_id=case_id,
        individual_masks=individual_masks,
        output_path=overlay_path,
        mode="filled",
        show=show,
        mr_lower_percentile=mr_lower_percentile,
        mr_upper_percentile=mr_upper_percentile,
        ct_window_center=ct_window_center,
        ct_window_width=ct_window_width,
    )
    _render_contact_sheet(
        image=image,
        combined_mask=combined_mask,
        qc=qc,
        modality=modality,
        case_id=case_id,
        individual_masks=individual_masks,
        output_path=contour_path,
        mode="contour",
        show=show,
        mr_lower_percentile=mr_lower_percentile,
        mr_upper_percentile=mr_upper_percentile,
        ct_window_center=ct_window_center,
        ct_window_width=ct_window_width,
    )
    return {"overlay": overlay_path, "contours": contour_path}


def _render_contact_sheet(
    *,
    image: sitk.Image,
    combined_mask: sitk.Image,
    qc: QCResult,
    modality: str,
    case_id: str,
    individual_masks: Sequence[tuple[IndividualMask, sitk.Image]],
    output_path: Path,
    mode: str,
    show: bool,
    mr_lower_percentile: float,
    mr_upper_percentile: float,
    ct_window_center: float,
    ct_window_width: float,
) -> None:
    image_array = sitk.GetArrayFromImage(image)
    combined_array = sitk.GetArrayFromImage(combined_mask) > 0
    individual_arrays = [
        (item.label, sitk.GetArrayFromImage(mask_image) > 0)
        for item, mask_image in individual_masks
    ]
    modality_normalized = normalize_modality(modality, allow_auto=False)
    if modality_normalized == "CT":
        display_array = apply_ct_window(image_array, ct_window_center, ct_window_width)
        vmin, vmax = ct_display_limits(ct_window_center, ct_window_width)
        display_note = f"CT window c={ct_window_center:g}, w={ct_window_width:g}"
    else:
        vmin, vmax = mr_display_limits(
            image_array,
            mr_lower_percentile,
            mr_upper_percentile,
        )
        display_array = np.clip(image_array, vmin, vmax)
        display_note = f"MR percentiles {mr_lower_percentile:g}-{mr_upper_percentile:g}"

    spacing_x, spacing_y, spacing_z = summarize_geometry(image).spacing_xyz
    panels = []
    for z in qc.selected_slices.axial_zyx:
        panels.append((f"Axial z={z}", display_array[z, :, :], combined_array[z, :, :], _individual_slices(individual_arrays, "axial", z), spacing_y / spacing_x))
    y = qc.selected_slices.coronal_y
    panels.append((f"Coronal y={y}", display_array[:, y, :], combined_array[:, y, :], _individual_slices(individual_arrays, "coronal", y), spacing_z / spacing_x))
    for label, x in qc.selected_slices.sagittal_x_by_label.items():
        panels.append((f"Sagittal x={x} ({label})", display_array[:, :, x], combined_array[:, :, x], _individual_slices(individual_arrays, "sagittal", x), spacing_z / spacing_y))

    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows + 1.0))
    axes_array = np.atleast_1d(axes).ravel()
    fig.suptitle(
        f"{case_id} | {modality_normalized} | shape z,y,x={image_array.shape} | "
        f"spacing x,y,z={tuple(round(v, 4) for v in (spacing_x, spacing_y, spacing_z))} | "
        f"volume={qc.metrics.volume_ml:.2f} mL | {display_note}",
        fontsize=11,
    )

    colors = ["tab:red", "tab:cyan", "tab:orange", "tab:green", "tab:purple"]
    for ax, (title, base, combined, individuals, aspect) in zip(axes_array, panels):
        ax.imshow(base, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)
        if mode == "filled":
            if individuals:
                for idx, (label, arr) in enumerate(individuals):
                    overlay = np.ma.masked_where(~arr, arr)
                    ax.imshow(overlay, cmap=_single_color_cmap(colors[idx % len(colors)]), alpha=0.35, aspect=aspect)
            else:
                overlay = np.ma.masked_where(~combined, combined)
                ax.imshow(overlay, cmap=_single_color_cmap("tab:red"), alpha=0.35, aspect=aspect)
        else:
            if individuals:
                for idx, (_label, arr) in enumerate(individuals):
                    contour = binary_contour(arr)
                    y_coords, x_coords = np.nonzero(contour)
                    ax.scatter(x_coords, y_coords, s=0.3, c=colors[idx % len(colors)], marker=".")
            else:
                contour = binary_contour(combined)
                y_coords, x_coords = np.nonzero(contour)
                ax.scatter(x_coords, y_coords, s=0.3, c="tab:red", marker=".")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for ax in axes_array[len(panels):]:
        ax.axis("off")

    legend_items = []
    if individual_arrays:
        for idx, (label, _arr) in enumerate(individual_arrays):
            legend_items.append(Patch(facecolor=colors[idx % len(colors)], label=label, alpha=0.5))
    else:
        legend_items.append(Patch(facecolor="tab:red", label="combined_lungs", alpha=0.5))
    fig.legend(handles=legend_items, loc="lower center", ncol=min(4, len(legend_items)))
    fig.text(
        0.01,
        0.01,
        "Coordinate conventions: NumPy arrays use z,y,x; SimpleITK uses x,y,z. "
        f"Selected axial z={qc.selected_slices.axial_zyx}, coronal y={qc.selected_slices.coronal_y}.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def _individual_slices(
    individual_arrays: Sequence[tuple[str, np.ndarray]], view: str, index: int
) -> list[tuple[str, np.ndarray]]:
    slices = []
    for label, array in individual_arrays:
        if view == "axial":
            slices.append((label, array[index, :, :]))
        elif view == "coronal":
            slices.append((label, array[:, index, :]))
        else:
            slices.append((label, array[:, :, index]))
    return slices


def _single_color_cmap(color: str):
    from matplotlib.colors import ListedColormap, to_rgba

    return ListedColormap([to_rgba(color)])


def build_qc_json(
    *,
    case_id: str,
    modality: str,
    image_path: Path,
    mask_path: Path,
    image: sitk.Image,
    qc: QCResult,
    figure_paths: dict[str, Path],
) -> dict[str, Any]:
    geometry = summarize_geometry(image)
    return {
        "case_id": case_id,
        "modality": normalize_modality(modality, allow_auto=False),
        "image_path": display_path(image_path),
        "mask_path": display_path(mask_path),
        "image_geometry": {
            "size_xyz": list(geometry.size_xyz),
            "array_shape_zyx": list(geometry.array_shape_zyx),
            "spacing_xyz_mm": list(geometry.spacing_xyz),
            "origin_xyz_mm": list(geometry.origin_xyz),
            "direction": list(geometry.direction),
        },
        "mask_metrics": qc.metrics.__dict__,
        "individual_masks": qc.individual_metrics,
        "individual_overlap_voxels": qc.individual_overlap_voxels,
        "individual_union_equals_combined": qc.individual_union_equals_combined,
        "selected_slices": {
            "axial_z_indices_numpy_zyx": qc.selected_slices.axial_zyx,
            "coronal_y_index_numpy_zyx": qc.selected_slices.coronal_y,
            "sagittal_x_indices_by_label_numpy_zyx": qc.selected_slices.sagittal_x_by_label,
        },
        "output_figures": {name: path.name for name, path in figure_paths.items()},
        "qc_warnings": qc.warnings,
        "coordinate_conventions": {
            "simpleitk_index_order": "x, y, z",
            "numpy_array_order": "z, y, x",
            "physical_points": "millimeters in the image coordinate frame",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_segmentation_qc(
    *,
    image_path: str | Path,
    mask_path: str | Path,
    modality: str,
    case_id: str,
    output_dir: str | Path | None = None,
    individual_masks: Sequence[IndividualMask] = (),
    show: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    mr_lower_percentile: float = DEFAULT_MR_LOWER_PERCENTILE,
    mr_upper_percentile: float = DEFAULT_MR_UPPER_PERCENTILE,
    ct_window_center: float = DEFAULT_CT_WINDOW_CENTER,
    ct_window_width: float = DEFAULT_CT_WINDOW_WIDTH,
) -> dict[str, Any]:
    normalized_modality = normalize_modality(modality, allow_auto=False)
    image_path = Path(image_path).expanduser().resolve(strict=False)
    mask_path = Path(mask_path).expanduser().resolve(strict=False)
    output_dir = (
        Path(output_dir).expanduser().resolve(strict=False)
        if output_dir is not None
        else (DEFAULT_OUTPUTS_DIR / "figures" / case_id).resolve(strict=False)
    )

    image, mask, loaded_individuals = load_image_and_masks(
        image_path,
        mask_path,
        individual_masks,
    )
    qc = compute_qc(image, mask, individual_masks=loaded_individuals)
    planned_figures = {
        "overlay": output_dir / "segmentation_overlay.png",
        "contours": output_dir / "segmentation_contours.png",
    }
    qc_json_path = output_dir / "segmentation_qc.json"

    if dry_run:
        return {
            "status": "dry-run",
            "planned_output_dir": display_path(output_dir),
            "planned_figures": {name: display_path(path) for name, path in planned_figures.items()},
            "planned_qc_json": display_path(qc_json_path),
            "selected_slices": {
                "axial_z_indices_numpy_zyx": qc.selected_slices.axial_zyx,
                "coronal_y_index_numpy_zyx": qc.selected_slices.coronal_y,
                "sagittal_x_indices_by_label_numpy_zyx": qc.selected_slices.sagittal_x_by_label,
            },
            "warnings": qc.warnings,
        }

    if output_dir.exists():
        for path in [*planned_figures.values(), qc_json_path]:
            if path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing QC output: {display_path(path)}")

    figures = render_qc_figures(
        image=image,
        combined_mask=mask,
        qc=qc,
        modality=normalized_modality,
        case_id=case_id,
        output_dir=output_dir,
        individual_masks=loaded_individuals,
        show=show,
        overwrite=overwrite,
        mr_lower_percentile=mr_lower_percentile,
        mr_upper_percentile=mr_upper_percentile,
        ct_window_center=ct_window_center,
        ct_window_width=ct_window_width,
    )
    qc_json = build_qc_json(
        case_id=case_id,
        modality=normalized_modality,
        image_path=image_path,
        mask_path=mask_path,
        image=image,
        qc=qc,
        figure_paths=figures,
    )
    qc_json_path.write_text(json.dumps(qc_json, indent=2) + "\n")
    return qc_json
