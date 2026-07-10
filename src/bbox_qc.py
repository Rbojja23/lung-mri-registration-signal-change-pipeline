from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lung_ct_qc_mpl"))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import SimpleITK as sitk

from bounding_box import BBoxZYX, physical_bbox_from_voxel_bbox
from medical_image_io import compare_image_geometry, normalize_modality, summarize_geometry
from project_paths import DEFAULT_OUTPUTS_DIR, display_path
from segmentation_qc import (
    DEFAULT_CT_WINDOW_CENTER,
    DEFAULT_CT_WINDOW_WIDTH,
    DEFAULT_MR_LOWER_PERCENTILE,
    DEFAULT_MR_UPPER_PERCENTILE,
    apply_ct_window,
    binary_contour,
    ct_display_limits,
    mr_display_limits,
)
from version import PIPELINE_VERSION


@dataclass(frozen=True)
class BBoxSliceSelection:
    axial_z: list[int]
    coronal_y: list[int]
    sagittal_x: list[int]


@dataclass(frozen=True)
class BBoxValidation:
    bbox: BBoxZYX
    physical_bbox_mm: dict[str, float]
    dimensions_voxels_zyx: tuple[int, int, int]
    dimensions_mm_zyx: tuple[float, float, float]
    all_mask_voxels_contained: bool
    warnings: list[str]


def load_bbox_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def bbox_from_json(data: dict[str, Any]) -> BBoxZYX:
    if "voxel_bbox_half_open_zyx" not in data:
        raise ValueError("bbox JSON missing voxel_bbox_half_open_zyx.")
    values = data["voxel_bbox_half_open_zyx"]
    required = ["z_min", "z_max", "y_min", "y_max", "x_min", "x_max"]
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"bbox JSON missing keys: {missing}")
    return BBoxZYX(
        z_min=int(values["z_min"]),
        z_max=int(values["z_max"]),
        y_min=int(values["y_min"]),
        y_max=int(values["y_max"]),
        x_min=int(values["x_min"]),
        x_max=int(values["x_max"]),
    )


def validate_bbox_inputs(
    image: sitk.Image,
    mask: sitk.Image,
    bbox_data: dict[str, Any],
    *,
    case_id: str,
) -> BBoxValidation:
    mismatches = compare_image_geometry(image, mask)
    if mismatches:
        raise ValueError("Image/mask geometry mismatch: " + "; ".join(mismatches))

    if bbox_data.get("case_id") and bbox_data["case_id"] != case_id:
        raise ValueError(
            f"bbox case_id {bbox_data['case_id']} does not match requested case_id {case_id}."
        )

    bbox = bbox_from_json(bbox_data)
    _validate_bbox_bounds(bbox, sitk.GetArrayFromImage(mask).shape)

    if bbox_data.get("original_image_size_xyz"):
        expected = tuple(int(v) for v in bbox_data["original_image_size_xyz"])
        if expected != image.GetSize():
            raise ValueError(
                f"bbox original image size {expected} does not match image size {image.GetSize()}."
            )

    mask_array = sitk.GetArrayFromImage(mask)
    values = np.unique(mask_array)
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(f"Mask must be binary with values 0 and 1. Found {values.tolist()}")
    if not np.any(mask_array):
        raise ValueError("Mask is empty.")

    all_contained = bbox_contains_mask(mask_array.astype(bool), bbox)
    if not all_contained:
        raise ValueError("bbox does not contain every nonzero mask voxel.")

    geom = summarize_geometry(image)
    dims_voxels = bbox.shape_zyx
    dims_mm = (
        float(dims_voxels[0] * geom.spacing_xyz[2]),
        float(dims_voxels[1] * geom.spacing_xyz[1]),
        float(dims_voxels[2] * geom.spacing_xyz[0]),
    )
    warnings = []
    if any(v == 0 for v in [bbox.z_min, bbox.y_min, bbox.x_min]) or (
        bbox.z_max == mask_array.shape[0]
        or bbox.y_max == mask_array.shape[1]
        or bbox.x_max == mask_array.shape[2]
    ):
        warnings.append("Bounding box touches one or more image boundaries.")
    return BBoxValidation(
        bbox=bbox,
        physical_bbox_mm=physical_bbox_from_voxel_bbox(image, bbox),
        dimensions_voxels_zyx=dims_voxels,
        dimensions_mm_zyx=dims_mm,
        all_mask_voxels_contained=all_contained,
        warnings=warnings,
    )


def bbox_contains_mask(mask_array: np.ndarray, bbox: BBoxZYX) -> bool:
    outside = mask_array.copy().astype(bool)
    outside[bbox.z_min:bbox.z_max, bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max] = False
    return not bool(np.any(outside))


def select_bbox_slices(mask_array: np.ndarray, bbox: BBoxZYX) -> BBoxSliceSelection:
    z_candidates = _representative_indices(bbox.z_min, bbox.z_max, 5)
    z_occupied = set(np.flatnonzero(mask_array.any(axis=(1, 2))).tolist())
    axial = [z for z in z_candidates if z in z_occupied] or z_candidates
    return BBoxSliceSelection(
        axial_z=axial,
        coronal_y=_representative_indices(bbox.y_min, bbox.y_max, 3),
        sagittal_x=_representative_indices(bbox.x_min, bbox.x_max, 3),
    )


def rectangle_for_view(bbox: BBoxZYX, view: str) -> tuple[float, float, float, float]:
    """Return matplotlib rectangle x, y, width, height for half-open bbox.

    Pixel centers are integer coordinates in imshow. The outer edge around
    included voxels [min:max) therefore starts at min - 0.5 and has width
    max - min.
    """

    if view == "axial":
        return (
            bbox.x_min - 0.5,
            bbox.y_min - 0.5,
            bbox.x_max - bbox.x_min,
            bbox.y_max - bbox.y_min,
        )
    if view == "coronal":
        return (
            bbox.x_min - 0.5,
            bbox.z_min - 0.5,
            bbox.x_max - bbox.x_min,
            bbox.z_max - bbox.z_min,
        )
    if view == "sagittal":
        return (
            bbox.y_min - 0.5,
            bbox.z_min - 0.5,
            bbox.y_max - bbox.y_min,
            bbox.z_max - bbox.z_min,
        )
    raise ValueError("view must be axial, coronal, or sagittal.")


def aspect_for_view(image: sitk.Image, view: str) -> float:
    spacing_x, spacing_y, spacing_z = summarize_geometry(image).spacing_xyz
    if view == "axial":
        return spacing_y / spacing_x
    if view == "coronal":
        return spacing_z / spacing_x
    if view == "sagittal":
        return spacing_z / spacing_y
    raise ValueError("view must be axial, coronal, or sagittal.")


def run_bbox_qc(
    *,
    image_path: str | Path,
    mask_path: str | Path,
    bbox_json_path: str | Path,
    case_id: str,
    modality: str,
    crop_image_path: str | Path | None = None,
    crop_mask_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    show: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    modality = normalize_modality(modality, allow_auto=False)
    image_path = Path(image_path).expanduser().resolve(strict=False)
    mask_path = Path(mask_path).expanduser().resolve(strict=False)
    bbox_json_path = Path(bbox_json_path).expanduser().resolve(strict=False)
    output_dir = (
        Path(output_dir).expanduser().resolve(strict=False)
        if output_dir is not None
        else (DEFAULT_OUTPUTS_DIR / "figures" / case_id).resolve(strict=False)
    )

    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    bbox_data = load_bbox_json(bbox_json_path)
    validation = validate_bbox_inputs(image, mask, bbox_data, case_id=case_id)
    mask_array = sitk.GetArrayFromImage(mask).astype(bool)
    slices = select_bbox_slices(mask_array, validation.bbox)

    planned = {
        "bounding_box_overlay": output_dir / "bounding_box_overlay.png",
        "bounding_box_crop_comparison": output_dir / "bounding_box_crop_comparison.png",
        "bounding_box_qc_json": output_dir / "bounding_box_qc.json",
    }

    crop_image = crop_mask = None
    if crop_image_path or crop_mask_path:
        if not (crop_image_path and crop_mask_path):
            raise ValueError("--crop-image and --crop-mask must be supplied together.")
        crop_image = sitk.ReadImage(str(crop_image_path))
        crop_mask = sitk.ReadImage(str(crop_mask_path))
        crop_mismatches = compare_image_geometry(crop_image, crop_mask)
        if crop_mismatches:
            raise ValueError("Crop image/mask geometry mismatch: " + "; ".join(crop_mismatches))

    summary = build_bbox_qc_json(
        case_id=case_id,
        modality=modality,
        bbox_validation=validation,
        slices=slices,
        output_paths=planned,
    )
    if dry_run:
        return {
            "status": "dry-run",
            "planned_outputs": {key: display_path(path) for key, path in planned.items()},
            "summary": summary,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [planned["bounding_box_overlay"], planned["bounding_box_qc_json"]]
    if crop_image is not None:
        outputs.append(planned["bounding_box_crop_comparison"])
    _check_overwrite(outputs, overwrite=overwrite)
    render_bbox_overlay(
        image=image,
        mask=mask,
        bbox=validation.bbox,
        slices=slices,
        case_id=case_id,
        modality=modality,
        output_path=planned["bounding_box_overlay"],
        show=show,
    )
    if crop_image is not None and crop_mask is not None:
        render_crop_comparison(
            image=image,
            mask=mask,
            bbox=validation.bbox,
            crop_image=crop_image,
            crop_mask=crop_mask,
            case_id=case_id,
            modality=modality,
            output_path=planned["bounding_box_crop_comparison"],
            show=show,
        )
    planned["bounding_box_qc_json"].write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def render_bbox_overlay(
    *,
    image: sitk.Image,
    mask: sitk.Image,
    bbox: BBoxZYX,
    slices: BBoxSliceSelection,
    case_id: str,
    modality: str,
    output_path: Path,
    show: bool = False,
) -> None:
    image_array, mask_array, vmin, vmax = _display_arrays(image, mask, modality)
    panels = []
    for z in slices.axial_z:
        panels.append((f"Axial z={z}", "axial", image_array[z], mask_array[z]))
    for y in slices.coronal_y:
        panels.append((f"Coronal y={y}", "coronal", image_array[:, y, :], mask_array[:, y, :]))
    for x in slices.sagittal_x:
        panels.append((f"Sagittal x={x}", "sagittal", image_array[:, :, x], mask_array[:, :, x]))
    _render_panels(
        panels=panels,
        image=image,
        bbox=bbox,
        case_id=case_id,
        modality=modality,
        output_path=output_path,
        vmin=vmin,
        vmax=vmax,
        show=show,
    )


def render_crop_comparison(
    *,
    image: sitk.Image,
    mask: sitk.Image,
    bbox: BBoxZYX,
    crop_image: sitk.Image,
    crop_mask: sitk.Image,
    case_id: str,
    modality: str,
    output_path: Path,
    show: bool = False,
) -> None:
    image_array, mask_array, vmin, vmax = _display_arrays(image, mask, modality)
    crop_array, crop_mask_array, crop_vmin, crop_vmax = _display_arrays(crop_image, crop_mask, modality)
    z = (bbox.z_min + bbox.z_max - 1) // 2
    crop_z = min(max(z - bbox.z_min, 0), crop_array.shape[0] - 1)
    panels = [
        (f"Original axial z={z}", "axial", image_array[z], mask_array[z], bbox),
        (
            f"Crop axial z={crop_z}",
            "crop_axial",
            crop_array[crop_z],
            crop_mask_array[crop_z],
            None,
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (title, view, base, mask_slice, maybe_bbox) in zip(axes, panels):
        ax.imshow(base, cmap="gray", vmin=vmin if maybe_bbox else crop_vmin, vmax=vmax if maybe_bbox else crop_vmax, aspect=aspect_for_view(image if maybe_bbox else crop_image, "axial"))
        overlay = np.ma.masked_where(~mask_slice, mask_slice)
        ax.imshow(overlay, cmap="Reds", alpha=0.25, aspect=aspect_for_view(image if maybe_bbox else crop_image, "axial"))
        contour = binary_contour(mask_slice)
        yy, xx = np.nonzero(contour)
        ax.scatter(xx, yy, s=0.4, c="cyan", marker=".")
        if maybe_bbox:
            ax.add_patch(_bbox_patch(bbox, "axial"))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{case_id} | {modality} crop comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def build_bbox_qc_json(
    *,
    case_id: str,
    modality: str,
    bbox_validation: BBoxValidation,
    slices: BBoxSliceSelection,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": case_id,
        "modality": modality,
        "voxel_bbox_half_open_zyx": bbox_validation.bbox.to_dict(),
        "physical_bbox_mm": bbox_validation.physical_bbox_mm,
        "displayed_slice_indices": {
            "axial_z": slices.axial_z,
            "coronal_y": slices.coronal_y,
            "sagittal_x": slices.sagittal_x,
        },
        "bbox_dimensions_voxels_zyx": list(bbox_validation.dimensions_voxels_zyx),
        "bbox_dimensions_mm_zyx": list(bbox_validation.dimensions_mm_zyx),
        "all_mask_voxels_contained": bbox_validation.all_mask_voxels_contained,
        "output_filenames": {key: path.name for key, path in output_paths.items()},
        "warnings": bbox_validation.warnings,
        "coordinate_conventions": {
            "numpy_array_order": "z, y, x",
            "simpleitk_index_order": "x, y, z",
            "bbox_intervals": "half-open [min:max)",
            "axial_display": "y versus x",
            "coronal_display": "z versus x",
            "sagittal_display": "z versus y",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _display_arrays(image: sitk.Image, mask: sitk.Image, modality: str):
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask).astype(bool)
    if normalize_modality(modality, allow_auto=False) == "CT":
        vmin, vmax = ct_display_limits(DEFAULT_CT_WINDOW_CENTER, DEFAULT_CT_WINDOW_WIDTH)
        return apply_ct_window(image_array), mask_array, vmin, vmax
    vmin, vmax = mr_display_limits(
        image_array,
        DEFAULT_MR_LOWER_PERCENTILE,
        DEFAULT_MR_UPPER_PERCENTILE,
    )
    return np.clip(image_array, vmin, vmax), mask_array, vmin, vmax


def _render_panels(
    *,
    panels: list[tuple[str, str, np.ndarray, np.ndarray]],
    image: sitk.Image,
    bbox: BBoxZYX,
    case_id: str,
    modality: str,
    output_path: Path,
    vmin: float,
    vmax: float,
    show: bool,
) -> None:
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.1 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (title, view, base, mask_slice) in zip(axes_flat, panels):
        aspect = aspect_for_view(image, view)
        ax.imshow(base, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)
        overlay = np.ma.masked_where(~mask_slice, mask_slice)
        ax.imshow(overlay, cmap="Reds", alpha=0.22, aspect=aspect)
        contour = binary_contour(mask_slice)
        yy, xx = np.nonzero(contour)
        ax.scatter(xx, yy, s=0.35, c="cyan", marker=".")
        ax.add_patch(_bbox_patch(bbox, view))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{case_id} | {modality} lung bbox QC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def _bbox_patch(bbox: BBoxZYX, view: str) -> Rectangle:
    x, y, width, height = rectangle_for_view(bbox, view)
    return Rectangle(
        (x, y),
        width,
        height,
        fill=False,
        edgecolor="yellow",
        linewidth=1.8,
        linestyle="-",
    )


def _representative_indices(start: int, stop: int, count: int) -> list[int]:
    if stop <= start:
        raise ValueError("Invalid bbox interval.")
    values = np.linspace(start, stop - 1, num=min(count, stop - start))
    return sorted({int(round(v)) for v in values})


def _validate_bbox_bounds(bbox: BBoxZYX, shape_zyx: tuple[int, int, int]) -> None:
    z_size, y_size, x_size = shape_zyx
    if not (0 <= bbox.z_min < bbox.z_max <= z_size):
        raise ValueError("Invalid bbox z bounds.")
    if not (0 <= bbox.y_min < bbox.y_max <= y_size):
        raise ValueError("Invalid bbox y bounds.")
    if not (0 <= bbox.x_min < bbox.x_max <= x_size):
        raise ValueError("Invalid bbox x bounds.")


def _check_overwrite(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing bbox QC outputs: "
            + ", ".join(display_path(path) for path in existing)
        )
