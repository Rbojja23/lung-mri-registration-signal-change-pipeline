from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import SimpleITK as sitk

from medical_image_io import compare_image_geometry, summarize_geometry
from project_paths import DEFAULT_OUTPUTS_DIR, display_path
from version import PIPELINE_VERSION


@dataclass(frozen=True)
class BBoxZYX:
    """Half-open NumPy bbox using z, y, x order: [min:max)."""

    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (
            self.z_max - self.z_min,
            self.y_max - self.y_min,
            self.x_max - self.x_min,
        )

    @property
    def start_xyz(self) -> tuple[int, int, int]:
        return self.x_min, self.y_min, self.z_min

    @property
    def size_xyz(self) -> tuple[int, int, int]:
        z_size, y_size, x_size = self.shape_zyx
        return x_size, y_size, z_size

    def to_dict(self) -> dict[str, int]:
        return {
            "z_min": self.z_min,
            "z_max": self.z_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "x_min": self.x_min,
            "x_max": self.x_max,
        }


@dataclass(frozen=True)
class PaddingZYX:
    z: int
    y: int
    x: int

    def to_dict(self) -> dict[str, int]:
        return {"z": self.z, "y": self.y, "x": self.x}


@dataclass(frozen=True)
class CropStats:
    original_voxel_count: int
    cropped_voxel_count: int
    reduction_percentage: float
    original_volume_mm3: float
    cropped_volume_mm3: float
    bounding_dimensions_voxels_zyx: tuple[int, int, int]
    bounding_dimensions_mm_zyx: tuple[float, float, float]
    lung_voxel_count: int
    lung_volume_mm3: float
    lung_volume_ml: float
    lung_fraction_of_cropped_volume: float


def validate_binary_mask(mask_array: np.ndarray) -> np.ndarray:
    values = np.unique(mask_array)
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(f"Mask must be binary with values 0 and 1. Found {values.tolist()}")
    if not np.any(mask_array):
        raise ValueError("Mask is empty.")
    return mask_array.astype(bool)


def validate_image_mask_geometry(image: sitk.Image, mask: sitk.Image, label: str = "mask") -> None:
    mismatches = compare_image_geometry(image, mask)
    if mismatches:
        raise ValueError(f"Geometry mismatch for {label}: " + "; ".join(mismatches))


def padding_from_mm(padding_mm: Sequence[float] | float | None, image: sitk.Image) -> PaddingZYX:
    """Convert physical padding in mm to outward voxel padding in z,y,x order."""

    values = _triple_float(padding_mm, default=0.0)
    if any(value < 0 for value in values):
        raise ValueError("padding-mm cannot be negative.")
    spacing_x, spacing_y, spacing_z = summarize_geometry(image).spacing_xyz
    pad_x = int(math.ceil(values[2] / spacing_x))
    pad_y = int(math.ceil(values[1] / spacing_y))
    pad_z = int(math.ceil(values[0] / spacing_z))
    return PaddingZYX(z=pad_z, y=pad_y, x=pad_x)


def padding_from_voxels(padding_voxels: Sequence[int] | int | None) -> PaddingZYX:
    values = _triple_int(padding_voxels, default=0)
    if any(value < 0 for value in values):
        raise ValueError("padding-voxels cannot be negative.")
    return PaddingZYX(z=values[0], y=values[1], x=values[2])


def combine_padding(mm_padding: PaddingZYX, voxel_padding: PaddingZYX) -> PaddingZYX:
    return PaddingZYX(
        z=mm_padding.z + voxel_padding.z,
        y=mm_padding.y + voxel_padding.y,
        x=mm_padding.x + voxel_padding.x,
    )


def compute_lung_bbox(mask_array: np.ndarray, padding: PaddingZYX | None = None) -> BBoxZYX:
    """Compute a half-open z,y,x bbox from every nonzero mask voxel."""

    mask = validate_binary_mask(mask_array)
    padding = padding or PaddingZYX(0, 0, 0)
    z_size, y_size, x_size = mask.shape
    coords = np.argwhere(mask)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1

    return BBoxZYX(
        z_min=max(int(z_min) - padding.z, 0),
        z_max=min(int(z_max) + padding.z, z_size),
        y_min=max(int(y_min) - padding.y, 0),
        y_max=min(int(y_max) + padding.y, y_size),
        x_min=max(int(x_min) - padding.x, 0),
        x_max=min(int(x_max) + padding.x, x_size),
    )


def physical_bbox_from_voxel_bbox(image: sitk.Image, bbox: BBoxZYX) -> dict[str, float]:
    """Transform half-open bbox corner coordinates to physical mm bounds.

    NumPy bbox intervals are z,y,x. SimpleITK continuous indices are x,y,z.
    The max corner uses the half-open stop index, which describes the crop
    region boundary and works with non-identity direction matrices.
    """

    points = []
    for x in (bbox.x_min, bbox.x_max):
        for y in (bbox.y_min, bbox.y_max):
            for z in (bbox.z_min, bbox.z_max):
                points.append(image.TransformContinuousIndexToPhysicalPoint((x, y, z)))
    arr = np.array(points, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    return {
        "xmin_mm": float(mins[0]),
        "xmax_mm": float(maxs[0]),
        "ymin_mm": float(mins[1]),
        "ymax_mm": float(maxs[1]),
        "zmin_mm": float(mins[2]),
        "zmax_mm": float(maxs[2]),
    }


def crop_image(image: sitk.Image, bbox: BBoxZYX) -> sitk.Image:
    return sitk.RegionOfInterest(image, size=bbox.size_xyz, index=bbox.start_xyz)


def compute_crop_stats(mask_array: np.ndarray, bbox: BBoxZYX, image: sitk.Image) -> CropStats:
    mask = validate_binary_mask(mask_array)
    geom = summarize_geometry(image)
    original_voxels = int(mask.size)
    cropped_voxels = int(np.prod(bbox.shape_zyx))
    voxel_volume_mm3 = float(np.prod(geom.spacing_xyz))
    lung_voxel_count = int(np.count_nonzero(mask))
    lung_volume_mm3 = lung_voxel_count * voxel_volume_mm3
    return CropStats(
        original_voxel_count=original_voxels,
        cropped_voxel_count=cropped_voxels,
        reduction_percentage=float((1.0 - cropped_voxels / original_voxels) * 100.0),
        original_volume_mm3=float(original_voxels * voxel_volume_mm3),
        cropped_volume_mm3=float(cropped_voxels * voxel_volume_mm3),
        bounding_dimensions_voxels_zyx=bbox.shape_zyx,
        bounding_dimensions_mm_zyx=(
            float(bbox.shape_zyx[0] * geom.spacing_xyz[2]),
            float(bbox.shape_zyx[1] * geom.spacing_xyz[1]),
            float(bbox.shape_zyx[2] * geom.spacing_xyz[0]),
        ),
        lung_voxel_count=lung_voxel_count,
        lung_volume_mm3=float(lung_volume_mm3),
        lung_volume_ml=float(lung_volume_mm3 / 1000.0),
        lung_fraction_of_cropped_volume=float(lung_voxel_count / cropped_voxels),
    )


def build_bbox_manifest(
    *,
    case_id: str,
    image_path: Path,
    mask_path: Path,
    image: sitk.Image,
    cropped_image: sitk.Image,
    bbox: BBoxZYX,
    padding_mm: Sequence[float] | None,
    padding_voxels_requested: Sequence[int] | None,
    effective_padding: PaddingZYX,
    stats: CropStats,
) -> dict[str, Any]:
    image_geom = summarize_geometry(image)
    crop_geom = summarize_geometry(cropped_image)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": case_id,
        "image_path": display_path(image_path),
        "mask_path": display_path(mask_path),
        "padding": {
            "requested_mm_zyx": list(_triple_float(padding_mm, default=0.0)),
            "requested_voxels_zyx": list(_triple_int(padding_voxels_requested, default=0)),
            "effective_voxels_zyx": effective_padding.to_dict(),
        },
        "voxel_bbox_half_open_zyx": bbox.to_dict(),
        "physical_bbox_mm": physical_bbox_from_voxel_bbox(image, bbox),
        "original_image_size_xyz": list(image_geom.size_xyz),
        "cropped_image_size_xyz": list(crop_geom.size_xyz),
        "cropped_spacing_xyz_mm": list(crop_geom.spacing_xyz),
        "cropped_origin_xyz_mm": list(crop_geom.origin_xyz),
        "cropped_direction": list(crop_geom.direction),
        "lung_voxel_count": stats.lung_voxel_count,
        "lung_volume_mm3": stats.lung_volume_mm3,
        "lung_volume_ml": stats.lung_volume_ml,
        "crop_statistics": {
            "original_voxel_count": stats.original_voxel_count,
            "cropped_voxel_count": stats.cropped_voxel_count,
            "reduction_percentage": stats.reduction_percentage,
            "original_volume_mm3": stats.original_volume_mm3,
            "cropped_volume_mm3": stats.cropped_volume_mm3,
            "bounding_dimensions_voxels_zyx": list(stats.bounding_dimensions_voxels_zyx),
            "bounding_dimensions_mm_zyx": list(stats.bounding_dimensions_mm_zyx),
            "lung_fraction_of_cropped_volume": stats.lung_fraction_of_cropped_volume,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coordinate_conventions": {
            "numpy_array_order": "z, y, x",
            "simpleitk_index_order": "x, y, z",
            "bbox_intervals": "half-open [min:max), matching Python slicing",
            "physical_coordinates": "millimeters in the image coordinate frame; direction matrices are respected",
        },
    }


def compute_bbox_and_crops(
    *,
    image_path: str | Path,
    mask_path: str | Path,
    case_id: str,
    padding_mm: Sequence[float] | float | None = None,
    padding_voxels: Sequence[int] | int | None = None,
    individual_mask_paths: Sequence[str | Path] = (),
    output_root: str | Path = DEFAULT_OUTPUTS_DIR,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    image_path = Path(image_path).expanduser().resolve(strict=False)
    mask_path = Path(mask_path).expanduser().resolve(strict=False)
    output_root = Path(output_root).expanduser().resolve(strict=False)
    bbox_dir = output_root / "bbox" / case_id
    crop_dir = output_root / "crops" / case_id

    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    validate_image_mask_geometry(image, mask, "combined lung mask")
    mask_array = validate_binary_mask(sitk.GetArrayFromImage(mask))
    mm_pad = padding_from_mm(padding_mm, image)
    voxel_pad = padding_from_voxels(padding_voxels)
    effective_padding = combine_padding(mm_pad, voxel_pad)
    bbox = compute_lung_bbox(mask_array, effective_padding)
    cropped_image = crop_image(image, bbox)
    cropped_mask = crop_image(mask, bbox)
    validate_image_mask_geometry(cropped_image, cropped_mask, "cropped combined lung mask")
    _validate_crop_contains_lung(mask_array, bbox)
    stats = compute_crop_stats(mask_array, bbox, image)

    individual_outputs = []
    individual_crops = []
    for item in individual_mask_paths:
        item_path = Path(item).expanduser().resolve(strict=False)
        individual = sitk.ReadImage(str(item_path))
        validate_image_mask_geometry(image, individual, item_path.name)
        validate_binary_mask(sitk.GetArrayFromImage(individual))
        individual_crop = crop_image(individual, bbox)
        validate_image_mask_geometry(cropped_image, individual_crop, f"cropped {item_path.name}")
        output_name = _crop_output_name(item_path)
        individual_outputs.append(crop_dir / output_name)
        individual_crops.append((individual_crop, crop_dir / output_name))

    manifest = build_bbox_manifest(
        case_id=case_id,
        image_path=image_path,
        mask_path=mask_path,
        image=image,
        cropped_image=cropped_image,
        bbox=bbox,
        padding_mm=_triple_float(padding_mm, default=0.0),
        padding_voxels_requested=_triple_int(padding_voxels, default=0),
        effective_padding=effective_padding,
        stats=stats,
    )
    manifest["outputs"] = {
        "bbox_json": display_path(bbox_dir / "bbox.json"),
        "image_crop": display_path(crop_dir / "image_crop.nii.gz"),
        "combined_lungs_crop": display_path(crop_dir / "combined_lungs_crop.nii.gz"),
        "individual_mask_crops": [display_path(path) for path in individual_outputs],
    }

    if dry_run:
        return {"status": "dry-run", "bbox": manifest}

    output_paths = [
        bbox_dir / "bbox.json",
        crop_dir / "image_crop.nii.gz",
        crop_dir / "combined_lungs_crop.nii.gz",
        *individual_outputs,
    ]
    _check_overwrite(output_paths, overwrite=overwrite)
    bbox_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(cropped_image, str(crop_dir / "image_crop.nii.gz"))
    sitk.WriteImage(cropped_mask, str(crop_dir / "combined_lungs_crop.nii.gz"))
    for individual_crop, output_path in individual_crops:
        sitk.WriteImage(individual_crop, str(output_path))
    (bbox_dir / "bbox.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _validate_crop_contains_lung(mask_array: np.ndarray, bbox: BBoxZYX) -> None:
    outside = mask_array.copy().astype(bool)
    outside[bbox.z_min:bbox.z_max, bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max] = False
    if np.any(outside):
        raise ValueError("Computed crop does not contain every lung voxel.")


def _check_overwrite(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(display_path(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {joined}")


def _crop_output_name(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = path.stem
    return f"{stem}_crop.nii.gz"


def _triple_float(value: Sequence[float] | float | None, *, default: float) -> tuple[float, float, float]:
    if value is None:
        return (default, default, default)
    if isinstance(value, (int, float)):
        return (float(value), float(value), float(value))
    if len(value) == 1:
        return (float(value[0]), float(value[0]), float(value[0]))
    if len(value) != 3:
        raise ValueError("Expected one value or three values in z,y,x order.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _triple_int(value: Sequence[int] | int | None, *, default: int) -> tuple[int, int, int]:
    if value is None:
        return (default, default, default)
    if isinstance(value, int):
        return (value, value, value)
    if len(value) == 1:
        return (int(value[0]), int(value[0]), int(value[0]))
    if len(value) != 3:
        raise ValueError("Expected one value or three values in z,y,x order.")
    return (int(value[0]), int(value[1]), int(value[2]))
