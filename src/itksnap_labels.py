from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import SimpleITK as sitk

from bounding_box import BBoxZYX, validate_bbox_bounds, validate_binary_mask
from medical_image_io import compare_image_geometry, normalize_modality, summarize_geometry
from project_paths import DEFAULT_OUTPUTS_DIR, display_path
from version import PIPELINE_VERSION


BACKGROUND_LABEL = 0
LEFT_LUNG_LABEL = 1
RIGHT_LUNG_LABEL = 2
BBOX_LABEL = 10
LABEL_MAPPING = {
    BACKGROUND_LABEL: "Clear Label",
    LEFT_LUNG_LABEL: "Lung Left",
    RIGHT_LUNG_LABEL: "Lung Right",
    BBOX_LABEL: "Lung Bounding Box",
}
OUTPUT_NAMES = (
    "lungs_multilabel.nii.gz",
    "bbox_label.nii.gz",
    "lungs_and_bbox_labels.nii.gz",
    "label_descriptions.txt",
    "launch_manifest.json",
)


def build_mri_lung_multilabel(
    reference: sitk.Image,
    left_lung: sitk.Image,
    right_lung: sitk.Image,
) -> sitk.Image:
    left = _validated_mask_array(reference, left_lung, "left lung")
    right = _validated_mask_array(reference, right_lung, "right lung")
    if np.any(left & right):
        raise ValueError("Left and right lung masks overlap.")
    labels = np.zeros(left.shape, dtype=np.uint8)
    labels[left] = LEFT_LUNG_LABEL
    labels[right] = RIGHT_LUNG_LABEL
    return _label_image(labels, reference)


def build_bbox_shell(reference: sitk.Image, bbox: BBoxZYX) -> sitk.Image:
    shape = tuple(reversed(reference.GetSize()))
    validate_bbox_bounds(bbox, shape)
    shell = np.zeros(shape, dtype=np.uint8)
    z0, z1 = bbox.z_min, bbox.z_max
    y0, y1 = bbox.y_min, bbox.y_max
    x0, x1 = bbox.x_min, bbox.x_max
    shell[z0, y0:y1, x0:x1] = BBOX_LABEL
    shell[z1 - 1, y0:y1, x0:x1] = BBOX_LABEL
    shell[z0:z1, y0, x0:x1] = BBOX_LABEL
    shell[z0:z1, y1 - 1, x0:x1] = BBOX_LABEL
    shell[z0:z1, y0:y1, x0] = BBOX_LABEL
    shell[z0:z1, y0:y1, x1 - 1] = BBOX_LABEL
    return _label_image(shell, reference)


def combine_lungs_and_bbox(lungs: sitk.Image, bbox_shell: sitk.Image) -> sitk.Image:
    mismatches = compare_image_geometry(lungs, bbox_shell)
    if mismatches:
        raise ValueError("Lung/bbox geometry mismatch: " + "; ".join(mismatches))
    lungs_array = sitk.GetArrayFromImage(lungs).astype(np.uint8, copy=True)
    bbox_array = sitk.GetArrayFromImage(bbox_shell)
    if not np.all(np.isin(np.unique(lungs_array), [0, 1, 2])):
        raise ValueError("Lung label image contains unexpected values.")
    if not np.all(np.isin(np.unique(bbox_array), [0, BBOX_LABEL])):
        raise ValueError("BBox label image contains unexpected values.")
    lungs_array[bbox_array == BBOX_LABEL] = BBOX_LABEL
    return _label_image(lungs_array, lungs)


def bbox_from_manifest(data: Mapping[str, Any], reference: sitk.Image, case_id: str) -> BBoxZYX:
    if data.get("case_id") and data["case_id"] != case_id:
        raise ValueError("bbox case_id does not match requested case_id.")
    values = data.get("voxel_bbox_half_open_zyx")
    required = ("z_min", "z_max", "y_min", "y_max", "x_min", "x_max")
    if not isinstance(values, Mapping) or any(key not in values for key in required):
        raise ValueError("bbox JSON is missing voxel_bbox_half_open_zyx values.")
    bbox = BBoxZYX(**{key: int(values[key]) for key in required})
    validate_bbox_bounds(bbox, tuple(reversed(reference.GetSize())))
    recorded_size = data.get("original_image_size_xyz")
    if recorded_size is not None and tuple(int(v) for v in recorded_size) != reference.GetSize():
        raise ValueError("bbox original image size does not match the reference image.")
    return bbox


def label_description_text() -> str:
    return "\n".join([
        "##########################################################",
        "# ITK-SNAP Label Description File",
        "# File format: IDX R G B A VIS MSH \"LABEL\"",
        '0   0   0   0   0 0 0 "Clear Label"',
        '1 230  85  40   1 1 1 "Lung Left"',
        '2  40 160 220   1 1 1 "Lung Right"',
        '10 255 220   0   1 1 1 "Lung Bounding Box"',
        "",
    ])


def prepare_itksnap_products(
    *,
    image_path: str | Path,
    left_lung_path: str | Path,
    right_lung_path: str | Path,
    bbox_json_path: str | Path,
    case_id: str,
    modality: str,
    output_root: str | Path = DEFAULT_OUTPUTS_DIR / "itksnap",
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    normalized_modality = normalize_modality(modality, allow_auto=False)
    if normalized_modality != "MR":
        raise ValueError("This checkpoint supports MRI lung labels only.")
    paths = {
        "image": _existing_file(image_path, "reference image"),
        "left": _existing_file(left_lung_path, "left lung mask"),
        "right": _existing_file(right_lung_path, "right lung mask"),
        "bbox": _existing_file(bbox_json_path, "bbox JSON"),
    }
    reference = sitk.ReadImage(str(paths["image"]))
    left = sitk.ReadImage(str(paths["left"]))
    right = sitk.ReadImage(str(paths["right"]))
    lungs = build_mri_lung_multilabel(reference, left, right)
    bbox_data = json.loads(paths["bbox"].read_text())
    bbox = bbox_from_manifest(bbox_data, reference, case_id)
    bbox_shell = build_bbox_shell(reference, bbox)
    combined = combine_lungs_and_bbox(lungs, bbox_shell)

    case_dir = Path(output_root).expanduser().resolve(strict=False) / case_id
    outputs = {name: case_dir / name for name in OUTPUT_NAMES}
    manifest = _build_manifest(
        case_id=case_id,
        modality=normalized_modality,
        image_path=paths["image"],
        outputs=outputs,
        reference=reference,
        lungs=lungs,
        bbox_shell=bbox_shell,
        combined=combined,
        bbox=bbox,
        status="dry-run" if dry_run else "complete",
    )
    if dry_run:
        return manifest
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing ITK-SNAP outputs: "
            + ", ".join(display_path(path) for path in existing)
        )
    case_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(lungs, str(outputs["lungs_multilabel.nii.gz"]))
    sitk.WriteImage(bbox_shell, str(outputs["bbox_label.nii.gz"]))
    sitk.WriteImage(combined, str(outputs["lungs_and_bbox_labels.nii.gz"]))
    outputs["label_descriptions.txt"].write_text(label_description_text())
    outputs["launch_manifest.json"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _validated_mask_array(reference: sitk.Image, mask: sitk.Image, name: str) -> np.ndarray:
    mismatches = compare_image_geometry(reference, mask)
    if mismatches:
        raise ValueError(f"Geometry mismatch for {name}: " + "; ".join(mismatches))
    return validate_binary_mask(sitk.GetArrayFromImage(mask))


def _label_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.uint8, copy=False))
    image.CopyInformation(reference)
    return sitk.Cast(image, sitk.sitkUInt8)


def _existing_file(path: str | Path, name: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name.capitalize()} does not exist: {display_path(resolved)}")
    return resolved


def _counts(image: sitk.Image, values: Sequence[int]) -> dict[str, int]:
    array = sitk.GetArrayViewFromImage(image)
    return {str(value): int(np.count_nonzero(array == value)) for value in values}


def _build_manifest(**items: Any) -> dict[str, Any]:
    reference = items["reference"]
    geom = summarize_geometry(reference)
    outputs = items["outputs"]
    return {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": items["case_id"],
        "modality": items["modality"],
        "image_path": display_path(items["image_path"]),
        "generated_files": {key: display_path(value) for key, value in outputs.items()},
        "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
        "bbox_convention": "half-open NumPy z,y,x [min:max); shell faces use min and max-1",
        "bbox": items["bbox"].to_dict(),
        "bbox_overlap_priority": "Label 10 replaces lung labels only in the combined viewer image.",
        "image_geometry": {
            "size_xyz": list(geom.size_xyz),
            "spacing_xyz": list(geom.spacing_xyz),
            "origin_xyz": list(geom.origin_xyz),
            "direction": list(geom.direction),
        },
        "label_voxel_counts": {
            "lungs_multilabel": _counts(items["lungs"], [0, 1, 2]),
            "bbox_label": _counts(items["bbox_shell"], [0, 10]),
            "lungs_and_bbox_labels": _counts(items["combined"], [0, 1, 2, 10]),
        },
        "output_pixel_types": {
            "lungs_multilabel": items["lungs"].GetPixelIDTypeAsString(),
            "bbox_label": items["bbox_shell"].GetPixelIDTypeAsString(),
            "lungs_and_bbox_labels": items["combined"].GetPixelIDTypeAsString(),
        },
        "warnings": ["The combined label image is for visualization, not quantitative analysis."],
        "completion_status": items["status"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
