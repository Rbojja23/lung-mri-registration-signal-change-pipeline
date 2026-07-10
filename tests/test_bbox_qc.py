from pathlib import Path
import sys
import json

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from bbox_qc import (  # noqa: E402
    aspect_for_view,
    bbox_contains_mask,
    rectangle_for_view,
    run_bbox_qc,
    validate_bbox_inputs,
)
from bounding_box import BBoxZYX, crop_image  # noqa: E402


def make_image(shape_zyx=(8, 10, 12), spacing_xyz=(1.0, 2.0, 3.0)) -> sitk.Image:
    array = np.arange(np.prod(shape_zyx), dtype=np.float32).reshape(shape_zyx)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing_xyz)
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def make_mask(shape_zyx=(8, 10, 12)) -> np.ndarray:
    mask = np.zeros(shape_zyx, dtype=np.uint8)
    mask[2:7, 3:8, 4:10] = 1
    return mask


def mask_image(mask: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(reference)
    return image


def write_image(path: Path, image: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def write_bbox(path: Path, bbox: BBoxZYX, case_id="case", size=(12, 10, 8)) -> None:
    path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "voxel_bbox_half_open_zyx": bbox.to_dict(),
                "physical_bbox_mm": {},
                "original_image_size_xyz": list(size),
            }
        )
        + "\n"
    )


def test_bbox_validation_and_mask_containment():
    image = make_image()
    mask = make_mask()
    bbox = BBoxZYX(2, 7, 3, 8, 4, 10)

    validation = validate_bbox_inputs(
        image,
        mask_image(mask, image),
        {
            "case_id": "case",
            "voxel_bbox_half_open_zyx": bbox.to_dict(),
            "original_image_size_xyz": list(image.GetSize()),
        },
        case_id="case",
    )

    assert validation.all_mask_voxels_contained
    assert validation.dimensions_voxels_zyx == (5, 5, 6)


def test_invalid_bbox_rejection():
    image = make_image()
    mask = mask_image(make_mask(), image)
    data = {"voxel_bbox_half_open_zyx": BBoxZYX(0, 99, 0, 2, 0, 2).to_dict()}

    with pytest.raises(ValueError, match="Invalid bbox"):
        validate_bbox_inputs(image, mask, data, case_id="case")


def test_bbox_must_contain_mask():
    mask = make_mask()
    bbox = BBoxZYX(2, 6, 3, 8, 4, 10)

    assert not bbox_contains_mask(mask.astype(bool), bbox)


def test_half_open_axial_rectangle_coordinates():
    bbox = BBoxZYX(2, 7, 3, 8, 4, 10)

    assert rectangle_for_view(bbox, "axial") == (3.5, 2.5, 6, 5)


def test_coronal_rectangle_coordinates():
    bbox = BBoxZYX(2, 7, 3, 8, 4, 10)

    assert rectangle_for_view(bbox, "coronal") == (3.5, 1.5, 6, 5)


def test_sagittal_rectangle_coordinates():
    bbox = BBoxZYX(2, 7, 3, 8, 4, 10)

    assert rectangle_for_view(bbox, "sagittal") == (2.5, 1.5, 5, 5)


def test_anisotropic_aspect_ratios():
    image = make_image(spacing_xyz=(1.0, 2.0, 4.0))

    assert aspect_for_view(image, "axial") == 2.0
    assert aspect_for_view(image, "coronal") == 4.0
    assert aspect_for_view(image, "sagittal") == 2.0


def test_boundary_clamped_bbox_warning():
    image = make_image()
    mask = np.zeros((8, 10, 12), dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1
    bbox = BBoxZYX(0, 2, 0, 2, 0, 2)

    validation = validate_bbox_inputs(
        image,
        mask_image(mask, image),
        {"voxel_bbox_half_open_zyx": bbox.to_dict()},
        case_id="case",
    )

    assert validation.warnings


def test_dry_run_writes_nothing(tmp_path):
    image = make_image()
    mask = make_mask()
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    bbox_path = tmp_path / "bbox.json"
    output_dir = tmp_path / "figures"
    write_image(image_path, image)
    write_image(mask_path, mask_image(mask, image))
    write_bbox(bbox_path, BBoxZYX(2, 7, 3, 8, 4, 10), size=image.GetSize())

    result = run_bbox_qc(
        image_path=image_path,
        mask_path=mask_path,
        bbox_json_path=bbox_path,
        case_id="case",
        modality="mr",
        output_dir=output_dir,
        dry_run=True,
    )

    assert result["status"] == "dry-run"
    assert not output_dir.exists()


def test_overwrite_protection(tmp_path):
    image = make_image()
    mask = make_mask()
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    bbox_path = tmp_path / "bbox.json"
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    (output_dir / "bounding_box_overlay.png").write_text("x")
    write_image(image_path, image)
    write_image(mask_path, mask_image(mask, image))
    write_bbox(bbox_path, BBoxZYX(2, 7, 3, 8, 4, 10), size=image.GetSize())

    with pytest.raises(FileExistsError):
        run_bbox_qc(
            image_path=image_path,
            mask_path=mask_path,
            bbox_json_path=bbox_path,
            case_id="case",
            modality="mr",
            output_dir=output_dir,
        )


def test_headless_png_creation_and_json(tmp_path):
    image = make_image()
    mask = make_mask()
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    bbox_path = tmp_path / "bbox.json"
    output_dir = tmp_path / "figures"
    write_image(image_path, image)
    write_image(mask_path, mask_image(mask, image))
    write_bbox(bbox_path, BBoxZYX(2, 7, 3, 8, 4, 10), size=image.GetSize())

    result = run_bbox_qc(
        image_path=image_path,
        mask_path=mask_path,
        bbox_json_path=bbox_path,
        case_id="case",
        modality="mr",
        output_dir=output_dir,
    )

    assert (output_dir / "bounding_box_overlay.png").exists()
    assert (output_dir / "bounding_box_qc.json").exists()
    assert result["all_mask_voxels_contained"]


def test_crop_comparison(tmp_path):
    image = make_image()
    mask = make_mask()
    bbox = BBoxZYX(2, 7, 3, 8, 4, 10)
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    bbox_path = tmp_path / "bbox.json"
    crop_image_path = tmp_path / "image_crop.nii.gz"
    crop_mask_path = tmp_path / "mask_crop.nii.gz"
    output_dir = tmp_path / "figures"
    write_image(image_path, image)
    write_image(mask_path, mask_image(mask, image))
    write_image(crop_image_path, crop_image(image, bbox))
    write_image(crop_mask_path, crop_image(mask_image(mask, image), bbox))
    write_bbox(bbox_path, bbox, size=image.GetSize())

    run_bbox_qc(
        image_path=image_path,
        mask_path=mask_path,
        bbox_json_path=bbox_path,
        case_id="case",
        modality="mr",
        crop_image_path=crop_image_path,
        crop_mask_path=crop_mask_path,
        output_dir=output_dir,
    )

    assert (output_dir / "bounding_box_crop_comparison.png").exists()

