from pathlib import Path
import sys

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from bounding_box import (  # noqa: E402
    PaddingZYX,
    compute_bbox_and_crops,
    compute_crop_stats,
    compute_lung_bbox,
    crop_image,
    padding_from_mm,
    physical_bbox_from_voxel_bbox,
    validate_binary_mask,
    validate_image_mask_geometry,
)


def make_image(
    shape_zyx=(6, 7, 8),
    spacing_xyz=(1.0, 2.0, 3.0),
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> sitk.Image:
    array = np.arange(np.prod(shape_zyx), dtype=np.int16).reshape(shape_zyx)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing_xyz)
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection(direction)
    return image


def write_image(path: Path, image: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def mask_image(mask: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(reference)
    return image


def test_single_voxel_bbox_half_open():
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[2, 3, 4] = 1

    bbox = compute_lung_bbox(mask)

    assert bbox.to_dict() == {
        "z_min": 2,
        "z_max": 3,
        "y_min": 3,
        "y_max": 4,
        "x_min": 4,
        "x_max": 5,
    }
    assert bbox.shape_zyx == (1, 1, 1)


def test_small_cube_bbox():
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[1:4, 2:6, 3:7] = 1

    bbox = compute_lung_bbox(mask)

    assert bbox.shape_zyx == (3, 4, 4)
    assert bbox.to_dict()["z_max"] == 4


def test_padding_mm_anisotropic_spacing():
    image = make_image(spacing_xyz=(1.0, 2.0, 3.0))

    padding = padding_from_mm((5.0, 5.0, 5.0), image)

    assert padding == PaddingZYX(z=2, y=3, x=5)


def test_padding_voxels_and_boundary_clamping():
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1

    bbox = compute_lung_bbox(mask, PaddingZYX(z=3, y=3, x=3))

    assert bbox.z_min == 0
    assert bbox.y_min == 0
    assert bbox.x_min == 0
    assert bbox.z_max == 5


def test_non_identity_direction_physical_bbox():
    image = make_image(
        spacing_xyz=(1.0, 2.0, 3.0),
        direction=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[1:3, 2:4, 3:5] = 1
    bbox = compute_lung_bbox(mask)

    physical = physical_bbox_from_voxel_bbox(image, bbox)

    assert physical["xmin_mm"] == pytest.approx(2.0)
    assert physical["xmax_mm"] == pytest.approx(6.0)
    assert physical["ymin_mm"] == pytest.approx(23.0)
    assert physical["ymax_mm"] == pytest.approx(25.0)
    assert physical["zmin_mm"] == pytest.approx(33.0)
    assert physical["zmax_mm"] == pytest.approx(39.0)


def test_crop_geometry_and_origin_update():
    image = make_image()
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[1:4, 2:6, 3:7] = 1
    bbox = compute_lung_bbox(mask)

    cropped = crop_image(image, bbox)

    assert cropped.GetSize() == (4, 4, 3)
    assert cropped.GetSpacing() == image.GetSpacing()
    assert cropped.GetDirection() == image.GetDirection()
    assert cropped.GetPixelID() == image.GetPixelID()
    assert cropped.GetOrigin() == pytest.approx(
        image.TransformIndexToPhysicalPoint((3, 2, 1))
    )


def test_half_open_slicing_matches_crop_size():
    image = make_image()
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[1:4, 2:6, 3:7] = 1
    bbox = compute_lung_bbox(mask)
    cropped = crop_image(image, bbox)
    array_crop = sitk.GetArrayFromImage(image)[
        bbox.z_min:bbox.z_max,
        bbox.y_min:bbox.y_max,
        bbox.x_min:bbox.x_max,
    ]

    assert sitk.GetArrayFromImage(cropped).shape == array_crop.shape


def test_empty_mask_rejected():
    with pytest.raises(ValueError, match="empty"):
        compute_lung_bbox(np.zeros((3, 3, 3), dtype=np.uint8))


def test_geometry_mismatch_rejected():
    image = make_image()
    mask = mask_image(np.ones((6, 7, 8), dtype=np.uint8), image)
    mask.SetSpacing((9.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="Geometry mismatch"):
        validate_image_mask_geometry(image, mask)


def test_binary_validation_rejected():
    with pytest.raises(ValueError, match="binary"):
        validate_binary_mask(np.array([0, 2], dtype=np.uint8))


def test_crop_stats():
    image = make_image(spacing_xyz=(1.0, 2.0, 3.0))
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[1:4, 2:6, 3:7] = 1
    bbox = compute_lung_bbox(mask)

    stats = compute_crop_stats(mask, bbox, image)

    assert stats.original_voxel_count == 336
    assert stats.cropped_voxel_count == 48
    assert stats.reduction_percentage == pytest.approx(85.7142857)
    assert stats.lung_volume_ml == pytest.approx(0.288)
    assert stats.bounding_dimensions_mm_zyx == pytest.approx((9.0, 8.0, 4.0))


def test_dry_run_creates_no_files(tmp_path):
    image = make_image()
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[1:4, 2:6, 3:7] = 1
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    write_image(image_path, image)
    write_image(mask_path, mask_image(mask, image))
    output_root = tmp_path / "outputs"

    result = compute_bbox_and_crops(
        image_path=image_path,
        mask_path=mask_path,
        case_id="case",
        output_root=output_root,
        dry_run=True,
    )

    assert result["status"] == "dry-run"
    assert not output_root.exists()


def test_overwrite_protection(tmp_path):
    image = make_image()
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[1:4, 2:6, 3:7] = 1
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    write_image(image_path, image)
    write_image(mask_path, mask_image(mask, image))
    output_root = tmp_path / "outputs"
    existing = output_root / "bbox" / "case" / "bbox.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}\n")

    with pytest.raises(FileExistsError):
        compute_bbox_and_crops(
            image_path=image_path,
            mask_path=mask_path,
            case_id="case",
            output_root=output_root,
        )


def test_bbox_json_fields_and_individual_mask_crop(tmp_path):
    image = make_image()
    combined = np.zeros((6, 7, 8), dtype=np.uint8)
    combined[1:4, 2:6, 3:7] = 1
    individual = np.zeros_like(combined)
    individual[1:4, 2:6, 3:5] = 1
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "combined_lungs.nii.gz"
    individual_path = tmp_path / "lung_left.nii.gz"
    write_image(image_path, image)
    write_image(mask_path, mask_image(combined, image))
    write_image(individual_path, mask_image(individual, image))
    output_root = tmp_path / "outputs"

    result = compute_bbox_and_crops(
        image_path=image_path,
        mask_path=mask_path,
        case_id="case",
        padding_voxels=(1, 1, 1),
        individual_mask_paths=[individual_path],
        output_root=output_root,
    )

    assert (output_root / "bbox" / "case" / "bbox.json").exists()
    assert (output_root / "crops" / "case" / "image_crop.nii.gz").exists()
    assert (output_root / "crops" / "case" / "combined_lungs_crop.nii.gz").exists()
    assert (output_root / "crops" / "case" / "lung_left_crop.nii.gz").exists()
    assert result["case_id"] == "case"
    assert result["voxel_bbox_half_open_zyx"] == {
        "z_min": 0,
        "z_max": 5,
        "y_min": 1,
        "y_max": 7,
        "x_min": 2,
        "x_max": 8,
    }
    assert "patient" not in " ".join(result.keys()).lower()

