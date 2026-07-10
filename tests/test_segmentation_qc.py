from pathlib import Path
import sys

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from segmentation_qc import (  # noqa: E402
    IndividualMask,
    apply_ct_window,
    binary_contour,
    compute_mask_metrics,
    compute_qc,
    mr_display_limits,
    render_qc_figures,
    run_segmentation_qc,
    select_representative_slices,
)


def make_image(shape_zyx=(8, 10, 12), spacing_xyz=(1.0, 2.0, 3.0)) -> sitk.Image:
    array = np.arange(np.prod(shape_zyx), dtype=np.float32).reshape(shape_zyx)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing_xyz)
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def make_mask_array(shape_zyx=(8, 10, 12)) -> np.ndarray:
    mask = np.zeros(shape_zyx, dtype=np.uint8)
    mask[2:7, 3:8, 4:10] = 1
    return mask


def image_from_mask(mask: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(reference)
    return image


def write_image(path: Path, image: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def test_image_mask_geometry_match():
    image = make_image()
    mask = image_from_mask(make_mask_array(), image)

    qc = compute_qc(image, mask)

    assert qc.metrics.voxel_count == 150


def test_geometry_mismatch_rejection():
    image = make_image()
    mask = image_from_mask(make_mask_array(), image)
    mask.SetSpacing((9.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="Geometry mismatch"):
        compute_qc(image, mask)


def test_empty_mask_rejection():
    image = make_image()
    mask = image_from_mask(np.zeros((8, 10, 12), dtype=np.uint8), image)

    with pytest.raises(ValueError, match="empty"):
        compute_qc(image, mask)


def test_representative_slice_selection_uses_extent():
    mask = make_mask_array()

    selected = select_representative_slices(mask.astype(bool))

    assert selected.axial_zyx[0] >= 2
    assert selected.axial_zyx[-1] <= 6
    assert selected.coronal_y in range(3, 8)


def test_physical_aspect_ratio_inputs_are_spacing_based(tmp_path):
    image = make_image(spacing_xyz=(1.0, 2.0, 4.0))
    mask = image_from_mask(make_mask_array(), image)
    qc = compute_qc(image, mask)

    figures = render_qc_figures(
        image=image,
        combined_mask=mask,
        qc=qc,
        modality="mr",
        case_id="synthetic",
        output_dir=tmp_path,
    )

    assert figures["overlay"].exists()
    assert figures["contours"].exists()


def test_ct_display_window_clipping():
    array = np.array([-2000.0, -1000.0, 0.0, 1000.0])

    clipped = apply_ct_window(array, center=-500.0, width=1000.0)

    assert clipped.tolist() == [-1000.0, -1000.0, 0.0, 0.0]


def test_mri_percentile_scaling():
    array = np.array([0, 1, 2, 3, 1000], dtype=float)

    lower, upper = mr_display_limits(array, 20, 80)

    assert lower == pytest.approx(0.8)
    assert upper == pytest.approx(202.4)


def test_binary_contour_generation():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    contour = binary_contour(mask)

    assert contour.sum() == 8
    assert not contour[2, 2]


def test_voxel_count_and_physical_volume():
    image = make_image(spacing_xyz=(1.0, 2.0, 3.0))
    mask = make_mask_array()

    metrics = compute_mask_metrics(mask.astype(bool), image)

    assert metrics.voxel_count == 150
    assert metrics.volume_mm3 == pytest.approx(900.0)
    assert metrics.volume_ml == pytest.approx(0.9)


def test_bounding_dimensions_with_anisotropic_spacing():
    image = make_image(spacing_xyz=(1.0, 2.0, 3.0))
    metrics = compute_mask_metrics(make_mask_array().astype(bool), image)

    assert metrics.physical_dimensions_mm == {"z": 15.0, "y": 10.0, "x": 6.0}


def test_voxel_and_physical_centroid():
    image = make_image(spacing_xyz=(1.0, 2.0, 3.0))
    metrics = compute_mask_metrics(make_mask_array().astype(bool), image)

    assert metrics.centroid_zyx == pytest.approx([4.0, 5.0, 6.5])
    assert metrics.centroid_xyz == pytest.approx([6.5, 5.0, 4.0])
    assert metrics.centroid_physical_xyz_mm == pytest.approx([16.0, 30.0, 42.0])


def test_connected_component_count():
    image = make_image()
    mask = np.zeros((8, 10, 12), dtype=bool)
    mask[1, 1, 1] = True
    mask[6, 8, 10] = True

    metrics = compute_mask_metrics(mask, image)

    assert metrics.connected_component_count == 2


def test_boundary_touch_detection():
    image = make_image()
    mask = np.zeros((8, 10, 12), dtype=bool)
    mask[0, 1, 1] = True
    mask[7, 9, 11] = True

    metrics = compute_mask_metrics(mask, image)

    assert metrics.touches_boundaries["z_min"]
    assert metrics.touches_boundaries["z_max"]
    assert metrics.touches_boundaries["y_max"]
    assert metrics.touches_boundaries["x_max"]


def test_union_validation_for_individual_masks():
    image = make_image()
    combined = make_mask_array()
    left = np.zeros_like(combined)
    right = np.zeros_like(combined)
    left[:, :, :7] = combined[:, :, :7]
    right[:, :, 7:] = combined[:, :, 7:]
    individual = [
        (IndividualMask("lung_left", Path("lung_left.nii.gz")), image_from_mask(left, image)),
        (IndividualMask("lung_right", Path("lung_right.nii.gz")), image_from_mask(right, image)),
    ]

    qc = compute_qc(image, image_from_mask(combined, image), individual_masks=individual)

    assert qc.individual_union_equals_combined is True
    assert qc.individual_overlap_voxels == 0


def test_detection_of_unexpected_overlap():
    image = make_image()
    combined = make_mask_array()
    left = combined.copy()
    right = combined.copy()
    individual = [
        (IndividualMask("lung_left", Path("lung_left.nii.gz")), image_from_mask(left, image)),
        (IndividualMask("lung_right", Path("lung_right.nii.gz")), image_from_mask(right, image)),
    ]

    qc = compute_qc(image, image_from_mask(combined, image), individual_masks=individual)

    assert qc.individual_overlap_voxels == int(combined.sum())
    assert any("overlap" in warning for warning in qc.warnings)


def test_dry_run_creates_no_files(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    output_dir = tmp_path / "figures"
    image = make_image()
    write_image(image_path, image)
    write_image(mask_path, image_from_mask(make_mask_array(), image))

    result = run_segmentation_qc(
        image_path=image_path,
        mask_path=mask_path,
        modality="mr",
        case_id="case",
        output_dir=output_dir,
        dry_run=True,
    )

    assert result["status"] == "dry-run"
    assert not output_dir.exists()


def test_overwrite_protection(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    (output_dir / "segmentation_qc.json").write_text("{}\n")
    image = make_image()
    write_image(image_path, image)
    write_image(mask_path, image_from_mask(make_mask_array(), image))

    with pytest.raises(FileExistsError):
        run_segmentation_qc(
            image_path=image_path,
            mask_path=mask_path,
            modality="ct",
            case_id="case",
            output_dir=output_dir,
        )


def test_safe_json_fields_and_headless_png_generation(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    output_dir = tmp_path / "figures"
    image = make_image()
    write_image(image_path, image)
    write_image(mask_path, image_from_mask(make_mask_array(), image))

    result = run_segmentation_qc(
        image_path=image_path,
        mask_path=mask_path,
        modality="mr",
        case_id="case",
        output_dir=output_dir,
    )

    assert (output_dir / "segmentation_overlay.png").exists()
    assert (output_dir / "segmentation_contours.png").exists()
    assert (output_dir / "segmentation_qc.json").exists()
    assert "patient" not in " ".join(result.keys()).lower()
    assert result["case_id"] == "case"

