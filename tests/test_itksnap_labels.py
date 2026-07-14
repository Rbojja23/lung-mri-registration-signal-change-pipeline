import json
from pathlib import Path
import sys

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import pipeline  # noqa: E402
from bounding_box import BBoxZYX  # noqa: E402
from itksnap_labels import (  # noqa: E402
    BBOX_LABEL,
    build_bbox_shell,
    build_mri_lung_multilabel,
    combine_lungs_and_bbox,
    label_description_text,
    prepare_itksnap_products,
)
from medical_image_io import geometry_matches  # noqa: E402


def make_reference(
    shape_zyx=(5, 6, 7),
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> sitk.Image:
    image = sitk.GetImageFromArray(np.zeros(shape_zyx, dtype=np.int16))
    image.SetSpacing((1.2, 2.3, 3.4))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection(direction)
    return image


def mask_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def lung_arrays(shape=(5, 6, 7)):
    left = np.zeros(shape, dtype=np.uint8)
    right = np.zeros(shape, dtype=np.uint8)
    left[1:4, 1:3, 1:3] = 1
    right[1:4, 3:5, 4:6] = 1
    return left, right


def write_inputs(tmp_path: Path):
    reference = make_reference()
    left, right = lung_arrays()
    paths = {
        "image": tmp_path / "image.nii.gz",
        "left": tmp_path / "left.nii.gz",
        "right": tmp_path / "right.nii.gz",
        "bbox": tmp_path / "bbox.json",
    }
    sitk.WriteImage(reference, str(paths["image"]))
    sitk.WriteImage(mask_image(left, reference), str(paths["left"]))
    sitk.WriteImage(mask_image(right, reference), str(paths["right"]))
    paths["bbox"].write_text(json.dumps({
        "case_id": "case",
        "original_image_size_xyz": list(reference.GetSize()),
        "voxel_bbox_half_open_zyx": BBoxZYX(0, 5, 0, 6, 0, 7).to_dict(),
    }))
    return paths


def test_mri_multilabel_values_type_geometry_and_semantic_assignment():
    direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    reference = make_reference(direction=direction)
    left, right = lung_arrays()
    output = build_mri_lung_multilabel(
        reference, mask_image(left, reference), mask_image(right, reference)
    )
    array = sitk.GetArrayFromImage(output)

    assert set(np.unique(array)) == {0, 1, 2}
    assert np.all(array[left.astype(bool)] == 1)
    assert np.all(array[right.astype(bool)] == 2)
    assert output.GetPixelID() == sitk.sitkUInt8
    assert geometry_matches(reference, output)
    assert output.GetDirection() == direction


def test_mask_validation_rejections():
    reference = make_reference()
    left, right = lung_arrays()
    overlapping = right.copy()
    overlapping[left.astype(bool)] = 1
    with pytest.raises(ValueError, match="overlap"):
        build_mri_lung_multilabel(
            reference, mask_image(left, reference), mask_image(overlapping, reference)
        )
    with pytest.raises(ValueError, match="empty"):
        build_mri_lung_multilabel(
            reference, mask_image(np.zeros_like(left), reference), mask_image(right, reference)
        )
    nonbinary = left.copy()
    nonbinary[1, 1, 1] = 3
    with pytest.raises(ValueError, match="binary"):
        build_mri_lung_multilabel(
            reference, mask_image(nonbinary, reference), mask_image(right, reference)
        )
    wrong_geometry = mask_image(left, reference)
    wrong_geometry.SetOrigin((99.0, 20.0, 30.0))
    with pytest.raises(ValueError, match="Geometry mismatch"):
        build_mri_lung_multilabel(
            reference, wrong_geometry, mask_image(right, reference)
        )


def test_six_face_shell_half_open_and_image_boundaries():
    reference = make_reference()
    bbox = BBoxZYX(0, 5, 0, 6, 0, 7)
    shell = sitk.GetArrayFromImage(build_bbox_shell(reference, bbox))

    assert np.all(shell[0, :, :] == BBOX_LABEL)
    assert np.all(shell[4, :, :] == BBOX_LABEL)
    assert np.all(shell[:, 0, :] == BBOX_LABEL)
    assert np.all(shell[:, 5, :] == BBOX_LABEL)
    assert np.all(shell[:, :, 0] == BBOX_LABEL)
    assert np.all(shell[:, :, 6] == BBOX_LABEL)
    assert shell[2, 3, 3] == 0
    assert set(np.unique(shell)) == {0, 10}


@pytest.mark.parametrize("bbox", [
    BBoxZYX(2, 3, 1, 5, 1, 6),
    BBoxZYX(1, 4, 2, 3, 1, 6),
    BBoxZYX(1, 4, 1, 5, 3, 4),
])
def test_one_voxel_bbox_dimensions(bbox):
    shell = sitk.GetArrayFromImage(build_bbox_shell(make_reference(), bbox))
    region = shell[bbox.z_min:bbox.z_max, bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]
    assert np.all(region == BBOX_LABEL)


@pytest.mark.parametrize("bbox", [
    BBoxZYX(-1, 2, 0, 2, 0, 2),
    BBoxZYX(2, 2, 0, 2, 0, 2),
    BBoxZYX(0, 6, 0, 2, 0, 2),
])
def test_invalid_bbox_rejected(bbox):
    with pytest.raises(ValueError, match="Invalid bbox"):
        build_bbox_shell(make_reference(), bbox)


def test_combined_bbox_priority_does_not_mutate_lung_image():
    reference = make_reference()
    left, right = lung_arrays()
    lungs = build_mri_lung_multilabel(
        reference, mask_image(left, reference), mask_image(right, reference)
    )
    before = sitk.GetArrayFromImage(lungs).copy()
    bbox = build_bbox_shell(reference, BBoxZYX(1, 4, 1, 5, 1, 6))
    combined = sitk.GetArrayFromImage(combine_lungs_and_bbox(lungs, bbox))
    shell = sitk.GetArrayFromImage(bbox) == BBOX_LABEL

    assert np.all(combined[shell] == BBOX_LABEL)
    assert np.array_equal(sitk.GetArrayFromImage(lungs), before)


def test_label_description_values_and_names():
    text = label_description_text()
    rows = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert {int(row.split()[0]) for row in rows} == {0, 1, 2, 10}
    assert all(name in text for name in ["Lung Left", "Lung Right", "Lung Bounding Box"])


def test_dry_run_writes_nothing_and_manifest_is_private(tmp_path):
    paths = write_inputs(tmp_path)
    output_root = tmp_path / "not-created"
    result = prepare_itksnap_products(
        image_path=paths["image"], left_lung_path=paths["left"],
        right_lung_path=paths["right"], bbox_json_path=paths["bbox"],
        case_id="case", modality="mr", output_root=output_root, dry_run=True,
    )

    assert result["completion_status"] == "dry-run"
    assert not output_root.exists()
    assert str(tmp_path) not in json.dumps(result)
    assert result["label_mapping"] == {
        "0": "Clear Label", "1": "Lung Left", "2": "Lung Right", "10": "Lung Bounding Box"
    }


def test_writes_products_and_protects_overwrite(tmp_path):
    paths = write_inputs(tmp_path)
    output_root = tmp_path / "outputs"
    kwargs = dict(
        image_path=paths["image"], left_lung_path=paths["left"],
        right_lung_path=paths["right"], bbox_json_path=paths["bbox"],
        case_id="case", modality="mr", output_root=output_root,
    )
    prepare_itksnap_products(**kwargs)
    case_dir = output_root / "case"
    assert {path.name for path in case_dir.iterdir()} == {
        "lungs_multilabel.nii.gz", "bbox_label.nii.gz",
        "lungs_and_bbox_labels.nii.gz", "label_descriptions.txt", "launch_manifest.json",
    }
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_itksnap_products(**kwargs)
    prepare_itksnap_products(**kwargs, overwrite=True)


def test_pipeline_subcommand_parsing():
    args = pipeline.build_parser().parse_args([
        "prepare-itksnap", "--image", "image.nii.gz", "--left-lung", "left.nii.gz",
        "--right-lung", "right.nii.gz", "--bbox-json", "bbox.json",
        "--case-id", "case", "--modality", "mr", "--dry-run",
    ])
    assert args.command == "prepare-itksnap"
    assert args.left_lung == "left.nii.gz"
    assert args.dry_run
