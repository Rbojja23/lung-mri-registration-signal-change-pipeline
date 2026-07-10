from pathlib import Path
import sys

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from bounding_box import BBoxZYX  # noqa: E402
from dicom_reference import inspect_reference_series, map_image_slices_to_dicom  # noqa: E402
from osirix_export import (  # noqa: E402
    bbox_to_contours_by_slice,
    binary_slice_to_index_loops,
    export_osirix_package,
    mask_to_contours_by_slice,
    validate_dicom_image_alignment,
    validate_rtstruct,
    write_rtstruct,
)


def make_image(shape_zyx=(3, 4, 5)) -> sitk.Image:
    array = np.zeros(shape_zyx, dtype=np.int16)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 2.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def make_mask(reference: sitk.Image) -> sitk.Image:
    arr = np.zeros((3, 4, 5), dtype=np.uint8)
    arr[:, 1:3, 1:4] = 1
    image = sitk.GetImageFromArray(arr)
    image.CopyInformation(reference)
    return image


def write_image(path: Path, image: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def make_dicom_series(
    output_dir: Path,
    *,
    frame_uid: str | None = "1.2.826.0.1.3680043.10.100.3",
    series_uid: str = "1.2.826.0.1.3680043.10.100.2",
    filename_prefix: str = "slice",
    duplicate_sop: bool = False,
    rows: int = 4,
    cols: int = 5,
    sop_class=None,
) -> list[Path]:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    sop_class = sop_class or MRImageStorage
    output_dir.mkdir(parents=True, exist_ok=True)
    study_uid = "1.2.826.0.1.3680043.10.100.1"
    sop_uid = generate_uid()
    paths = []
    for z in range(3):
        path = output_dir / f"{filename_prefix}_{z:03d}.dcm"
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = sop_class
        file_meta.MediaStorageSOPInstanceUID = sop_uid if duplicate_sop else generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        if frame_uid:
            ds.FrameOfReferenceUID = frame_uid
        ds.Modality = "MR"
        ds.Rows = rows
        ds.Columns = cols
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.0
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, float(z * 2)]
        ds.InstanceNumber = z + 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = np.zeros((rows, cols), dtype=np.int16).tobytes()
        ds.save_as(str(path), write_like_original=False)
        paths.append(path)
    return paths


def test_reference_series_validation(tmp_path):
    make_dicom_series(tmp_path)

    series = inspect_reference_series(tmp_path)

    assert series.report.rtstruct_compatible
    assert series.report.classification == "RTSTRUCT-compatible"
    assert series.report.image_count == 3


def test_rejection_of_mixed_series(tmp_path):
    make_dicom_series(tmp_path, series_uid="1.2.3.1", filename_prefix="a")
    make_dicom_series(tmp_path, series_uid="1.2.3.2", filename_prefix="b")

    series = inspect_reference_series(tmp_path)

    assert not series.report.rtstruct_compatible
    assert any("multiple Series" in item for item in series.report.failures)


def test_rejection_of_duplicate_sop_uids(tmp_path):
    make_dicom_series(tmp_path, duplicate_sop=True)

    series = inspect_reference_series(tmp_path)

    assert not series.report.rtstruct_compatible
    assert any("not unique" in item for item in series.report.failures)


def test_rejection_of_inconsistent_dimensions(tmp_path):
    make_dicom_series(tmp_path, rows=4, filename_prefix="a")
    make_dicom_series(tmp_path, rows=5, filename_prefix="b")

    series = inspect_reference_series(tmp_path)

    assert not series.report.rtstruct_compatible
    assert any("Rows/Columns" in item for item in series.report.failures)


def test_missing_frame_of_reference_handling(tmp_path):
    make_dicom_series(tmp_path, frame_uid=None)

    series = inspect_reference_series(tmp_path)

    assert not series.report.rtstruct_compatible
    assert series.report.image_viewing_only


def test_secondary_capture_capability_classification(tmp_path):
    from pydicom.uid import SecondaryCaptureImageStorage

    make_dicom_series(tmp_path, sop_class=SecondaryCaptureImageStorage)

    series = inspect_reference_series(tmp_path)

    assert series.report.source_sop_class_category == "secondary-capture"
    assert series.report.rtstruct_compatible
    assert series.report.warnings


def test_image_to_dicom_slice_mapping(tmp_path):
    make_dicom_series(tmp_path)
    series = inspect_reference_series(tmp_path)
    image = make_image()

    mapping = map_image_slices_to_dicom(image, series)

    assert len(mapping) == 3


def test_geometry_mismatch_rejection(tmp_path):
    make_dicom_series(tmp_path)
    series = inspect_reference_series(tmp_path)
    image = make_image()
    image.SetSpacing((9.0, 1.0, 2.0))

    with pytest.raises(ValueError, match="geometry mismatch"):
        validate_dicom_image_alignment(image, series)


def test_contour_extraction_and_multiple_disconnected_contours():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[3, 3] = True

    loops = binary_slice_to_index_loops(mask)

    assert len(loops) == 2
    assert all(len(set(loop[:-1])) >= 4 for loop in loops)


def test_mask_physical_coordinate_conversion(tmp_path):
    make_dicom_series(tmp_path)
    series = inspect_reference_series(tmp_path)
    image = make_image()
    mask = make_mask(image)
    mapping = validate_dicom_image_alignment(image, series)

    contours = mask_to_contours_by_slice(mask, mapping)

    first = next(iter(contours.values()))[0]
    assert len(first) >= 12
    assert np.isfinite(np.array(first)).all()


def test_bbox_rectangle_generation():
    image = make_image()
    bbox = BBoxZYX(z_min=0, z_max=2, y_min=1, y_max=3, x_min=1, x_max=4)

    contours = bbox_to_contours_by_slice(image, bbox)

    assert set(contours) == {0, 1}
    assert len(contours[0][0]) == 15


def test_generated_rtstruct_reopening_and_references(tmp_path):
    make_dicom_series(tmp_path / "dicom")
    series = inspect_reference_series(tmp_path / "dicom")
    image = make_image()
    mask = make_mask(image)
    mapping = validate_dicom_image_alignment(image, series)
    roi_contours = {
        "Whole Lung Segmentation": mask_to_contours_by_slice(mask, mapping),
        "Lung Bounding Box": bbox_to_contours_by_slice(
            image, BBoxZYX(0, 3, 1, 3, 1, 4)
        ),
    }
    out = tmp_path / "rtstruct.dcm"

    write_rtstruct(series=series, roi_contours=roi_contours, slice_mapping=mapping, output_path=out)
    validation = validate_rtstruct(out, {s.sop_instance_uid for s in series.slices})

    assert out.exists()
    assert validation["referenced_sop_count"] == 3


def test_dry_run_writes_nothing(tmp_path):
    dicom = tmp_path / "dicom"
    make_dicom_series(dicom)
    image = make_image()
    mask = make_mask(image)
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    write_image(image_path, image)
    write_image(mask_path, mask)
    output_root = tmp_path / "out"

    report = export_osirix_package(
        dicom_dir=dicom,
        image_path=image_path,
        mask_path=mask_path,
        case_id="case",
        bbox_json=None,
        output_root=output_root,
        dry_run=True,
    )

    assert report["completion_status"] == "dry-run"
    assert not output_root.exists()


def test_overwrite_protection(tmp_path):
    dicom = tmp_path / "dicom"
    make_dicom_series(dicom)
    image = make_image()
    mask = make_mask(image)
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    write_image(image_path, image)
    write_image(mask_path, mask)
    output_root = tmp_path / "out"
    (output_root / "case").mkdir(parents=True)
    (output_root / "case" / "existing.txt").write_text("x")

    with pytest.raises(FileExistsError):
        export_osirix_package(
            dicom_dir=dicom,
            image_path=image_path,
            mask_path=mask_path,
            case_id="case",
            bbox_json=None,
            output_root=output_root,
        )


def test_safe_export_report_fields(tmp_path):
    dicom = tmp_path / "dicom"
    make_dicom_series(dicom)
    image = make_image()
    mask = make_mask(image)
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    write_image(image_path, image)
    write_image(mask_path, mask)

    report = export_osirix_package(
        dicom_dir=dicom,
        image_path=image_path,
        mask_path=mask_path,
        case_id="case",
        bbox_json=None,
        output_root=tmp_path / "out",
        dry_run=True,
    )

    text = " ".join(report.keys()).lower()
    assert "patient" not in text
    assert "uid" not in jsonish(report).lower()


def test_fallback_package_creation_when_rtstruct_not_possible(tmp_path):
    dicom = tmp_path / "dicom"
    make_dicom_series(dicom, frame_uid=None)
    image = make_image()
    mask = make_mask(image)
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    bbox_path = tmp_path / "bbox.json"
    bbox_path.write_text('{"voxel_bbox_half_open_zyx":{"z_min":0,"z_max":1,"y_min":0,"y_max":1,"x_min":0,"x_max":1}}')
    write_image(image_path, image)
    write_image(mask_path, mask)

    report = export_osirix_package(
        dicom_dir=dicom,
        image_path=image_path,
        mask_path=mask_path,
        case_id="case",
        bbox_json=bbox_path,
        output_root=tmp_path / "out",
    )

    assert report["selected_export_format"] == "fallback-package"
    assert (tmp_path / "out" / "case" / "review_package" / "README.txt").exists()


def jsonish(value) -> str:
    import json

    return json.dumps(value)
