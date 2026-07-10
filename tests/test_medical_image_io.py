from pathlib import Path
import sys

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import medical_image_io  # noqa: E402
from medical_image_io import (  # noqa: E402
    DicomFileInfo,
    compare_image_geometry,
    geometry_matches,
    load_dicom_directory,
    load_medical_image,
    load_nifti,
    numpy_index_zyx_to_sitk_index_xyz,
    save_summary_json,
    sitk_index_xyz_to_numpy_index_zyx,
    sort_dicom_files_for_loading,
    voxel_index_xyz_to_physical_point,
    write_nifti,
)


def make_test_image() -> sitk.Image:
    array = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.25, 1.5, 2.0))
    image.SetOrigin((10.0, 20.0, -30.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0))
    return image


def write_test_nifti(path: Path) -> sitk.Image:
    image = make_test_image()
    sitk.WriteImage(image, str(path))
    return image


def make_dicom_series(
    output_dir: Path,
    *,
    series_uid: str = "1.2.826.0.1.3680043.10.543.1",
    modality: str = "CT",
    filename_prefix: str = "slice",
    positions: list[float] | None = None,
    include_positions: bool = True,
    include_instance_numbers: bool = True,
) -> list[Path]:
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    output_dir.mkdir(parents=True, exist_ok=True)
    positions = positions or [0.0, 2.0, 4.0]
    files = []
    for index, z_position in enumerate(positions, start=1):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        path = output_dir / f"{filename_prefix}_{index:03d}.dcm"
        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = "1.2.826.0.1.3680043.10.543.99"
        dataset.SeriesInstanceUID = series_uid
        dataset.Modality = modality
        dataset.Rows = 4
        dataset.Columns = 5
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SliceThickness = 2.0
        if include_positions:
            dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            dataset.ImagePositionPatient = [0, 0, z_position]
        if include_instance_numbers:
            dataset.InstanceNumber = index
        pixels = np.full((4, 5), index, dtype=np.int16)
        dataset.PixelData = pixels.tobytes()
        dataset.save_as(str(path), write_like_original=False)
        files.append(path)
    return files


def test_load_synthetic_nii_file(tmp_path):
    path = tmp_path / "image.nii"
    original = write_test_nifti(path)

    loaded = load_nifti(path, modality="mr")

    assert loaded.modality == "MR"
    assert loaded.input_type == "nifti"
    assert loaded.size_xyz == original.GetSize()
    assert loaded.array_shape_zyx == sitk.GetArrayFromImage(original).shape


def test_load_synthetic_nii_gz_file(tmp_path):
    path = tmp_path / "image.nii.gz"
    original = write_test_nifti(path)

    loaded = load_medical_image(path, input_type="auto", modality="ct")

    assert loaded.modality == "CT"
    assert loaded.input_type == "nifti"
    assert loaded.spacing_xyz == original.GetSpacing()


def test_nifti_requires_modality(tmp_path):
    path = tmp_path / "image.nii"
    write_test_nifti(path)

    with pytest.raises(ValueError, match="Unsupported"):
        load_nifti(path, modality="auto")


def test_nifti_preserves_geometry(tmp_path):
    path = tmp_path / "image.nii.gz"
    original = write_test_nifti(path)

    loaded = load_nifti(path, modality="CT")

    assert loaded.spacing_xyz == original.GetSpacing()
    assert loaded.origin_xyz == original.GetOrigin()
    assert loaded.direction == original.GetDirection()


def test_index_conversions():
    assert sitk_index_xyz_to_numpy_index_zyx((1, 2, 3)) == (3, 2, 1)
    assert numpy_index_zyx_to_sitk_index_xyz((3, 2, 1)) == (1, 2, 3)


def test_voxel_index_to_physical_coordinate():
    image = make_test_image()

    point = voxel_index_xyz_to_physical_point(image, (2, 1, 1))

    assert point == pytest.approx((12.5, 22.0, -31.5))


def test_geometry_comparison_success_and_failure():
    image_a = make_test_image()
    image_b = sitk.Image(image_a)

    assert geometry_matches(image_a, image_b)
    assert compare_image_geometry(image_a, image_b) == []

    image_b.SetSpacing((9.0, 1.5, 2.0))

    mismatches = compare_image_geometry(image_a, image_b)
    assert any("spacing mismatch" in item for item in mismatches)
    assert not geometry_matches(image_a, image_b)


def test_dicom_primary_series_loading(tmp_path):
    dicom_dir = tmp_path / "dicom"
    make_dicom_series(dicom_dir)

    loaded = load_dicom_directory(dicom_dir, modality="auto")

    assert loaded.modality == "CT"
    assert loaded.input_type == "dicom"
    assert loaded.dicom_series_id is not None
    assert loaded.size_xyz == (5, 4, 3)


def test_fallback_loading_when_gdcm_discovers_no_series(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom"
    make_dicom_series(dicom_dir)
    monkeypatch.setattr(
        medical_image_io.sitk.ImageSeriesReader,
        "GetGDCMSeriesIDs",
        staticmethod(lambda _path: ()),
    )

    loaded = load_dicom_directory(dicom_dir, modality="auto")

    assert loaded.modality == "CT"
    assert loaded.size_xyz == (5, 4, 3)
    assert any("metadata fallback" in item for item in loaded.loader_warnings)


def test_fallback_sorting_by_image_position_patient(tmp_path):
    infos = [
        DicomFileInfo(
            tmp_path / "b.dcm",
            "1",
            "MR",
            4,
            5,
            (1, 0, 0, 0, 1, 0),
            (0, 0, 2),
            2,
        ),
        DicomFileInfo(
            tmp_path / "a.dcm",
            "1",
            "MR",
            4,
            5,
            (1, 0, 0, 0, 1, 0),
            (0, 0, 1),
            1,
        ),
    ]

    ordered, warning = sort_dicom_files_for_loading(infos)

    assert [info.path.name for info in ordered] == ["a.dcm", "b.dcm"]
    assert warning is None


def test_fallback_sorting_by_instance_number(tmp_path):
    infos = [
        DicomFileInfo(tmp_path / "b.dcm", "1", "MR", 4, 5, None, None, 2),
        DicomFileInfo(tmp_path / "a.dcm", "1", "MR", 4, 5, None, None, 1),
    ]

    ordered, warning = sort_dicom_files_for_loading(infos)

    assert [info.path.name for info in ordered] == ["a.dcm", "b.dcm"]
    assert warning is None


def test_fallback_sorting_by_filename_warns(tmp_path):
    infos = [
        DicomFileInfo(tmp_path / "z.dcm", "1", "MR", 4, 5, None, None, None),
        DicomFileInfo(tmp_path / "a.dcm", "1", "MR", 4, 5, None, None, None),
    ]

    ordered, warning = sort_dicom_files_for_loading(infos)

    assert [info.path.name for info in ordered] == ["a.dcm", "z.dcm"]
    assert "filename order" in warning


def test_rejects_mixed_incompatible_dicom_series(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom"
    make_dicom_series(dicom_dir, series_uid="1.2.3.1", filename_prefix="a")
    make_dicom_series(dicom_dir, series_uid="1.2.3.2", filename_prefix="b")
    monkeypatch.setattr(
        medical_image_io.sitk.ImageSeriesReader,
        "GetGDCMSeriesIDs",
        staticmethod(lambda _path: ()),
    )

    with pytest.raises(ValueError, match="Multiple incompatible DICOM series"):
        load_dicom_directory(dicom_dir, modality="auto")


def test_rejects_unsafe_writes_under_data_raw():
    loaded = load_nifti_from_image_for_write_tests()
    unsafe_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "test.nii.gz"

    with pytest.raises(ValueError, match="data/raw"):
        write_nifti(loaded, unsafe_path)


def test_no_output_directory_creation_during_read_only(tmp_path):
    path = tmp_path / "image.nii"
    write_test_nifti(path)
    output_dir = tmp_path / "outputs" / "nifti"

    load_nifti(path, modality="ct")

    assert not output_dir.exists()


def test_output_directory_creation_only_for_requested_write(tmp_path):
    loaded = load_nifti_from_image_for_write_tests(tmp_path)
    output_path = tmp_path / "outputs" / "nifti" / "image.nii.gz"

    write_nifti(loaded, output_path)

    assert output_path.exists()


def test_overwrite_protection(tmp_path):
    loaded = load_nifti_from_image_for_write_tests(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}\n")

    with pytest.raises(FileExistsError):
        save_summary_json(loaded, summary_path)

    save_summary_json(loaded, summary_path, overwrite=True)
    assert "modality" in summary_path.read_text()


def load_nifti_from_image_for_write_tests(tmp_path: Path | None = None):
    tmp_path = tmp_path or Path.cwd()
    image = make_test_image()
    return medical_image_io.LoadedMedicalImage(
        image=image,
        modality="CT",
        input_type="nifti",
        source_path=tmp_path / "synthetic.nii.gz",
        size_xyz=image.GetSize(),
        array_shape_zyx=sitk.GetArrayFromImage(image).shape,
        spacing_xyz=image.GetSpacing(),
        origin_xyz=image.GetOrigin(),
        direction=image.GetDirection(),
    )
