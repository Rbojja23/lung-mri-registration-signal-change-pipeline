from pathlib import Path
import json
import sys

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import prepare_registration_inputs as cli  # noqa: E402
from registration_inputs import (  # noqa: E402
    RegistrationInputPair,
    compare_registration_geometry,
    prepare_registration_inputs,
    sha256_file,
    validate_registration_image,
)


def write_image(
    path: Path,
    *,
    size=(7, 6, 5),
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    direction=None,
    pixel_id=sitk.sitkFloat32,
    value=2.0,
) -> Path:
    image = sitk.Image(size, pixel_id)
    image += value
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def pair(fixed: Path, moving: Path, **kwargs) -> RegistrationInputPair:
    return RegistrationInputPair(
        fixed_source=fixed,
        moving_source=moving,
        case_id=kwargs.pop("case_id", "synthetic_case"),
        experiment=kwargs.pop("experiment", "full"),
        role_assignment_source=kwargs.pop("role_assignment_source", "manual-test"),
        roles_confirmed=kwargs.pop("roles_confirmed", True),
        **kwargs,
    )


def validate_pair(fixed: Path, moving: Path, tolerance=1e-5):
    return (
        validate_registration_image(fixed, role="fixed", isotropy_tolerance=tolerance),
        validate_registration_image(moving, role="moving", isotropy_tolerance=tolerance),
    )


def test_cli_requires_explicit_fixed_and_moving():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--case-id", "case"])
    args = parser.parse_args(
        ["--fixed", "fixed.nii", "--moving", "moving.nii", "--case-id", "case"]
    )
    assert args.fixed == "fixed.nii"
    assert args.moving == "moving.nii"


def test_roles_are_recorded_without_inference(tmp_path):
    fixed = write_image(tmp_path / "arbitrary_z.nii")
    moving = write_image(tmp_path / "arbitrary_a.nii")
    result = prepare_registration_inputs(
        pair(
            fixed,
            moving,
            role_assignment_source="inferred-from-repeated-bimg1",
            roles_confirmed=False,
        ),
        output_root=tmp_path / "out",
        dry_run=True,
    )
    assert result["role_assignment"]["source"] == "inferred-from-repeated-bimg1"
    assert result["role_assignment"]["confirmed"] is False
    assert "not researcher-confirmed" in result["warnings"][0]


@pytest.mark.parametrize("extension", [".nii", ".nii.gz"])
def test_supported_nifti_extensions(extension, tmp_path):
    image = write_image(tmp_path / f"image{extension}")
    result = validate_registration_image(image, role="fixed")
    assert result.extension == extension
    assert result.summary["dimensionality"] == 3
    assert result.summary["numpy_shape_zyx"] == [5, 6, 7]


def test_missing_directory_and_unsupported_inputs_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_registration_image(tmp_path / "missing.nii", role="fixed")
    directory = tmp_path / "dicom"
    directory.mkdir()
    with pytest.raises(ValueError, match="not a directory"):
        validate_registration_image(directory, role="fixed")
    unsupported = tmp_path / "image.nrrd"
    unsupported.write_text("not an image")
    with pytest.raises(ValueError, match=".nii or .nii.gz"):
        validate_registration_image(unsupported, role="fixed")


def test_2d_4d_and_vector_images_are_rejected(tmp_path):
    image_2d = sitk.Image([5, 4], sitk.sitkFloat32)
    path_2d = tmp_path / "2d.nii"
    sitk.WriteImage(image_2d, str(path_2d))
    with pytest.raises(ValueError, match="must be 3D"):
        validate_registration_image(path_2d, role="fixed")

    image_4d = sitk.Image([5, 4, 3, 2], sitk.sitkFloat32)
    path_4d = tmp_path / "4d.nii"
    sitk.WriteImage(image_4d, str(path_4d))
    with pytest.raises(ValueError, match="must be 3D"):
        validate_registration_image(path_4d, role="fixed")

    vector = sitk.GetImageFromArray(np.zeros((3, 4, 5, 2), dtype=np.float32), isVector=True)
    vector_path = tmp_path / "vector.nii"
    sitk.WriteImage(vector, str(vector_path))
    with pytest.raises(ValueError, match="scalar pixels"):
        validate_registration_image(vector_path, role="fixed")


@pytest.mark.parametrize(("invalid", "message"), [(np.nan, "NaN"), (np.inf, "infinite")])
def test_nonfinite_values_are_rejected(tmp_path, monkeypatch, invalid, message):
    array = np.ones((3, 4, 5), dtype=np.float32)
    array[1, 2, 3] = invalid
    image = sitk.GetImageFromArray(array)
    path = tmp_path / "invalid.nii.gz"
    path.touch()
    import registration_inputs

    # SimpleITK's NIfTI writer normalizes non-finite test values to zero, so
    # inject the in-memory image at the reader boundary to exercise validation.
    monkeypatch.setattr(registration_inputs.sitk, "ReadImage", lambda _: image)
    with pytest.raises(ValueError, match=message):
        validate_registration_image(path, role="moving")


def test_finite_statistics_and_physical_extent(tmp_path):
    array = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 1.0, 2.0))
    path = tmp_path / "finite.nii.gz"
    sitk.WriteImage(image, str(path))
    result = validate_registration_image(path, role="fixed")
    assert result.summary["finite_min"] == 0
    assert result.summary["finite_max"] == 23
    assert result.summary["has_nan"] is False
    assert result.summary["physical_voxel_cell_extent_xyz_mm"] == [2.0, 3.0, 4.0]


def test_same_grid_classification_and_gzip_requirement(tmp_path):
    fixed = write_image(tmp_path / "fixed.nii")
    moving = write_image(tmp_path / "moving.nii.gz")
    fixed_v, moving_v = validate_pair(fixed, moving)
    result = compare_registration_geometry(fixed_v, moving_v)
    assert result["classification"] == "same-grid"
    assert result["voxel_grid_matches"] is True
    assert result["resampling_required"] is False
    assert result["gzip_preparation_required"] is True


@pytest.mark.parametrize(
    ("change", "expected_field"),
    [
        ({"spacing": (1.1, 1.1, 1.1)}, "spacing_matches"),
        ({"origin": (2.0, 0.0, 0.0)}, "origin_matches"),
        ({"direction": (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)}, "direction_matches"),
    ],
)
def test_same_dimensions_different_geometry(tmp_path, change, expected_field):
    fixed = write_image(tmp_path / "fixed.nii.gz")
    moving = write_image(tmp_path / "moving.nii.gz", **change)
    result = compare_registration_geometry(*validate_pair(fixed, moving))
    assert result["classification"] == "same-dimensions-different-geometry"
    assert result["dimensions_match"] is True
    assert result[expected_field] is False
    assert result["resampling_required"] is True


def test_different_dimensions_classification(tmp_path):
    fixed = write_image(tmp_path / "fixed.nii.gz")
    moving = write_image(tmp_path / "moving.nii.gz", size=(8, 6, 5))
    result = compare_registration_geometry(*validate_pair(fixed, moving))
    assert result["classification"] == "different-dimensions"
    assert result["dimensions_match"] is False


def test_isotropy_tolerance_and_anisotropic_classification(tmp_path):
    fixed = write_image(tmp_path / "fixed.nii.gz", spacing=(1.0, 1.0, 1.000001))
    moving = write_image(tmp_path / "moving.nii.gz", spacing=(1.0, 1.0, 1.000001))
    isotropic = validate_pair(fixed, moving, tolerance=1e-5)
    assert isotropic[0].summary["isotropic_spacing"] is True
    assert compare_registration_geometry(*isotropic)["classification"] == "same-grid"

    anisotropic = validate_pair(fixed, moving, tolerance=1e-8)
    result = compare_registration_geometry(*anisotropic)
    assert result["classification"] == "anisotropic"
    assert result["anisotropy_detected"] is True
    assert result["resampling_required"] is True


def test_incompatible_real_mode_fails_but_dry_run_reports(tmp_path):
    fixed = write_image(tmp_path / "fixed.nii", spacing=(1.0, 1.0, 2.0))
    moving = write_image(tmp_path / "moving.nii", spacing=(1.0, 1.0, 2.0))
    output = tmp_path / "out"
    report = prepare_registration_inputs(pair(fixed, moving), output_root=output, dry_run=True)
    assert report["geometry_comparison"]["classification"] == "anisotropic"
    assert not output.exists()
    with pytest.raises(ValueError, match="resampling is not implemented"):
        prepare_registration_inputs(pair(fixed, moving), output_root=output)
    assert not output.exists()


def test_dry_run_reports_plans_without_writing(tmp_path):
    fixed = write_image(tmp_path / "fixed.nii")
    moving = write_image(tmp_path / "moving.nii")
    output = tmp_path / "outputs with spaces"
    result = prepare_registration_inputs(pair(fixed, moving), output_root=output, dry_run=True)
    assert result["completion_status"] == "dry-run"
    assert result["geometry_verification"] == {"performed": False, "passed": None}
    assert result["geometry_comparison"]["gzip_preparation_required"] is True
    assert not output.exists()


def test_real_preparation_preserves_geometry_type_range_and_sources(tmp_path):
    fixed = write_image(tmp_path / "inputs with spaces/fixed.nii", pixel_id=sitk.sitkInt16, value=-4)
    moving = write_image(tmp_path / "inputs with spaces/moving.nii", pixel_id=sitk.sitkUInt16, value=9)
    source_hashes = {"fixed": sha256_file(fixed), "moving": sha256_file(moving)}
    output_root = tmp_path / "outputs with spaces"
    result = prepare_registration_inputs(pair(fixed, moving), output_root=output_root)
    directory = output_root / "synthetic_case/full/preprocessed"
    fixed_output = sitk.ReadImage(str(directory / "fixed.nii.gz"))
    moving_output = sitk.ReadImage(str(directory / "moving.nii.gz"))

    assert result["completion_status"] == "complete"
    assert result["geometry_verification"]["passed"] is True
    assert fixed_output.GetPixelID() == sitk.sitkInt16
    assert moving_output.GetPixelID() == sitk.sitkUInt16
    assert fixed_output.GetSize() == moving_output.GetSize()
    assert result["fixed_geometry"]["finite_min"] == -4
    assert result["moving_geometry"]["finite_max"] == 9
    assert sha256_file(fixed) == source_hashes["fixed"]
    assert sha256_file(moving) == source_hashes["moving"]
    assert result["checksums_sha256"]["source_unchanged"] is True
    assert (directory / "preprocessing_manifest.json").is_file()


def test_overwrite_protection_and_scoped_overwrite(tmp_path):
    fixed = write_image(tmp_path / "fixed.nii")
    moving = write_image(tmp_path / "moving.nii")
    output = tmp_path / "out"
    kwargs = {"output_root": output}
    prepare_registration_inputs(pair(fixed, moving), **kwargs)
    experiment = output / "synthetic_case/full"
    unrelated = experiment / "keep-me.txt"
    unrelated.write_text("preserve")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_registration_inputs(pair(fixed, moving), **kwargs)
    prepare_registration_inputs(pair(fixed, moving), **kwargs, overwrite=True)
    assert unrelated.read_text() == "preserve"


def test_safe_manifest_contains_no_arbitrary_metadata_or_absolute_paths(tmp_path):
    fixed = write_image(tmp_path / "fixed.nii")
    moving = write_image(tmp_path / "moving.nii")
    result = prepare_registration_inputs(
        pair(fixed, moving), output_root=tmp_path / "out", dry_run=True
    )
    encoded = json.dumps(result)
    assert str(tmp_path) not in encoded
    assert "patient" not in encoded.lower()
    assert "environment" not in encoded.lower()
    assert result["intended_use"] == "Research-only and non-diagnostic."
    assert "pixel_id" not in result["fixed_geometry"]


def test_output_under_project_raw_is_rejected(tmp_path, monkeypatch):
    fixed = write_image(tmp_path / "fixed.nii")
    moving = write_image(tmp_path / "moving.nii")
    import registration_inputs

    raw = tmp_path / "data/raw"
    monkeypatch.setattr(registration_inputs, "DEFAULT_RAW_DATA_DIR", raw)
    with pytest.raises(ValueError, match="data/raw"):
        prepare_registration_inputs(pair(fixed, moving), output_root=raw / "registration", dry_run=True)


def test_pair_validation_rejects_same_file_and_unsafe_components(tmp_path):
    image = write_image(tmp_path / "image.nii")
    with pytest.raises(ValueError, match="different files"):
        prepare_registration_inputs(pair(image, image), output_root=tmp_path / "out", dry_run=True)
    moving = write_image(tmp_path / "moving.nii")
    with pytest.raises(ValueError, match="Case id"):
        prepare_registration_inputs(
            pair(image, moving, case_id="../escape"), output_root=tmp_path / "out", dry_run=True
        )


def test_module_has_no_deedsbcv_execution_dependency():
    import registration_inputs

    source = Path(registration_inputs.__file__).read_text()
    assert "subprocess" not in source
    assert "linearBCV" not in source
    assert "deedsBCV" in source  # limitation text only, never an executable invocation
