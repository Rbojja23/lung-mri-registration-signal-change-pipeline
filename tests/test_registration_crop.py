from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import prepare_shared_registration_crop as cli  # noqa: E402
import registration_crop  # noqa: E402
from deedsbcv_runner import validate_prepared_inputs  # noqa: E402
from registration_crop import (  # noqa: E402
    SharedCropRequest,
    prepare_shared_registration_crop,
)
from registration_inputs import sha256_file  # noqa: E402


def write_image(
    path: Path,
    *,
    size=(8, 7, 6),
    spacing=(1.0, 1.0, 1.0),
    origin=(10.0, 20.0, 30.0),
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    pixel_id=sitk.sitkFloat32,
    offset=0,
) -> Path:
    array = np.arange(np.prod(size), dtype=np.float32).reshape(size[::-1]) + offset
    image = sitk.GetImageFromArray(array)
    image = sitk.Cast(image, pixel_id)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def write_bbox(
    path: Path,
    *,
    size=(8, 7, 6),
    bounds=(1, 6, 2, 7, 0, 5),
    image_path="[external path]",
    convention="half-open [min:max), matching Python slicing",
) -> Path:
    z_min, z_max, y_min, y_max, x_min, x_max = bounds
    cropped_size = [x_max - x_min, y_max - y_min, z_max - z_min]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "original_image_size_xyz": list(size),
                "cropped_image_size_xyz": cropped_size,
                "voxel_bbox_half_open_zyx": {
                    "z_min": z_min,
                    "z_max": z_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "x_min": x_min,
                    "x_max": x_max,
                },
                "coordinate_conventions": {"bbox_intervals": convention},
            }
        )
    )
    return path


def make_request(tmp_path: Path, **kwargs):
    spacing = kwargs.pop("spacing", (1.0, 1.0, 1.0))
    direction = kwargs.pop(
        "direction", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )
    fixed = write_image(
        tmp_path / "inputs/fixed.nii.gz",
        spacing=spacing,
        direction=direction,
        pixel_id=kwargs.pop("fixed_type", sitk.sitkInt16),
    )
    moving = write_image(
        tmp_path / "inputs/moving.nii.gz",
        spacing=spacing,
        direction=direction,
        pixel_id=kwargs.pop("moving_type", sitk.sitkUInt16),
        offset=100,
    )
    bbox = write_bbox(tmp_path / "bbox/bbox.json", **kwargs)
    return SharedCropRequest(
        fixed,
        moving,
        bbox,
        "synthetic_case",
        role_assignment_source="inferred-from-repeated-bimg1",
        roles_confirmed=False,
    )


def prepare(tmp_path: Path, request=None, **kwargs):
    request = request or make_request(tmp_path)
    return prepare_shared_registration_crop(
        request, output_root=tmp_path / "registration", **kwargs
    )


def test_exact_shared_bbox_axis_conversion_geometry_and_half_open_max(tmp_path):
    request = make_request(tmp_path)
    result = prepare(tmp_path, request)
    directory = tmp_path / "registration/synthetic_case/cropped/preprocessed"
    fixed = sitk.ReadImage(str(directory / "fixed.nii.gz"))
    moving = sitk.ReadImage(str(directory / "moving.nii.gz"))
    source = sitk.ReadImage(str(request.fixed))

    assert result["simpleitk_start_index_xyz"] == [0, 2, 1]
    assert result["simpleitk_size_xyz"] == [5, 5, 5]
    assert fixed.GetSize() == (5, 5, 5)
    assert fixed.GetOrigin() == pytest.approx(source.TransformIndexToPhysicalPoint((0, 2, 1)))
    assert fixed.GetSpacing() == source.GetSpacing()
    assert fixed.GetDirection() == source.GetDirection()
    assert fixed.GetOrigin() == moving.GetOrigin()
    assert fixed.GetSize() == moving.GetSize()
    assert sitk.GetArrayFromImage(fixed)[-1, -1, -1] == sitk.GetArrayFromImage(source)[5, 6, 4]


def test_nonidentity_direction_origin_and_pixel_types_are_preserved(tmp_path):
    direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    request = make_request(tmp_path, direction=direction)
    prepare(tmp_path, request)
    directory = tmp_path / "registration/synthetic_case/cropped/preprocessed"
    fixed = sitk.ReadImage(str(directory / "fixed.nii.gz"))
    moving = sitk.ReadImage(str(directory / "moving.nii.gz"))
    source = sitk.ReadImage(str(request.fixed))
    assert fixed.GetDirection() == direction
    assert fixed.GetOrigin() == pytest.approx(source.TransformIndexToPhysicalPoint((0, 2, 1)))
    assert fixed.GetPixelID() == sitk.sitkInt16
    assert moving.GetPixelID() == sitk.sitkUInt16


@pytest.mark.parametrize(
    ("bounds", "touches"),
    [
        ((0, 5, 1, 6, 1, 6), "z_min"),
        ((1, 6, 0, 6, 1, 6), "y_min"),
        ((1, 6, 1, 6, 0, 6), "x_min"),
        ((1, 6, 1, 6, 1, 8), "x_max"),
        ((1, 6, 1, 7, 1, 6), "y_max"),
        ((1, 6, 1, 6, 1, 6), "z_max"),
    ],
)
def test_boundary_touch_reporting(tmp_path, bounds, touches):
    result = prepare(tmp_path, make_request(tmp_path, bounds=bounds), dry_run=True)
    assert result["boundary_touch"][touches] is True
    assert "padding may be clipped" in " ".join(result["warnings"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bounds": (4, 4, 1, 6, 1, 6)}, "Invalid bbox z bounds"),
        ({"size": (9, 7, 6)}, "original image size"),
        ({"bounds": (1, 7, 1, 6, 1, 6)}, "Invalid bbox z bounds"),
        ({"convention": "closed [min,max]"}, "half-open"),
    ],
)
def test_invalid_bbox_metadata_is_rejected(tmp_path, kwargs, message):
    request = make_request(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        prepare(tmp_path, request, dry_run=True)


def test_bbox_fixed_image_path_mismatch_is_rejected(tmp_path):
    other = tmp_path / "other.nii.gz"
    request = make_request(tmp_path, image_path=other)
    with pytest.raises(ValueError, match="image path"):
        prepare(tmp_path, request, dry_run=True)


def test_original_grid_mismatch_is_rejected(tmp_path):
    request = make_request(tmp_path)
    moving = sitk.ReadImage(str(request.moving))
    moving.SetOrigin((99.0, 20.0, 30.0))
    sitk.WriteImage(moving, str(request.moving))
    with pytest.raises(ValueError, match="exact original voxel grid"):
        prepare(tmp_path, request, dry_run=True)


def test_anisotropic_pair_reports_in_dry_run_and_rejects_real_preparation(tmp_path):
    request = make_request(tmp_path, spacing=(1.0, 1.0, 2.0))
    result = prepare(tmp_path, request, dry_run=True)
    assert result["isotropic_spacing"] is False
    assert result["resampling_required"] is True
    with pytest.raises(ValueError, match="anisotropic"):
        prepare(tmp_path, request)


def test_nonfinite_source_is_rejected(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    original = registration_crop.sitk.ReadImage
    invalid = original(str(request.fixed))
    array = sitk.GetArrayFromImage(invalid).astype(np.float32)
    array[0, 0, 0] = np.nan
    invalid = sitk.GetImageFromArray(array)
    invalid.CopyInformation(original(str(request.fixed)))

    def fake_read(path):
        return invalid if Path(path) == request.fixed.resolve() else original(str(path))

    monkeypatch.setattr(registration_crop.sitk, "ReadImage", fake_read)
    with pytest.raises(ValueError, match="NaN"):
        prepare(tmp_path, request, dry_run=True)


def test_runner_compatible_manifest_roles_and_checksums(tmp_path):
    request = make_request(tmp_path)
    fixed_hash = sha256_file(request.fixed)
    moving_hash = sha256_file(request.moving)
    prepare(tmp_path, request)
    directory = tmp_path / "registration/synthetic_case/cropped/preprocessed"
    validated = validate_prepared_inputs(
        directory, case_id="synthetic_case", experiment="cropped"
    )
    manifest = validated["manifest"]
    assert manifest["geometry_comparison"]["classification"] == "same-grid"
    assert manifest["geometry_comparison"]["resampling_required"] is False
    assert manifest["role_assignment"]["source"] == "inferred-from-repeated-bimg1"
    assert manifest["role_assignment"]["confirmed"] is False
    assert sha256_file(request.fixed) == fixed_hash
    assert sha256_file(request.moving) == moving_hash


def test_dry_run_writes_nothing_and_cli_reports_plan(tmp_path, capsys):
    request = make_request(tmp_path)
    output = tmp_path / "planned output"
    code = cli.main(
        [
            "--fixed", str(request.fixed),
            "--moving", str(request.moving),
            "--bbox-json", str(request.bbox_json),
            "--case-id", request.case_id,
            "--output-root", str(output),
            "--dry-run",
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["writes_performed"] is False
    assert report["registration_executed"] is False
    assert report["start_index_xyz"] == [0, 2, 1]
    assert not output.exists()


def test_overwrite_protection_is_scoped_and_manifests_are_privacy_safe(tmp_path):
    request = make_request(tmp_path)
    result = prepare(tmp_path, request)
    directory = tmp_path / "registration/synthetic_case/cropped"
    full_product = tmp_path / "registration/synthetic_case/full/preprocessed/fixed.nii.gz"
    full_product.parent.mkdir(parents=True)
    full_product.write_text("preserve full experiment")
    unrelated = directory / "keep-me.txt"
    unrelated.write_text("preserve")
    with pytest.raises(FileExistsError):
        prepare(tmp_path, request)
    prepare(tmp_path, request, overwrite=True)
    assert unrelated.read_text() == "preserve"
    assert full_product.read_text() == "preserve full experiment"
    shared = json.loads((directory / "preprocessed/shared_crop_manifest.json").read_text())
    preprocessing = json.loads((directory / "preprocessed/preprocessing_manifest.json").read_text())
    encoded = json.dumps([result, shared, preprocessing])
    assert str(tmp_path) not in encoded
    assert "patient" not in encoded.lower()
    assert shared["registration_executed"] is False


def test_preparation_executes_no_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("shared cropping must not execute subprocesses"),
    )
    result = prepare(tmp_path, make_request(tmp_path), dry_run=True)
    assert result["registration_executed"] is False
