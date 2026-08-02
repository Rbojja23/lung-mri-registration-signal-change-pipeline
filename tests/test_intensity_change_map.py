from __future__ import annotations

import json
import nibabel as nib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import intensity_change_map as module  # noqa: E402
from intensity_change_map import TERMINOLOGY, generate_intensity_change_map  # noqa: E402


DIRECTION = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def image(array, *, spacing=(1.2, 1.5, 2.0), origin=(3.0, 4.0, 5.0), direction=DIRECTION):
    result = sitk.GetImageFromArray(np.asarray(array))
    result.SetSpacing(spacing)
    result.SetOrigin(origin)
    result.SetDirection(direction)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(value, str(path))
    return path


def inputs(tmp_path, fixed=None, registered=None, mask=None):
    shape = (5, 6, 7)
    fixed = np.full(shape, 10.0, np.float32) if fixed is None else np.asarray(fixed)
    registered = fixed.copy() if registered is None else np.asarray(registered)
    if mask is None:
        mask = np.zeros(shape, np.uint8)
        mask[1:4, 1:5, 1:6] = 1
    return (
        write(tmp_path / "fixed.nii.gz", image(fixed)),
        write(tmp_path / "registered.nii.gz", image(registered)),
        write(tmp_path / "mask.nii.gz", image(mask)),
    )


def run(tmp_path, *, fixed=None, registered=None, mask=None, dry_run=False, **kwargs):
    paths = inputs(tmp_path, fixed, registered, mask)
    result = generate_intensity_change_map(
        fixed_path=paths[0], registered_path=paths[1], lung_mask_path=paths[2],
        case_id="synthetic", output_root=tmp_path / "outputs", skip_normalized=True,
        dry_run=dry_run, **kwargs,
    )
    return result, paths


def array(path):
    return np.transpose(np.asanyarray(nib.load(str(path)).dataobj), (2, 1, 0))


def generated(tmp_path, name):
    return tmp_path / "outputs/synthetic/full/intensity_change" / name


@pytest.mark.parametrize("delta", [0.0, 3.0, -4.0])
def test_known_signed_and_absolute_differences(tmp_path, delta):
    fixed = np.full((5, 6, 7), 10.0, np.float32)
    result, _ = run(tmp_path, fixed=fixed, registered=fixed - delta)
    signed = array(generated(tmp_path, "signed_difference.nii.gz"))
    absolute = array(generated(tmp_path, "absolute_difference.nii.gz"))
    assert np.unique(signed[np.isfinite(signed)]) == pytest.approx([delta])
    assert np.unique(absolute[np.isfinite(absolute)]) == pytest.approx([abs(delta)])
    assert np.isnan(signed[0, 0, 0])


def test_absolute_threshold_excludes_zero_and_near_zero(tmp_path):
    fixed = np.full((5, 6, 7), 10.0, np.float32)
    fixed[2, 2, 2] = 0
    fixed[2, 2, 3] = 0.01
    f, r, m = inputs(tmp_path, fixed, fixed - 1)
    result = generate_intensity_change_map(
        fixed_path=f, registered_path=r, lung_mask_path=m, case_id="synthetic",
        output_root=tmp_path / "outputs", denominator_policy="absolute",
        denominator_threshold=0.1,
    )
    normalized = array(generated(tmp_path, "normalized_difference.nii.gz"))
    assert np.isnan(normalized[2, 2, 2]) and np.isnan(normalized[2, 2, 3])
    assert result["denominator_policy"]["invalid_denominator_voxel_count"] == 2


def test_explicit_percentile_policy_is_deterministic(tmp_path):
    fixed = np.arange(210, dtype=np.float32).reshape(5, 6, 7)
    f, r, m = inputs(tmp_path, fixed, fixed - 1)
    result = generate_intensity_change_map(
        fixed_path=f, registered_path=r, lung_mask_path=m, case_id="synthetic",
        output_root=tmp_path / "outputs", denominator_policy="percentile",
        denominator_percentile=50, dry_run=True,
    )
    mask = array(m).astype(bool)
    assert result["denominator_policy"]["derived_threshold"] == pytest.approx(np.percentile(np.abs(fixed[mask]), 50))
    assert "exploratory" in " ".join(result["warnings"]).lower()


def test_normalized_requires_explicit_policy_and_threshold(tmp_path):
    f, r, m = inputs(tmp_path)
    with pytest.raises(ValueError, match="explicit denominator policy"):
        generate_intensity_change_map(fixed_path=f, registered_path=r, lung_mask_path=m, case_id="x", output_root=tmp_path)
    with pytest.raises(ValueError, match="requires a finite"):
        generate_intensity_change_map(fixed_path=f, registered_path=r, lung_mask_path=m, case_id="x", output_root=tmp_path, denominator_policy="absolute")
    with pytest.raises(ValueError, match="finite positive"):
        generate_intensity_change_map(fixed_path=f, registered_path=r, lung_mask_path=m, case_id="x", output_root=tmp_path, denominator_policy="absolute", denominator_threshold=0)


def test_skip_normalized_display_is_finite_zero_outside_and_analytical_unclipped(tmp_path):
    fixed = np.full((5, 6, 7), 100.0, np.float32)
    registered = fixed.copy()
    registered[2, 2, 2] = -900
    result, (_, _, mask_path) = run(tmp_path, fixed=fixed, registered=registered, display_percentile=90)
    signed = array(generated(tmp_path, "signed_difference.nii.gz"))
    display = array(generated(tmp_path, "signed_difference_display.nii.gz"))
    mask = array(mask_path).astype(bool)
    assert np.nanmax(signed) == 1000
    assert np.isfinite(display).all() and np.all(display[~mask] == 0)
    scale = result["display_scaling"]["signed"]
    assert scale["display_min"] == -scale["display_max"]
    assert np.max(display) <= scale["display_max"]
    assert "normalized_analytical" not in result["outputs"]


def test_geometry_and_nonidentity_direction_preserved(tmp_path):
    result, (fixed, _, _) = run(tmp_path)
    output = sitk.ReadImage(str(generated(tmp_path, "signed_difference.nii.gz")))
    reference = sitk.ReadImage(str(fixed))
    assert output.GetDirection() == DIRECTION
    assert output.GetSpacing() == reference.GetSpacing()
    assert output.GetOrigin() == reference.GetOrigin()


@pytest.mark.parametrize("target", ["registered", "mask"])
def test_geometry_mismatch_rejected(tmp_path, target):
    f, r, m = inputs(tmp_path)
    path = r if target == "registered" else m
    changed = sitk.ReadImage(str(path))
    changed.SetOrigin((9.0, 4.0, 5.0))
    sitk.WriteImage(changed, str(path))
    with pytest.raises(ValueError, match="geometry mismatch"):
        generate_intensity_change_map(fixed_path=f, registered_path=r, lung_mask_path=m, case_id="x", output_root=tmp_path, skip_normalized=True)


@pytest.mark.parametrize("kind", ["empty", "nonbinary"])
def test_invalid_masks_rejected(tmp_path, kind):
    mask = np.zeros((5, 6, 7), np.uint8)
    if kind == "nonbinary":
        mask[1, 1, 1] = 2
    f, r, m = inputs(tmp_path, mask=mask)
    with pytest.raises(ValueError, match="empty|binary"):
        generate_intensity_change_map(fixed_path=f, registered_path=r, lung_mask_path=m, case_id="x", output_root=tmp_path, skip_normalized=True)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_image_rejected(tmp_path, bad):
    fixed = np.ones((5, 6, 7), np.float32)
    fixed[0, 0, 0] = bad
    f, r, m = inputs(tmp_path, np.ones_like(fixed), np.ones_like(fixed))
    invalid = image(fixed)
    original = module.sitk.ReadImage
    def reader(path):
        return invalid if Path(path) == Path(f).resolve() else original(str(path))
    module.sitk.ReadImage = reader
    with pytest.raises(ValueError, match="NaN or infinity"):
        generate_intensity_change_map(fixed_path=f, registered_path=r, lung_mask_path=m, case_id="x", output_root=tmp_path, skip_normalized=True)
    module.sitk.ReadImage = original


def test_left_right_statistics_and_exact_union(tmp_path):
    f, r, m = inputs(tmp_path, registered=np.full((5, 6, 7), 8.0, np.float32))
    combined = array(m).astype(bool)
    left = combined.copy(); left[:, :, 4:] = False
    right = combined & ~left
    lp = write(tmp_path / "left.nii.gz", image(left.astype(np.uint8)))
    rp = write(tmp_path / "right.nii.gz", image(right.astype(np.uint8)))
    result = generate_intensity_change_map(
        fixed_path=f, registered_path=r, lung_mask_path=m, left_lung_mask_path=lp,
        right_lung_mask_path=rp, case_id="synthetic", output_root=tmp_path / "outputs",
        skip_normalized=True, dry_run=True,
    )
    assert result["statistics"]["signed"]["left_lung"]["mean"] == 2
    assert result["statistics"]["signed"]["right_lung"]["mean"] == 2


def test_dry_run_writes_nothing_and_manifest_is_privacy_safe(tmp_path):
    result, _ = run(tmp_path, dry_run=True)
    assert result["writes_performed"] is False
    assert not (tmp_path / "outputs").exists()
    serialized = json.dumps(result)
    assert str(tmp_path) not in serialized
    assert "patient" not in serialized.lower()
    assert result["terminology"] == TERMINOLOGY


def test_overwrite_protection_and_outputs(tmp_path):
    result, paths = run(tmp_path)
    assert generated(tmp_path, "intensity_change_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        generate_intensity_change_map(
            fixed_path=paths[0], registered_path=paths[1], lung_mask_path=paths[2],
            case_id="synthetic", output_root=tmp_path / "outputs", skip_normalized=True,
        )


def test_no_registration_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("subprocess called"))
    result, _ = run(tmp_path, dry_run=True)
    assert result["completion_status"] == "dry-run"
