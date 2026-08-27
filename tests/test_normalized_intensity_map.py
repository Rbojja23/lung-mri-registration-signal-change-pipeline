from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import normalized_intensity_map as module  # noqa: E402
from generate_normalized_intensity_map import build_parser  # noqa: E402
from normalized_intensity_map import MACHINE_EPSILON_FLOAT64, TERMINOLOGY, generate_normalized_intensity_map  # noqa: E402


DIRECTION = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def image(array, *, origin=(2.0, 3.0, 4.0)):
    result = sitk.GetImageFromArray(np.asarray(array)); result.SetSpacing((1.1, 1.4, 2.0)); result.SetOrigin(origin); result.SetDirection(DIRECTION); return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); sitk.WriteImage(value, str(path)); return path


def inputs(tmp_path, *, fixed=None, registered=None, with_sides=True):
    shape = (5, 6, 8)
    fixed = np.full(shape, 10.0, np.float32) if fixed is None else np.asarray(fixed)
    registered = np.full(shape, 8.0, np.float32) if registered is None else np.asarray(registered)
    mask = np.zeros(shape, np.uint8); mask[1:4, 1:5, 1:7] = 1
    paths = {"fixed": write(tmp_path / "fixed.nii.gz", image(fixed)), "registered": write(tmp_path / "registered.nii.gz", image(registered)), "mask": write(tmp_path / "mask.nii.gz", image(mask))}
    if with_sides:
        left = np.zeros(shape, np.uint8); left[1:4, 1:5, 4:7] = 1
        right = np.zeros(shape, np.uint8); right[1:4, 1:5, 1:4] = 1
        paths["left"] = write(tmp_path / "left.nii.gz", image(left)); paths["right"] = write(tmp_path / "right.nii.gz", image(right))
    return paths, mask.astype(bool)


def run(tmp_path, *, dry_run=False, with_sides=True, fixed=None, registered=None, **kwargs):
    paths, mask = inputs(tmp_path, fixed=fixed, registered=registered, with_sides=with_sides)
    result = generate_normalized_intensity_map(
        fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"],
        left_lung_path=paths.get("left"), right_lung_path=paths.get("right"), output_dir=tmp_path / "normalized",
        denominator_policy=kwargs.pop("denominator_policy", "absolute"),
        denominator_threshold=kwargs.pop("denominator_threshold", 1.0), dry_run=dry_run, **kwargs,
    )
    return result, paths, mask


def nib_array(path):
    return np.transpose(np.asanyarray(nib.load(str(path)).dataobj), (2, 1, 0))


def test_formula_and_whole_lung_statistics(tmp_path):
    result, _, _ = run(tmp_path, dry_run=True)
    stats = result["statistics"]["whole_lung"]
    assert stats["mean"] == pytest.approx(0.2)
    assert stats["median"] == pytest.approx(0.2)
    assert stats["positive_fraction"] == 1
    assert stats["negative_fraction"] == 0
    assert result["formula"]["expression"] == "(fixed - registered) / fixed"


def test_positive_and_negative_values(tmp_path):
    fixed = np.full((5, 6, 8), 10.0, np.float32)
    registered = fixed.copy(); registered[:, :, :4] = 5; registered[:, :, 4:] = 15
    result, _, _ = run(tmp_path, dry_run=True, fixed=fixed, registered=registered)
    stats = result["statistics"]["whole_lung"]
    assert stats["minimum"] == pytest.approx(-0.5)
    assert stats["maximum"] == pytest.approx(0.5)
    assert stats["positive_fraction"] == pytest.approx(0.5)
    assert stats["negative_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("threshold", [0, -1, float("inf")])
def test_threshold_must_be_positive_and_finite(tmp_path, threshold):
    paths, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="greater than zero"):
        generate_normalized_intensity_map(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], output_dir=tmp_path / "out", denominator_policy="absolute", denominator_threshold=threshold, dry_run=True)


def test_unknown_policy_rejected(tmp_path):
    paths, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="absolute or epsilon"):
        generate_normalized_intensity_map(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], output_dir=tmp_path / "out", denominator_policy="percentile", denominator_threshold=1, dry_run=True)


@pytest.mark.parametrize("epsilon", ["machine", "1e-6"])
def test_cli_accepts_machine_and_explicit_epsilon(epsilon):
    args = build_parser().parse_args([
        "--fixed", "fixed.nii.gz", "--registered", "registered.nii.gz",
        "--lung-mask", "mask.nii.gz", "--output-dir", "epsilon-output",
        "--denominator-policy", "epsilon", "--epsilon", epsilon,
    ])
    assert args.denominator_policy == "epsilon" and args.epsilon == epsilon


def test_epsilon_formula_exactly_matches_requested_equation(tmp_path):
    fixed = np.full((5, 6, 8), 4.0, np.float32)
    registered = np.full_like(fixed, 1.0)
    result, _, mask = run(
        tmp_path, fixed=fixed, registered=registered, denominator_policy="epsilon",
        denominator_threshold=None, epsilon=0.5,
    )
    analytical = nib_array(tmp_path / "normalized/normalized_difference.nii.gz")
    assert analytical[mask] == pytest.approx((4.0 - 1.0) / (4.0 + 0.5))
    assert result["formula"]["expression"] == "(fixed - registered) / (fixed + epsilon)"
    assert result["denominator_policy"]["threshold_exclusion_performed"] is False


def test_machine_epsilon_and_zero_fixed_denominator_are_finite(tmp_path):
    fixed = np.full((5, 6, 8), 2.0, np.float32); fixed[2, 2, 2] = 0
    registered = fixed.copy(); registered[2, 2, 2] = -1
    result, _, mask = run(
        tmp_path, fixed=fixed, registered=registered, denominator_policy="epsilon",
        denominator_threshold=None, epsilon="machine",
    )
    analytical = nib_array(tmp_path / "normalized/normalized_difference.nii.gz")
    assert result["denominator_policy"]["epsilon_source"] == "machine"
    assert result["denominator_policy"]["epsilon"] == MACHINE_EPSILON_FLOAT64 == 2.220446049250313e-16
    assert analytical[2, 2, 2] == pytest.approx(1.0 / MACHINE_EPSILON_FLOAT64)
    assert np.isfinite(analytical[mask]).all()
    assert result["fixed_lung_intensity"]["exact_zero_voxel_count"] == 1


def test_explicit_epsilon_small_denominators_and_negative_fixed_values(tmp_path):
    fixed = np.full((5, 6, 8), 2.0, np.float64)
    registered = fixed.copy()
    fixed[2, 2, 2] = 1e-12; registered[2, 2, 2] = 0
    fixed[2, 2, 3] = -2; registered[2, 2, 3] = -1
    result, _, _ = run(
        tmp_path, fixed=fixed, registered=registered, denominator_policy="epsilon",
        denominator_threshold=None, epsilon=1e-6,
    )
    analytical = nib_array(tmp_path / "normalized/normalized_difference.nii.gz")
    assert analytical[2, 2, 2] == pytest.approx(1e-12 / (1e-12 + 1e-6))
    assert analytical[2, 2, 3] == pytest.approx((-2 - -1) / (-2 + 1e-6))
    assert result["denominator_policy"]["epsilon_source"] == "explicit"
    assert result["fixed_lung_intensity"]["negative_voxel_count"] == 1
    assert any("asymmetric" in warning for warning in result["warnings"])


@pytest.mark.parametrize("epsilon", [0, -1, float("nan"), float("inf"), "not-a-number", None])
def test_epsilon_must_be_positive_and_finite(tmp_path, epsilon):
    paths, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="Epsilon must"):
        generate_normalized_intensity_map(
            fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"],
            output_dir=tmp_path / "out", denominator_policy="epsilon", epsilon=epsilon, dry_run=True,
        )


def test_epsilon_rejects_exact_cancellation_for_negative_fixed_value(tmp_path):
    fixed = np.full((5, 6, 8), 2.0, np.float64); fixed[2, 2, 2] = -0.5
    with pytest.raises(ValueError, match=r"F \+ epsilon is exactly zero"):
        run(
            tmp_path, fixed=fixed, denominator_policy="epsilon",
            denominator_threshold=None, epsilon=0.5, dry_run=True,
        )


def test_epsilon_outputs_nan_outside_lung_and_finite_display(tmp_path):
    result, _, mask = run(
        tmp_path, denominator_policy="epsilon", denominator_threshold=None, epsilon=1e-6,
    )
    analytical = nib_array(tmp_path / "normalized/normalized_difference.nii.gz")
    display = nib_array(tmp_path / "normalized/normalized_difference_display.nii.gz")
    assert np.isnan(analytical[~mask]).all()
    assert np.isfinite(display).all() and np.all(display[~mask] == 0)
    assert result["denominator_policy"]["valid_voxel_count"] == int(np.count_nonzero(mask))


def test_epsilon_left_right_statistics_geometry_and_report(tmp_path):
    fixed = np.full((5, 6, 8), 10.0, np.float32); registered = fixed.copy()
    registered[:, :, :4] = 5; registered[:, :, 4:] = 12
    result, paths, _ = run(
        tmp_path, fixed=fixed, registered=registered, denominator_policy="epsilon",
        denominator_threshold=None, epsilon=1.0,
    )
    assert result["statistics"]["right_lung"]["mean"] == pytest.approx(5 / 11)
    assert result["statistics"]["left_lung"]["mean"] == pytest.approx(-2 / 11)
    output = sitk.ReadImage(str(tmp_path / "normalized/normalized_difference.nii.gz"))
    source = sitk.ReadImage(str(paths["fixed"]))
    assert (output.GetSize(), output.GetSpacing(), output.GetOrigin(), output.GetDirection()) == (source.GetSize(), source.GetSpacing(), source.GetOrigin(), source.GetDirection())
    assert nib.load(str(tmp_path / "normalized/normalized_difference.nii.gz")).get_data_dtype() == np.dtype(np.float32)
    assert result["denominator_policy"]["computation_dtype"] == "float64"


def test_epsilon_dry_run_privacy_and_no_registration_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("subprocess executed"))
    result, _, _ = run(
        tmp_path, dry_run=True, denominator_policy="epsilon", denominator_threshold=None, epsilon="machine",
    )
    assert result["writes_performed"] is False and not (tmp_path / "normalized").exists()
    serialized = json.dumps(result).lower()
    assert str(tmp_path).lower() not in serialized and "patient" not in serialized


def test_epsilon_overwrite_protection(tmp_path):
    _, paths, _ = run(tmp_path, denominator_policy="epsilon", denominator_threshold=None, epsilon=1e-6)
    with pytest.raises(FileExistsError):
        generate_normalized_intensity_map(
            fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"],
            left_lung_path=paths["left"], right_lung_path=paths["right"], output_dir=tmp_path / "normalized",
            denominator_policy="epsilon", epsilon=1e-6,
        )


def test_invalid_denominators_and_outside_lung_are_nan(tmp_path):
    fixed = np.full((5, 6, 8), 10.0, np.float32); fixed[2, 2, 2] = 0; fixed[2, 2, 3] = 0.1
    result, _, mask = run(tmp_path, fixed=fixed, registered=fixed - 1, denominator_threshold=0.5)
    analytical = nib_array(tmp_path / "normalized/normalized_difference.nii.gz")
    assert np.isnan(analytical[2, 2, 2]) and np.isnan(analytical[2, 2, 3])
    assert np.isnan(analytical[~mask]).all()
    assert result["denominator_policy"]["invalid_denominator_count"] == 2
    assert result["denominator_policy"]["valid_voxel_count"] == int(np.count_nonzero(mask)) - 2


def test_display_is_finite_zero_outside_and_symmetrically_clipped(tmp_path):
    fixed = np.full((5, 6, 8), 10.0, np.float32); registered = fixed.copy(); registered[2, 2, 2] = -990
    result, _, mask = run(tmp_path, fixed=fixed, registered=registered, display_percentile=90)
    analytical = nib_array(tmp_path / "normalized/normalized_difference.nii.gz")
    display = nib_array(tmp_path / "normalized/normalized_difference_display.nii.gz")
    assert np.nanmax(analytical) == 100
    assert np.isfinite(display).all() and np.all(display[~mask] == 0)
    low, high = result["display_scaling"]["limits"]
    assert low == -high and np.max(display) <= high
    assert result["display_scaling"]["clipped_voxel_count"] >= 1


def test_fixed_geometry_and_float32_preserved(tmp_path):
    _, paths, _ = run(tmp_path)
    output_path = tmp_path / "normalized/normalized_difference.nii.gz"
    output = sitk.ReadImage(str(output_path)); fixed = sitk.ReadImage(str(paths["fixed"]))
    assert output.GetSize() == fixed.GetSize(); assert output.GetSpacing() == fixed.GetSpacing(); assert output.GetOrigin() == fixed.GetOrigin(); assert output.GetDirection() == fixed.GetDirection()
    assert nib.load(str(output_path)).get_data_dtype() == np.dtype(np.float32)


def test_left_right_statistics_and_union(tmp_path):
    fixed = np.full((5, 6, 8), 10.0, np.float32); registered = fixed.copy(); registered[:, :, :4] = 5; registered[:, :, 4:] = 12
    result, _, _ = run(tmp_path, dry_run=True, fixed=fixed, registered=registered)
    assert result["statistics"]["right_lung"]["mean"] == pytest.approx(0.5)
    assert result["statistics"]["left_lung"]["mean"] == pytest.approx(-0.2)
    assert set(result["statistics"]["whole_lung"]["percentiles"]) == {"1", "5", "25", "50", "75", "95", "99"}


def test_geometric_partition_when_sides_omitted(tmp_path):
    result, _, _ = run(tmp_path, dry_run=True, with_sides=False)
    assert "geometric" in result["left_right_partition_source"]
    assert any("geometric" in warning.lower() for warning in result["warnings"])


def test_side_masks_paired_nonoverlapping_and_exact_union(tmp_path):
    paths, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="supplied together"):
        generate_normalized_intensity_map(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], left_lung_path=paths["left"], output_dir=tmp_path / "out", denominator_policy="absolute", denominator_threshold=1, dry_run=True)
    left = sitk.GetArrayFromImage(sitk.ReadImage(str(paths["left"]))); left[1, 1, 3] = 1; write(paths["left"], image(left))
    with pytest.raises(ValueError, match="overlap"):
        generate_normalized_intensity_map(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], left_lung_path=paths["left"], right_lung_path=paths["right"], output_dir=tmp_path / "out", denominator_policy="absolute", denominator_threshold=1, dry_run=True)


def test_requested_figures_report_and_terminology(tmp_path):
    result, _, _ = run(tmp_path)
    for name in ("normalized_difference.nii.gz", "normalized_difference_display.nii.gz", "normalized_overlay_axial.png", "normalized_overlay_coronal.png", "normalized_overlay_sagittal.png", "normalized_multi_slice.png", "normalized_intensity_report.json"):
        assert (tmp_path / "normalized" / name).is_file()
    report = json.loads((tmp_path / "normalized/normalized_intensity_report.json").read_text())
    assert report["analysis_type"] == TERMINOLOGY
    assert report["interpretation"]["terminology"] == TERMINOLOGY
    assert "NOT a validated ventilation" in report["interpretation"]["not_ventilation"]
    assert result["completion_status"] == "complete"


def test_dry_run_writes_nothing(tmp_path):
    result, _, _ = run(tmp_path, dry_run=True)
    assert result["writes_performed"] is False and not (tmp_path / "normalized").exists()


def test_overwrite_protection(tmp_path):
    _, paths, _ = run(tmp_path)
    with pytest.raises(FileExistsError):
        generate_normalized_intensity_map(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], left_lung_path=paths["left"], right_lung_path=paths["right"], output_dir=tmp_path / "normalized", denominator_policy="absolute", denominator_threshold=1)


@pytest.mark.parametrize("target", ["registered", "mask", "left", "right"])
def test_geometry_mismatch_rejected(tmp_path, target):
    paths, _ = inputs(tmp_path); changed = sitk.ReadImage(str(paths[target])); changed.SetOrigin((9.0, 3.0, 4.0)); sitk.WriteImage(changed, str(paths[target]))
    with pytest.raises(ValueError, match="geometry mismatch"):
        generate_normalized_intensity_map(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], left_lung_path=paths["left"], right_lung_path=paths["right"], output_dir=tmp_path / "out", denominator_policy="absolute", denominator_threshold=1, dry_run=True)


@pytest.mark.parametrize("kind", ["empty", "nonbinary"])
def test_invalid_mask_rejected(tmp_path, kind):
    paths, _ = inputs(tmp_path); values = np.zeros((5, 6, 8), np.uint8)
    if kind == "nonbinary": values[1, 1, 1] = 2
    write(paths["mask"], image(values))
    with pytest.raises(ValueError, match="empty|binary"):
        generate_normalized_intensity_map(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], output_dir=tmp_path / "out", denominator_policy="absolute", denominator_threshold=1, dry_run=True)


def test_nonfinite_image_rejected_with_synthetic_memory_image(tmp_path, monkeypatch):
    paths, _ = inputs(tmp_path); values = np.ones((5, 6, 8), np.float32); values[0, 0, 0] = np.nan; invalid = image(values); original = module.sitk.ReadImage
    monkeypatch.setattr(module.sitk, "ReadImage", lambda path: invalid if Path(path) == paths["fixed"].resolve() else original(str(path)))
    with pytest.raises(ValueError, match="NaN or infinity"):
        generate_normalized_intensity_map(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], output_dir=tmp_path / "out", denominator_policy="absolute", denominator_threshold=1, dry_run=True)


def test_privacy_safe_and_no_registration_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("subprocess executed"))
    result, _, _ = run(tmp_path, dry_run=True)
    serialized = json.dumps(result).lower()
    assert str(tmp_path).lower() not in serialized and "patient" not in serialized
    assert result["formula"]["interpolation_performed_by_this_checkpoint"] is False
