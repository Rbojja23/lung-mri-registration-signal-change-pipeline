from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import registration_metric_analysis as module  # noqa: E402
from registration_metric_analysis import analyze_registration_metrics  # noqa: E402


DIRECTION = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def make_image(array, *, origin=(2.0, 3.0, 4.0), vector=False):
    image = sitk.GetImageFromArray(np.asarray(array), isVector=vector)
    image.SetSpacing((1.1, 1.4, 2.0))
    image.SetOrigin(origin)
    image.SetDirection(DIRECTION)
    return image


def write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def inputs(tmp_path, *, fixed=None, moving=None, registered=None, with_mask=True):
    shape = (5, 6, 7)
    fixed = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) if fixed is None else np.asarray(fixed)
    moving = fixed + 4 if moving is None else np.asarray(moving)
    registered = fixed.astype(np.float32) + 1 if registered is None else np.asarray(registered)
    paths = {
        "fixed": write(tmp_path / "fixed.nii.gz", make_image(fixed)),
        "moving": write(tmp_path / "moving.nii.gz", make_image(moving)),
        "registered": write(tmp_path / "registered.nii.gz", make_image(registered)),
    }
    if with_mask:
        mask = np.zeros(shape, np.uint8); mask[1:4, 1:5, 1:6] = 1
        paths["mask"] = write(tmp_path / "mask.nii.gz", make_image(mask))
    return paths


def analyze(tmp_path, *, dry_run=False, with_mask=True, **kwargs):
    paths = inputs(tmp_path, with_mask=with_mask)
    result = analyze_registration_metrics(
        fixed_path=paths["fixed"], moving_path=paths["moving"],
        registered_path=paths["registered"], lung_mask_path=paths.get("mask"),
        output_dir=tmp_path / "analysis", dry_run=dry_run, **kwargs,
    )
    return result, paths


def test_known_metrics_and_improvements(tmp_path):
    result, _ = analyze(tmp_path, dry_run=True)
    metrics = result["metrics"]["whole_image"]["metrics"]
    assert metrics["mse"]["before"] == 16
    assert metrics["mse"]["after"] == 1
    assert metrics["mae"]["before"] == 4
    assert metrics["mae"]["after"] == 1
    assert metrics["rmse"]["before"] == 4
    assert metrics["ssd"]["before"] == 16 * 5 * 6 * 7
    assert metrics["mse"]["absolute_improvement"] == 15
    assert metrics["mse"]["percent_improvement"] == pytest.approx(93.75)
    assert metrics["mse"]["direction"] == "lower-is-better"
    assert metrics["ncc"]["direction"] == "higher-is-better"


def test_signed_absolute_error_distributions(tmp_path):
    result, _ = analyze(tmp_path, dry_run=True)
    domain = result["error_distributions"]["whole_image"]
    assert domain["signed_error_before"]["mean"] == -4
    assert domain["signed_error_after"]["mean"] == -1
    assert domain["absolute_error_before"]["median"] == 4
    assert domain["absolute_error_after"]["maximum"] == 1


def test_mask_domains_cover_whole_image(tmp_path):
    result, _ = analyze(tmp_path, dry_run=True)
    domains = result["analysis_domains"]
    assert set(domains) == {"whole_image", "lung_only", "outside_lung"}
    assert domains["lung_only"] + domains["outside_lung"] == domains["whole_image"]
    assert result["metrics"]["lung_only"]["voxel_count"] == domains["lung_only"]


def test_without_mask_only_whole_image(tmp_path):
    result, _ = analyze(tmp_path, dry_run=True, with_mask=False)
    assert list(result["analysis_domains"]) == ["whole_image"]
    assert result["display_scaling"]["domain"] == "whole_image"


def test_interpolation_statistics_and_types(tmp_path):
    result, _ = analyze(tmp_path, dry_run=True)
    interpolation = result["interpolation_analysis"]
    assert interpolation["moving_is_uint16"] is True
    assert interpolation["registered_is_float"] is True
    whole = interpolation["domains"]["whole_image"]
    assert whole["difference_in_mean_registered_minus_moving"] == -3
    assert whole["difference_in_variance_registered_minus_moving"] == pytest.approx(0)
    assert 0 <= whole["histogram_difference_l1_probability"] <= 2
    assert "Trilinear" in interpolation["new_value_explanation"]


def test_constant_images_have_undefined_ncc(tmp_path):
    shape = (5, 6, 7)
    paths = inputs(tmp_path, fixed=np.ones(shape), moving=np.ones(shape) * 2, registered=np.ones(shape))
    result = analyze_registration_metrics(
        fixed_path=paths["fixed"], moving_path=paths["moving"], registered_path=paths["registered"],
        output_dir=tmp_path / "analysis", dry_run=True,
    )
    assert result["metrics"]["whole_image"]["metrics"]["ncc"]["before"] is None


def test_figures_and_report_are_written(tmp_path):
    result, _ = analyze(tmp_path)
    for name in ("registration_metric_report.json", "metric_summary.png", "error_histograms.png", "difference_statistics.png"):
        assert (tmp_path / "analysis" / name).is_file()
    report = json.loads((tmp_path / "analysis/registration_metric_report.json").read_text())
    assert report["completion_status"] == "complete"
    assert report["writes_performed"] is True
    assert "universally" in report["interpretation"]["overall"]
    assert result["representative_slice"]["axial_z"] == 2


def test_dry_run_writes_nothing(tmp_path):
    result, _ = analyze(tmp_path, dry_run=True)
    assert result["writes_performed"] is False
    assert not (tmp_path / "analysis").exists()


def test_overwrite_protection(tmp_path):
    _, paths = analyze(tmp_path)
    with pytest.raises(FileExistsError):
        analyze_registration_metrics(
            fixed_path=paths["fixed"], moving_path=paths["moving"], registered_path=paths["registered"],
            lung_mask_path=paths["mask"], output_dir=tmp_path / "analysis",
        )


@pytest.mark.parametrize("target", ["moving", "registered", "mask"])
def test_geometry_mismatch_rejected(tmp_path, target):
    paths = inputs(tmp_path)
    changed = sitk.ReadImage(str(paths[target])); changed.SetOrigin((9.0, 3.0, 4.0)); sitk.WriteImage(changed, str(paths[target]))
    with pytest.raises(ValueError, match="geometry mismatch"):
        analyze_registration_metrics(
            fixed_path=paths["fixed"], moving_path=paths["moving"], registered_path=paths["registered"],
            lung_mask_path=paths["mask"], output_dir=tmp_path / "analysis", dry_run=True,
        )


@pytest.mark.parametrize("kind", ["empty", "nonbinary"])
def test_invalid_mask_rejected(tmp_path, kind):
    paths = inputs(tmp_path)
    values = np.zeros((5, 6, 7), np.uint8)
    if kind == "nonbinary": values[1, 1, 1] = 2
    write(paths["mask"], make_image(values))
    with pytest.raises(ValueError, match="empty|binary"):
        analyze_registration_metrics(
            fixed_path=paths["fixed"], moving_path=paths["moving"], registered_path=paths["registered"],
            lung_mask_path=paths["mask"], output_dir=tmp_path / "analysis", dry_run=True,
        )


def test_nonfinite_image_rejected_with_synthetic_in_memory_image(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    invalid_values = np.ones((5, 6, 7), np.float32); invalid_values[0, 0, 0] = np.nan
    invalid = make_image(invalid_values)
    original = module.sitk.ReadImage
    monkeypatch.setattr(module.sitk, "ReadImage", lambda path: invalid if Path(path) == paths["fixed"].resolve() else original(str(path)))
    with pytest.raises(ValueError, match="NaN or infinity"):
        analyze_registration_metrics(
            fixed_path=paths["fixed"], moving_path=paths["moving"], registered_path=paths["registered"],
            output_dir=tmp_path / "analysis", dry_run=True,
        )


def test_no_registration_or_subprocess_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("subprocess executed"))
    result, _ = analyze(tmp_path, dry_run=True)
    assert result["completion_status"] == "dry-run"


def test_report_is_privacy_safe_for_external_temp_paths(tmp_path):
    result, _ = analyze(tmp_path, dry_run=True)
    serialized = json.dumps(result).lower()
    assert str(tmp_path).lower() not in serialized
    assert "patient" not in serialized
