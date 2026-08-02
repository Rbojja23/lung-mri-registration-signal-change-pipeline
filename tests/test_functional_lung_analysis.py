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

import functional_lung_analysis as module  # noqa: E402
from functional_lung_analysis import TERMINOLOGY, analyze_functional_lung  # noqa: E402


DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def image(array, *, origin=(-4.0, 2.0, 3.0)):
    result = sitk.GetImageFromArray(np.asarray(array))
    result.SetSpacing((1.0, 1.5, 2.0)); result.SetOrigin(origin); result.SetDirection(DIRECTION)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); sitk.WriteImage(value, str(path)); return path


def synthetic_inputs(tmp_path, *, difference=None, include_sides=True):
    shape = (6, 8, 10)
    fixed = np.full(shape, 20.0, np.float32)
    mask = np.zeros(shape, np.uint8); mask[1:5, 1:7, 1:9] = 1
    if difference is None:
        difference = np.zeros(shape, np.float32)
        difference[1:5, 1:7, 1:5] = 3
        difference[1:5, 1:7, 5:9] = -2
    registered = fixed - np.asarray(difference)
    paths = {
        "fixed": write(tmp_path / "fixed.nii.gz", image(fixed)),
        "registered": write(tmp_path / "registered.nii.gz", image(registered)),
        "mask": write(tmp_path / "mask.nii.gz", image(mask)),
    }
    if include_sides:
        left = np.zeros(shape, np.uint8); left[1:5, 1:7, 5:9] = 1
        right = np.zeros(shape, np.uint8); right[1:5, 1:7, 1:5] = 1
        paths["left"] = write(tmp_path / "left.nii.gz", image(left))
        paths["right"] = write(tmp_path / "right.nii.gz", image(right))
    return paths, fixed, registered, mask.astype(bool)


def run(tmp_path, *, dry_run=False, include_sides=True, **kwargs):
    paths, _, _, _ = synthetic_inputs(tmp_path, include_sides=include_sides)
    result = analyze_functional_lung(
        fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"],
        left_lung_path=paths.get("left"), right_lung_path=paths.get("right"),
        output_dir=tmp_path / "functional", dry_run=dry_run, **kwargs,
    )
    return result, paths


def test_regional_signed_and_absolute_statistics(tmp_path):
    result, _ = run(tmp_path, dry_run=True)
    stats = result["regional_statistics"]
    assert stats["right_lung"]["signed_change"]["mean"] == 3
    assert stats["left_lung"]["signed_change"]["mean"] == -2
    assert stats["whole_lung"]["absolute_change"]["median"] == 2.5
    assert stats["right_lung"]["signed_change"]["positive_fraction"] == 1
    assert stats["left_lung"]["signed_change"]["negative_fraction"] == 1
    assert set(stats["whole_lung"]["signed_change"]["percentiles"]) == {"1", "5", "25", "50", "75", "95", "99"}


def test_outside_lung_never_contributes(tmp_path):
    paths, fixed, _, mask = synthetic_inputs(tmp_path)
    registered = fixed.copy(); registered[~mask] = -10000
    write(paths["registered"], image(registered))
    result = analyze_functional_lung(
        fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"],
        left_lung_path=paths["left"], right_lung_path=paths["right"], output_dir=tmp_path / "out", dry_run=True,
    )
    assert result["regional_statistics"]["whole_lung"]["signed_change"]["maximum"] == 0
    assert result["analysis_domain"]["outside_lung_contribution"] is False


def test_no_normalization_is_present(tmp_path):
    result, _ = run(tmp_path, dry_run=True)
    assert result["map_definitions"]["normalization_performed"] is False
    assert "(fixed - registered_moving) / fixed" not in json.dumps(result)


def test_slice_statistics_include_every_image_slice(tmp_path):
    result, _ = run(tmp_path, dry_run=True)
    slices = result["slice_statistics"]
    assert len(slices) == 6
    assert slices[0]["lung_voxel_count"] == 0 and slices[0]["signed_mean"] is None
    assert slices[2]["lung_voxel_count"] > 0
    assert slices[2]["physical_position_z_mm"] == 7.0


def test_hotspots_connected_regions_volume_centroid_bbox(tmp_path):
    result, _ = run(tmp_path, dry_run=True)
    positive = result["hotspot_statistics"]["largest_positive_change_region"]
    negative = result["hotspot_statistics"]["largest_negative_change_region"]
    assert positive["available"] and negative["available"]
    assert positive["voxel_count"] == 4 * 6 * 4
    assert positive["volume_mm3"] == pytest.approx(positive["voxel_count"] * 3.0)
    assert positive["bounding_box_half_open_zyx"]["minimum"] == [1, 1, 1]
    assert negative["signed_change_maximum"] < 0


def test_asymmetry_is_descriptive(tmp_path):
    result, _ = run(tmp_path, dry_run=True)
    asymmetry = result["asymmetry"]
    assert asymmetry["left_minus_right_mean_difference"] == -5
    assert "no biological interpretation" in asymmetry["effect_size_definition"]


def test_geometric_partitions_when_side_masks_omitted(tmp_path):
    result, _ = run(tmp_path, dry_run=True, include_sides=False)
    assert "geometric" in result["analysis_domain"]["left_right_partition_source"]
    assert result["lung_volumes"]["left_lung"]["voxel_count"] + result["lung_volumes"]["right_lung"]["voxel_count"] == result["lung_volumes"]["whole_lung"]["voxel_count"]
    assert any("geometric" in warning.lower() for warning in result["warnings"])


def test_side_masks_must_be_paired_and_match_union(tmp_path):
    paths, _, _, _ = synthetic_inputs(tmp_path)
    with pytest.raises(ValueError, match="supplied together"):
        analyze_functional_lung(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], left_lung_path=paths["left"], output_dir=tmp_path / "out", dry_run=True)
    bad = sitk.GetArrayFromImage(sitk.ReadImage(str(paths["left"]))); bad[1, 1, 4] = 1
    write(paths["left"], image(bad))
    with pytest.raises(ValueError, match="overlap"):
        analyze_functional_lung(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], left_lung_path=paths["left"], right_lung_path=paths["right"], output_dir=tmp_path / "out", dry_run=True)


def test_all_requested_figures_and_report_written(tmp_path):
    result, _ = run(tmp_path)
    names = ("functional_lung_report.json", "slice_mean.png", "slice_std.png", "axial_overlay.png", "coronal_overlay.png", "sagittal_overlay.png", "multi_slice_panel.png", "lung_histograms.png")
    for name in names:
        assert (tmp_path / "functional" / name).is_file()
    assert result["completion_status"] == "complete"
    assert result["display_scaling"]["limits"][0] == -result["display_scaling"]["limits"][1]
    report = json.loads((tmp_path / "functional/functional_lung_report.json").read_text())
    assert report["analysis_type"] == TERMINOLOGY
    assert "not a validated ventilation" in report["interpretation"]["conclusion"]


def test_dry_run_writes_nothing(tmp_path):
    result, _ = run(tmp_path, dry_run=True)
    assert result["writes_performed"] is False
    assert not (tmp_path / "functional").exists()


def test_overwrite_protection(tmp_path):
    _, paths = run(tmp_path)
    with pytest.raises(FileExistsError):
        analyze_functional_lung(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], left_lung_path=paths["left"], right_lung_path=paths["right"], output_dir=tmp_path / "functional")


@pytest.mark.parametrize("target", ["registered", "mask", "left", "right"])
def test_geometry_mismatch_rejected(tmp_path, target):
    paths, _, _, _ = synthetic_inputs(tmp_path)
    changed = sitk.ReadImage(str(paths[target])); changed.SetOrigin((8.0, 2.0, 3.0)); sitk.WriteImage(changed, str(paths[target]))
    with pytest.raises(ValueError, match="geometry mismatch"):
        analyze_functional_lung(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], left_lung_path=paths["left"], right_lung_path=paths["right"], output_dir=tmp_path / "out", dry_run=True)


@pytest.mark.parametrize("kind", ["empty", "nonbinary"])
def test_invalid_combined_mask_rejected(tmp_path, kind):
    paths, _, _, _ = synthetic_inputs(tmp_path)
    values = np.zeros((6, 8, 10), np.uint8)
    if kind == "nonbinary": values[1, 1, 1] = 2
    write(paths["mask"], image(values))
    with pytest.raises(ValueError, match="empty|binary"):
        analyze_functional_lung(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], output_dir=tmp_path / "out", dry_run=True)


def test_nonfinite_image_rejected_with_synthetic_memory_image(tmp_path, monkeypatch):
    paths, _, _, _ = synthetic_inputs(tmp_path)
    invalid_values = np.ones((6, 8, 10), np.float32); invalid_values[0, 0, 0] = np.inf
    invalid = image(invalid_values); original = module.sitk.ReadImage
    monkeypatch.setattr(module.sitk, "ReadImage", lambda path: invalid if Path(path) == paths["fixed"].resolve() else original(str(path)))
    with pytest.raises(ValueError, match="NaN or infinity"):
        analyze_functional_lung(fixed_path=paths["fixed"], registered_path=paths["registered"], lung_mask_path=paths["mask"], output_dir=tmp_path / "out", dry_run=True)


def test_privacy_safe_report_and_no_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("subprocess executed"))
    result, _ = run(tmp_path, dry_run=True)
    serialized = json.dumps(result).lower()
    assert str(tmp_path).lower() not in serialized
    assert "patient" not in serialized
    assert result["completion_status"] == "dry-run"
