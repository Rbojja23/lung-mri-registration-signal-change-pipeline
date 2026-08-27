from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from case_comparison import OUTPUT_NAMES, compare_registration_cases  # noqa: E402


EPSILON = 2.220446049250313e-16


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def registration(case_id, experiment, affine, deformable, *, commit="pinned"):
    return {
        "completion_status": "complete", "case_id": case_id, "experiment": experiment,
        "deedsbcv": {"pinned_upstream_commit": commit, "binaries": {
            "linearBCV": {"sha256": "linear"}, "deedsBCV": {"sha256": "deformable"},
        }},
        "commands": {
            "affine": ["linearBCV", "-F", "fixed", "-M", "moving", "-O", "out"],
            "deformable": ["deedsBCV", "-F", "fixed", "-M", "moving", "-O", "out", "-A", "affine"],
            "default_parameters_by_omission": True,
        },
        "stages": {"affine": {"runtime_seconds": affine}, "deformable": {"runtime_seconds": deformable}},
    }


def qc(case_id, experiment, voxels, runtime, scale=1.0):
    before = {"mean_squared_error": 100 * scale, "mean_absolute_error": 10 * scale, "normalized_cross_correlation": 0.5, "mutual_information": 0.2}
    after = {"mean_squared_error": 25 * scale, "mean_absolute_error": 5 * scale, "normalized_cross_correlation": 0.75, "mutual_information": 0.3}
    return {
        "completion_status": "complete", "case_id": case_id, "experiment": experiment,
        "voxel_count": voxels, "runtime": {"total_seconds": runtime}, "histogram_bins": 64,
        "metrics": {"before": before, "after": after},
        "deedsbcv_ssd": {"before": 1000 * scale, "after": 250 * scale, "source": "synthetic report"},
    }


def metric_report(scale=1.0):
    metrics = {}
    for name, before, after, direction in (
        ("mse", 100 * scale, 25 * scale, "lower-is-better"),
        ("mae", 10 * scale, 5 * scale, "lower-is-better"),
        ("rmse", 10 * scale, 5 * scale, "lower-is-better"),
        ("ncc", 0.5, 0.75, "higher-is-better"),
        ("mutual_information", 0.2, 0.3, "higher-is-better"),
    ):
        improvement = before - after if direction.startswith("lower") else after - before
        metrics[name] = {"before": before, "after": after, "absolute_improvement": improvement, "percent_improvement": 100 * improvement / before, "direction": direction}
    return {
        "completion_status": "complete",
        "histogram_configuration": {"mutual_information_bins": 64},
        "metric_definitions": {name: name for name in metrics},
        "metrics": {"lung_only": {"voxel_count": 100, "metrics": metrics}},
        "geometry_validation": {"resampling_performed": False},
        "display_scaling": {"percentile": 99.0},
    }


def stats(mean, *, signed=True):
    return {
        "mean": mean, "median": mean - 1, "standard_deviation": 3.0,
        "minimum": mean - 10, "maximum": mean + 10,
        "percentiles": {str(p): mean + p / 100 for p in (1, 5, 25, 50, 75, 95, 99)},
        "positive_fraction": 0.6 if signed else 1.0,
        "negative_fraction": 0.4 if signed else 0.0,
        "valid_voxel_count": 50,
    }


def intensity_report(mean):
    signed = {"combined_lung": stats(mean), "left_lung": stats(mean - 1), "right_lung": stats(mean + 1)}
    absolute = {"combined_lung": stats(abs(mean) + 5, signed=False), "left_lung": stats(abs(mean) + 4, signed=False), "right_lung": stats(abs(mean) + 6, signed=False)}
    return {
        "completion_status": "complete", "statistics": {"signed": signed, "absolute": absolute},
        "display_scaling": {"signed": {"clipping_percentile": 99.0}, "absolute": {"clipping_percentile": 99.0}},
        "geometry_validation": {"exact_agreement_required": True},
    }


def functional_report(mean):
    region = {"available": True, "voxel_count": 10, "volume_mm3": 20.0, "signed_change_mean": mean, "signed_change_minimum": mean - 2, "signed_change_maximum": mean + 2}
    return {
        "completion_status": "complete",
        "lung_volumes": {name: {"voxel_count": count, "volume_mm3": count * 2.0, "volume_ml": count / 500} for name, count in (("whole_lung", 100), ("left_lung", 45), ("right_lung", 55))},
        "slice_statistics": [
            {"lung_voxel_count": 10, "signed_mean": mean - 2, "signed_standard_deviation": 1.0},
            {"lung_voxel_count": 20, "signed_mean": mean + 2, "signed_standard_deviation": 4.0},
        ],
        "hotspot_statistics": {"largest_positive_change_region": region, "largest_negative_change_region": {**region, "signed_change_mean": -mean}, "connectivity": "6-connected", "threshold_rule": "sign only"},
        "asymmetry": {"left_mean_signed_change": mean - 1, "right_mean_signed_change": mean + 1, "left_minus_right_mean_difference": -2, "pooled_standardized_effect_size": -0.2},
        "display_scaling": {"lung_only_absolute_percentile": 99.0},
        "geometry_validation": {"exact_agreement_required": True},
    }


def normalized_report(mean, *, policy="epsilon", epsilon=EPSILON):
    return {
        "completion_status": "complete",
        "formula": {"expression": "(fixed - registered) / (fixed + epsilon)"},
        "denominator_policy": {"policy": policy, "epsilon_source": "machine", "epsilon": epsilon},
        "statistics": {"whole_lung": stats(mean), "left_lung": stats(mean - 0.1), "right_lung": stats(mean + 0.1)},
        "fixed_lung_intensity": {"minimum": 0, "maximum": 10, "exact_zero_voxel_count": 2, "near_zero_absolute_threshold_counts": {"1e-12": 2}},
        "display_scaling": {"valid_lung_absolute_percentile": 99.0, "limits": [-1, 1]},
        "geometry_validation": {"exact_agreement_required": True},
        "left_right_partition_source": "supplied semantic masks",
        "warnings": ["Machine epsilon does not stabilize small denominators."],
    }


def fixtures(tmp_path):
    root = tmp_path / "outputs"
    case_dirs = []
    for index, case_id in enumerate(("registration_case_1", "registration_case_2"), start=1):
        directory = root / "registration" / case_id; case_dirs.append(directory)
        scale = float(index)
        reports = {
            "full/registration_manifest.json": registration(case_id, "full", 10 * scale, 30 * scale),
            "cropped/registration_manifest.json": registration(case_id, "cropped", 3 * scale, 7 * scale),
            "full/qc/registration_qc.json": qc(case_id, "full", 1000, 40 * scale, scale),
            "cropped/qc/registration_qc.json": qc(case_id, "cropped", 400, 10 * scale, scale),
            "comparison/registration_comparison.json": {"completion_status": "complete", "case_id": case_id},
            "full/metric_analysis/registration_metric_report.json": metric_report(scale),
            "full/intensity_change/intensity_change_manifest.json": intensity_report(10 * scale),
            "full/functional_lung/functional_lung_report.json": functional_report(10 * scale),
            "full/normalized_intensity_epsilon_machine/normalized_intensity_report.json": normalized_report(0.1 * scale),
            "cropped/preprocessed/shared_crop_manifest.json": {"completion_status": "complete", "source_experiment": "full", "output_experiment": "cropped", "geometry_verification": {"same_bbox_applied": True, "no_resampling": True}},
        }
        for relative, report in reports.items(): write_json(directory / relative, report)
        write_json(root / "bbox" / f"{case_id}_fixed" / "bbox.json", {"padding": {"requested_mm_zyx": [5.0, 5.0, 5.0]}})
    return case_dirs


def run(tmp_path, **kwargs):
    case_1, case_2 = fixtures(tmp_path)
    result = compare_registration_cases(case_1_dir=case_1, case_2_dir=case_2, output_dir=tmp_path / "comparison", **kwargs)
    return result, case_1, case_2


def test_complete_comparison_and_calculations(tmp_path):
    result, _, _ = run(tmp_path)
    assert result["completion_status"] == "complete"
    assert result["compatibility_validation"]["core_method_compatible"] is True
    assert result["registration"]["case_1"]["derived"]["voxel_reduction_percent"] == pytest.approx(60)
    assert result["registration"]["case_1"]["derived"]["runtime_reduction_percent"] == pytest.approx(75)
    assert result["registration"]["case_1"]["full"]["metrics"]["mse"]["percent_improvement"] == pytest.approx(75)
    assert result["registration"]["case_1"]["full"]["metrics"]["rmse"]["before"] == 10
    assert result["overall_case_winner"] is None
    for filename in OUTPUT_NAMES.values(): assert (tmp_path / "comparison" / filename).is_file()


def test_signed_absolute_and_functional_summaries(tmp_path):
    result, _, _ = run(tmp_path, dry_run=True)
    intensity = result["intensity_change_comparison"]["cases"]["case_1"]
    assert intensity["signed_whole_lung"]["mean"] == 10
    assert intensity["absolute_whole_lung"]["percentiles"]["99"] == pytest.approx(15.99)
    assert intensity["left_right_signed"]["left_minus_right_mean"] == -2
    functional = result["functional_lung_comparison"]["cases"]["case_1"]
    assert functional["slice_wise_signed_mean_range"] == {"minimum": 8.0, "maximum": 12.0}
    assert functional["largest_positive_connected_region"]["voxel_count"] == 10


def test_missing_optional_report_is_reported_without_recomputation(tmp_path):
    case_1, case_2 = fixtures(tmp_path)
    (case_2 / "full/functional_lung/functional_lung_report.json").unlink()
    result = compare_registration_cases(case_1_dir=case_1, case_2_dir=case_2, output_dir=tmp_path / "out")
    assert result["functional_lung_comparison"]["available"] is False
    assert (tmp_path / "out/functional_lung_comparison.png").is_file()


def test_incompatible_methods_rejected(tmp_path):
    case_1, case_2 = fixtures(tmp_path)
    path = case_2 / "full/registration_manifest.json"; value = json.loads(path.read_text()); value["deedsbcv"]["pinned_upstream_commit"] = "different"; write_json(path, value)
    with pytest.raises(ValueError, match="Incompatible case methods"):
        compare_registration_cases(case_1_dir=case_1, case_2_dir=case_2, output_dir=tmp_path / "out", dry_run=True)


@pytest.mark.parametrize("field,value", [("policy", "absolute"), ("epsilon", 1e-6)])
def test_mismatched_epsilon_omits_normalized_comparison(tmp_path, field, value):
    case_1, case_2 = fixtures(tmp_path)
    path = case_2 / "full/normalized_intensity_epsilon_machine/normalized_intensity_report.json"; report = json.loads(path.read_text()); report["denominator_policy"][field] = value; write_json(path, report)
    result = compare_registration_cases(case_1_dir=case_1, case_2_dir=case_2, output_dir=tmp_path / "out")
    assert result["epsilon_normalized_comparison"]["available"] is False
    assert result["outputs"]["normalized"] is None
    assert not (tmp_path / "out/normalized_intensity_comparison.png").exists()


def test_dry_run_writes_nothing_and_paths_are_private(tmp_path):
    result, _, _ = run(tmp_path, dry_run=True)
    assert result["writes_performed"] is False and not (tmp_path / "comparison").exists()
    assert str(tmp_path) not in json.dumps(result)


def test_overwrite_protection(tmp_path):
    _, case_1, case_2 = run(tmp_path)
    with pytest.raises(FileExistsError):
        compare_registration_cases(case_1_dir=case_1, case_2_dir=case_2, output_dir=tmp_path / "comparison")


def test_figures_are_deterministic(tmp_path):
    case_1, case_2 = fixtures(tmp_path)
    hashes = []
    for name in ("first", "second"):
        destination = tmp_path / name
        compare_registration_cases(case_1_dir=case_1, case_2_dir=case_2, output_dir=destination)
        hashes.append({filename: hashlib.sha256((destination / filename).read_bytes()).hexdigest() for key, filename in OUTPUT_NAMES.items() if key != "report"})
    assert hashes[0] == hashes[1]


def test_no_registration_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("subprocess executed"))
    result, _, _ = run(tmp_path, dry_run=True)
    assert result["overall_case_winner"] is None
