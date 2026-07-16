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

import registration_qc  # noqa: E402
from registration_qc import (  # noqa: E402
    build_registration_itksnap_plan,
    compare_registration_experiments,
    compute_similarity_metrics,
    evaluate_registration,
    extract_runtime_summary,
    parse_deeds_ssd,
    select_representative_slices,
)
from viewer_launcher import ITKSnapInstallation  # noqa: E402


def image_from_array(array, *, spacing=(1.0, 2.0, 3.0), origin=(4.0, 5.0, 6.0)):
    image = sitk.GetImageFromArray(np.asarray(array))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image


def write_image(path: Path, image: sitk.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def synthetic_experiment(tmp_path: Path, *, experiment="full"):
    root = tmp_path / experiment
    fixed_array = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    moving_array = fixed_array + 10
    registered_array = fixed_array + 2
    fixed = write_image(root / "preprocessed/fixed.nii.gz", image_from_array(fixed_array))
    moving = write_image(root / "preprocessed/moving.nii.gz", image_from_array(moving_array))
    registered = write_image(
        root / "deformable/moving_registered.nii.gz", image_from_array(registered_array)
    )
    manifest = root / "registration_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "case_id": "case",
                "experiment": experiment,
                "stages": {
                    "affine": {"runtime_seconds": 4.0},
                    "deformable": {"runtime_seconds": 6.0},
                },
            }
        )
    )
    log = root / "deformable/deeds_stdout.log"
    log.write_text("SSD before registration: 100 and after 25\n")
    return root, fixed, moving, registered, manifest


def test_identical_images_have_zero_errors_and_unit_ncc():
    array = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    image = image_from_array(array)
    metrics = compute_similarity_metrics(image, image)
    assert metrics["mean_squared_error"] == 0
    assert metrics["mean_absolute_error"] == 0
    assert metrics["normalized_cross_correlation"] == pytest.approx(1.0)


def test_registered_image_is_closer_and_mi_is_deterministic():
    fixed = image_from_array(np.arange(120).reshape(4, 5, 6).astype(np.float32))
    moving = image_from_array(sitk.GetArrayFromImage(fixed) + 20)
    registered = image_from_array(sitk.GetArrayFromImage(fixed) + 2)
    before = compute_similarity_metrics(fixed, moving, histogram_bins=16)
    after = compute_similarity_metrics(fixed, registered, histogram_bins=16)
    repeated = compute_similarity_metrics(fixed, registered, histogram_bins=16)
    assert after["mean_squared_error"] < before["mean_squared_error"]
    assert after["mean_absolute_error"] < before["mean_absolute_error"]
    assert after["mutual_information"] == repeated["mutual_information"]


def test_constant_image_ncc_is_explicitly_undefined():
    fixed = image_from_array(np.ones((3, 4, 5), dtype=np.float32))
    moving = image_from_array(np.ones((3, 4, 5), dtype=np.float32) * 2)
    metrics = compute_similarity_metrics(fixed, moving)
    assert metrics["normalized_cross_correlation"] is None
    assert "zero variance" in metrics["ncc_note"]


def test_grid_mismatch_is_rejected():
    fixed = image_from_array(np.ones((3, 4, 5)))
    moving = image_from_array(np.ones((3, 4, 6)))
    with pytest.raises(ValueError, match="grid mismatch"):
        compute_similarity_metrics(fixed, moving)


def test_masked_metrics_and_empty_mask_rejection(tmp_path):
    root, fixed, moving, registered, manifest = synthetic_experiment(tmp_path)
    mask_array = np.zeros((8, 10, 12), dtype=np.uint8)
    mask_array[2:6, 2:8, 3:10] = 1
    mask = write_image(tmp_path / "mask.nii.gz", image_from_array(mask_array))
    result = evaluate_registration(
        case_id="case",
        experiment="full",
        experiment_dir=root,
        fixed_path=fixed,
        moving_path=moving,
        registered_path=registered,
        registration_manifest_path=manifest,
        fixed_mask_path=mask,
        moving_mask_path=mask,
        registered_mask_path=mask,
        dry_run=True,
    )
    assert "fixed_mask_domain" in result["masked_metrics"]
    assert result["mask_usage"]["dice"]["before"] == 1.0
    empty = write_image(tmp_path / "empty.nii.gz", image_from_array(np.zeros_like(mask_array)))
    with pytest.raises(ValueError, match="mask is empty"):
        evaluate_registration(
            case_id="case", experiment="full", experiment_dir=root,
            fixed_path=fixed, moving_path=moving, registered_path=registered,
            registration_manifest_path=manifest, fixed_mask_path=empty, dry_run=True,
        )


def test_dice_is_not_fabricated_without_masks(tmp_path):
    root, fixed, moving, registered, manifest = synthetic_experiment(tmp_path)
    result = evaluate_registration(
        case_id="case", experiment="full", experiment_dir=root,
        fixed_path=fixed, moving_path=moving, registered_path=registered,
        registration_manifest_path=manifest, dry_run=True,
    )
    assert result["mask_usage"]["dice"]["before"] is None
    assert result["mask_usage"]["dice"]["after"] is None


def test_nan_and_inf_are_rejected(tmp_path, monkeypatch):
    root, fixed, moving, registered, manifest = synthetic_experiment(tmp_path)
    original = registration_qc.sitk.ReadImage
    invalid_array = np.ones((8, 10, 12), dtype=np.float32)
    invalid_array[0, 0, 0] = np.nan
    invalid = image_from_array(invalid_array)

    def reader(path):
        return invalid if Path(path) == Path(moving).resolve() else original(str(path))

    monkeypatch.setattr(registration_qc.sitk, "ReadImage", reader)
    with pytest.raises(ValueError, match="NaN"):
        evaluate_registration(
            case_id="case", experiment="full", experiment_dir=root,
            fixed_path=fixed, moving_path=moving, registered_path=registered,
            registration_manifest_path=manifest, dry_run=True,
        )


def test_runtime_and_final_ssd_parsing(tmp_path):
    root, _, _, _, manifest = synthetic_experiment(tmp_path)
    assert extract_runtime_summary(manifest) == {
        "affine_seconds": 4.0,
        "deformable_seconds": 6.0,
        "total_seconds": 10.0,
    }
    log = root / "deformable/deeds_stdout.log"
    log.write_text(
        "SSD before registration: 200 and after 100\n"
        "SSD before registration: 100 and after 25\n"
    )
    assert parse_deeds_ssd(log)["after"] == 25
    assert parse_deeds_ssd(tmp_path / "missing.log") is None


def test_representative_slice_center_and_foreground_extent():
    image = image_from_array(np.zeros((9, 11, 13)))
    assert select_representative_slices(image) == {
        "axial_z": 4, "coronal_y": 5, "sagittal_x": 6
    }
    mask = np.zeros((9, 11, 13), dtype=bool)
    mask[1:4, 2:7, 3:10] = True
    assert select_representative_slices(image, mask) == {
        "axial_z": 2, "coronal_y": 4, "sagittal_x": 6
    }


def test_figures_manifest_overwrite_and_privacy(tmp_path):
    root, fixed, moving, registered, manifest = synthetic_experiment(tmp_path)
    result = evaluate_registration(
        case_id="case", experiment="full", experiment_dir=root,
        fixed_path=fixed, moving_path=moving, registered_path=registered,
        registration_manifest_path=manifest,
    )
    for name in registration_qc.FIGURE_NAMES:
        assert (root / "qc" / name).is_file()
    assert (root / "qc/registration_qc.json").is_file()
    assert str(tmp_path) not in json.dumps(result)
    assert "patient" not in json.dumps(result).lower()
    with pytest.raises(FileExistsError):
        evaluate_registration(
            case_id="case", experiment="full", experiment_dir=root,
            fixed_path=fixed, moving_path=moving, registered_path=registered,
            registration_manifest_path=manifest,
        )


def test_dry_run_writes_nothing(tmp_path):
    root, fixed, moving, registered, manifest = synthetic_experiment(tmp_path)
    result = evaluate_registration(
        case_id="case", experiment="full", experiment_dir=root,
        fixed_path=fixed, moving_path=moving, registered_path=registered,
        registration_manifest_path=manifest, dry_run=True,
    )
    assert result["writes_performed"] is False
    assert not (root / "qc").exists()


def test_itksnap_command_uses_main_and_intensity_overlays(tmp_path):
    root, fixed, moving, registered, _ = synthetic_experiment(tmp_path)
    executable = tmp_path / "ITK-SNAP"
    executable.write_text("fake")
    executable.chmod(0o755)
    plan = build_registration_itksnap_plan(
        fixed,
        moving,
        registered,
        installation=ITKSnapInstallation(status="available", executable_path=executable),
    )
    command = plan.commands[0]
    assert command[1:3] == ["-g", str(Path(fixed).resolve())]
    assert command[3] == "-o"
    assert command[4:] == [str(Path(moving).resolve()), str(Path(registered).resolve())]
    assert "-s" not in command


def test_launch_mode_reuses_complete_qc_without_overwriting_or_external_app(tmp_path):
    root, fixed, moving, registered, manifest = synthetic_experiment(tmp_path)
    evaluate_registration(
        case_id="case", experiment="full", experiment_dir=root,
        fixed_path=fixed, moving_path=moving, registered_path=registered,
        registration_manifest_path=manifest,
    )
    qc_manifest_path = root / "qc/registration_qc.json"
    original = qc_manifest_path.read_bytes()
    calls = []

    result = evaluate_registration(
        case_id="case", experiment="full", experiment_dir=root,
        fixed_path=fixed, moving_path=moving, registered_path=registered,
        registration_manifest_path=manifest,
        launch_itksnap=True,
        launch_builder=lambda *args: "safe-plan",
        launcher=lambda plan, dry_run: calls.append((plan, dry_run)) or {"status": "mocked"},
    )

    assert result["completion_status"] == "reused"
    assert result["writes_performed"] is False
    assert calls == [("safe-plan", False)]
    assert qc_manifest_path.read_bytes() == original


def qc_manifest(path: Path, experiment: str, *, mse=1.0, runtime=10.0):
    data = {
        "case_id": "case",
        "experiment": experiment,
        "completion_status": "complete",
        "runtime": {"affine_seconds": runtime / 2, "deformable_seconds": runtime / 2, "total_seconds": runtime},
        "voxel_count": 100 if experiment == "full" else 50,
        "physical_volume_mm3": 100.0 if experiment == "full" else 50.0,
        "metrics": {
            "before": {"mean_squared_error": 10.0, "mean_absolute_error": 3.0, "normalized_cross_correlation": 0.5, "mutual_information": 0.2},
            "after": {"mean_squared_error": mse, "mean_absolute_error": mse, "normalized_cross_correlation": 0.8 if experiment == "full" else 0.9, "mutual_information": 0.6},
            "changes": {"mean_squared_error": {"improvement": 10.0 - mse}},
        },
        "deedsbcv_ssd": {"before": 10, "after": mse},
    }
    path.write_text(json.dumps(data))
    return path


def test_full_vs_cropped_comparison_has_metric_winners_but_no_overall_winner(tmp_path):
    full = qc_manifest(tmp_path / "full.json", "full", mse=1.0, runtime=20)
    cropped = qc_manifest(tmp_path / "cropped.json", "cropped", mse=2.0, runtime=10)
    output = tmp_path / "comparison"
    result = compare_registration_experiments(
        case_id="case", full_qc_path=full, cropped_qc_path=cropped, output_dir=output
    )
    assert result["metric_winners"]["mean_squared_error"] == "full"
    assert result["metric_winners"]["normalized_cross_correlation"] == "cropped"
    assert result["runtime_winner"] == "cropped"
    assert result["overall_anatomical_winner"] is None
    assert "expert visual review" in result["interpretation"]
    assert (output / "registration_comparison.json").is_file()
    assert (output / "registration_comparison.png").is_file()


def test_no_external_process_launch_in_evaluation_tests(tmp_path, monkeypatch):
    root, fixed, moving, registered, manifest = synthetic_experiment(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: pytest.fail("external process launched")
    )
    result = evaluate_registration(
        case_id="case", experiment="full", experiment_dir=root,
        fixed_path=fixed, moving_path=moving, registered_path=registered,
        registration_manifest_path=manifest, dry_run=True,
    )
    assert result["completion_status"] == "dry-run"
