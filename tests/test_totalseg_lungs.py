from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import totalseg_lungs  # noqa: E402
from totalseg_lungs import (  # noqa: E402
    DeviceResolution,
    MaskStats,
    build_manifest,
    build_totalseg_command,
    combine_lung_masks,
    make_plan,
    resolve_device,
    run_lung_segmentation,
    select_lung_masks,
    task_for_modality,
)


def make_reference_image() -> sitk.Image:
    image = sitk.GetImageFromArray(np.zeros((3, 4, 5), dtype=np.int16))
    image.SetSpacing((1.0, 2.0, 3.0))
    image.SetOrigin((4.0, 5.0, 6.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def write_mask(path: Path, array: np.ndarray, reference: sitk.Image) -> None:
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


def write_nifti(path: Path, image: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def test_ct_resolves_to_total():
    assert task_for_modality("ct") == "total"


def test_mr_resolves_to_total_mr():
    assert task_for_modality("MR") == "total_mr"


def test_safe_subprocess_command_construction(tmp_path):
    command = build_totalseg_command(
        executable="/bin/TotalSegmentator",
        input_path=tmp_path / "input.nii.gz",
        raw_dir=tmp_path / "raw",
        task="total_mr",
        device=DeviceResolution("mps", "mps", "mps"),
        fast=True,
    )

    assert command == [
        "/bin/TotalSegmentator",
        "-i",
        str(tmp_path / "input.nii.gz"),
        "-o",
        str(tmp_path / "raw"),
        "-ta",
        "total_mr",
        "-ot",
        "nifti",
        "-d",
        "mps",
        "--fast",
    ]


def test_dry_run_does_not_execute_subprocess_or_create_dirs(tmp_path, monkeypatch):
    input_path = tmp_path / "input.nii.gz"
    write_nifti(input_path, make_reference_image())
    output_root = tmp_path / "out"
    called = False

    def runner(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess should not run")

    monkeypatch.setattr(totalseg_lungs, "ensure_totalseg_available", lambda: "/bin/TotalSegmentator")
    monkeypatch.setattr(totalseg_lungs, "resolve_device", lambda _device: DeviceResolution("cpu", "cpu", "cpu"))

    result = run_lung_segmentation(
        input_path=input_path,
        input_type="nifti",
        modality="mr",
        case_id="case_a",
        output_root=output_root,
        dry_run=True,
        runner=runner,
    )

    assert result["status"] == "dry-run"
    assert not called
    assert not output_root.exists()


def test_output_overwrite_protection(tmp_path, monkeypatch):
    input_path = tmp_path / "input.nii.gz"
    write_nifti(input_path, make_reference_image())
    output_root = tmp_path / "out"
    (output_root / "case_a").mkdir(parents=True)
    monkeypatch.setattr(totalseg_lungs, "ensure_totalseg_available", lambda: "/bin/TotalSegmentator")
    monkeypatch.setattr(totalseg_lungs, "resolve_device", lambda _device: DeviceResolution("cpu", "cpu", "cpu"))

    with pytest.raises(FileExistsError):
        run_lung_segmentation(
            input_path=input_path,
            input_type="nifti",
            modality="ct",
            case_id="case_a",
            output_root=output_root,
            dry_run=True,
        )


def test_ct_label_selection_from_synthetic_filenames(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ["lung_upper_lobe_left", "lung_lower_lobe_right", "heart"]:
        (raw / f"{name}.nii.gz").touch()

    selected = select_lung_masks(raw, "CT")

    assert [label for label, _ in selected] == [
        "lung_upper_lobe_left",
        "lung_lower_lobe_right",
    ]


def test_mri_label_selection_from_synthetic_filenames(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ["lung_left", "lung_right", "liver"]:
        (raw / f"{name}.nii.gz").touch()

    selected = select_lung_masks(raw, "MR")

    assert [label for label, _ in selected] == ["lung_left", "lung_right"]


def test_rejects_unrelated_masks_and_no_verified_lung_labels(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "lung_vessels.nii.gz").touch()
    (raw / "heart.nii.gz").touch()

    with pytest.raises(FileNotFoundError, match="No verified"):
        select_lung_masks(raw, "MR")


def test_union_of_synthetic_binary_masks_values_and_geometry(tmp_path):
    reference = make_reference_image()
    mask_a = np.zeros((3, 4, 5), dtype=np.uint8)
    mask_b = np.zeros((3, 4, 5), dtype=np.uint8)
    mask_a[0, 0, 0] = 1
    mask_b[1, 2, 3] = 7
    path_a = tmp_path / "lung_left.nii.gz"
    path_b = tmp_path / "lung_right.nii.gz"
    write_mask(path_a, mask_a, reference)
    write_mask(path_b, mask_b, reference)

    combined, stats = combine_lung_masks(
        selected_mask_paths=[path_a, path_b],
        reference_image=reference,
    )

    array = sitk.GetArrayFromImage(combined)
    assert set(np.unique(array)) == {0, 1}
    assert array[0, 0, 0] == 1
    assert array[1, 2, 3] == 1
    assert combined.GetSize() == reference.GetSize()
    assert combined.GetSpacing() == reference.GetSpacing()
    assert combined.GetOrigin() == reference.GetOrigin()
    assert combined.GetDirection() == reference.GetDirection()
    assert stats.voxel_count == 2
    assert stats.volume_mm3 == pytest.approx(12.0)
    assert stats.volume_ml == pytest.approx(0.012)


def test_rejects_geometry_mismatch(tmp_path):
    reference = make_reference_image()
    other = make_reference_image()
    other.SetSpacing((9.0, 2.0, 3.0))
    path = tmp_path / "lung_left.nii.gz"
    write_mask(path, np.ones((3, 4, 5), dtype=np.uint8), other)

    with pytest.raises(ValueError, match="Geometry mismatch"):
        combine_lung_masks(selected_mask_paths=[path], reference_image=reference)


def test_safe_manifest_fields(tmp_path):
    image = make_reference_image()
    loaded = totalseg_lungs.LoadedMedicalImage(
        image=image,
        modality="MR",
        input_type="nifti",
        source_path=tmp_path / "input.nii.gz",
        size_xyz=image.GetSize(),
        array_shape_zyx=sitk.GetArrayFromImage(image).shape,
        spacing_xyz=image.GetSpacing(),
        origin_xyz=image.GetOrigin(),
        direction=image.GetDirection(),
    )
    plan = totalseg_lungs.SegmentationPlan(
        input_path=loaded.source_path,
        modality="MR",
        case_id="case_a",
        output_root=tmp_path / "out",
        case_dir=tmp_path / "out" / "case_a",
        raw_dir=tmp_path / "out" / "case_a" / "raw",
        lungs_dir=tmp_path / "out" / "case_a" / "lungs",
        manifest_path=tmp_path / "out" / "case_a" / "manifest.json",
        combined_mask_path=tmp_path / "out" / "case_a" / "lungs" / "combined_lungs.nii.gz",
        task="total_mr",
        device=DeviceResolution("cpu", "cpu", "cpu"),
        fast=False,
        command=[],
        totalseg_version="2.15.0",
    )

    manifest = build_manifest(
        plan=plan,
        loaded=loaded,
        selected_labels=["lung_left", "lung_right"],
        stats=MaskStats(2, 12.0, 0.012),
        start=totalseg_lungs.datetime.fromtimestamp(0, tz=totalseg_lungs.timezone.utc),
        end=totalseg_lungs.datetime.fromtimestamp(1, tz=totalseg_lungs.timezone.utc),
        runtime_seconds=1.0,
        status="success",
    )

    assert "patient" not in " ".join(manifest).lower()
    assert manifest["selected_lung_labels"] == ["lung_left", "lung_right"]
    assert manifest["completion_status"] == "success"


def test_device_resolution_logic_with_mocks(monkeypatch):
    monkeypatch.setattr(totalseg_lungs, "_cuda_available", lambda: True)
    monkeypatch.setattr(totalseg_lungs, "_mps_available", lambda: True)
    assert resolve_device("auto").resolved == "cuda"

    monkeypatch.setattr(totalseg_lungs, "_cuda_available", lambda: False)
    assert resolve_device("auto").resolved == "mps"

    monkeypatch.setattr(totalseg_lungs, "_mps_available", lambda: False)
    assert resolve_device("auto").resolved == "cpu"

    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_process_failure_handling(tmp_path, monkeypatch):
    input_path = tmp_path / "input.nii.gz"
    write_nifti(input_path, make_reference_image())
    output_root = tmp_path / "out"

    monkeypatch.setattr(totalseg_lungs, "ensure_totalseg_available", lambda: "/bin/TotalSegmentator")
    monkeypatch.setattr(totalseg_lungs, "resolve_device", lambda _device: DeviceResolution("cpu", "cpu", "cpu"))

    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=9, stderr="failed")

    with pytest.raises(RuntimeError, match="exit code 9"):
        run_lung_segmentation(
            input_path=input_path,
            input_type="nifti",
            modality="ct",
            case_id="case_a",
            output_root=output_root,
            overwrite=False,
            dry_run=False,
            runner=runner,
        )

    assert not (output_root / "case_a" / "manifest.json").exists()


def test_successful_mocked_run_creates_combined_mask_and_manifest(tmp_path, monkeypatch):
    input_path = tmp_path / "input.nii.gz"
    reference = make_reference_image()
    write_nifti(input_path, reference)
    output_root = tmp_path / "out"

    monkeypatch.setattr(totalseg_lungs, "ensure_totalseg_available", lambda: "/bin/TotalSegmentator")
    monkeypatch.setattr(totalseg_lungs, "resolve_device", lambda _device: DeviceResolution("cpu", "cpu", "cpu"))

    def runner(command, **kwargs):
        raw_dir = Path(command[command.index("-o") + 1])
        raw_dir.mkdir(parents=True, exist_ok=True)
        left = np.zeros((3, 4, 5), dtype=np.uint8)
        right = np.zeros((3, 4, 5), dtype=np.uint8)
        left[0, 0, 0] = 1
        right[0, 0, 1] = 1
        write_mask(raw_dir / "lung_left.nii.gz", left, reference)
        write_mask(raw_dir / "lung_right.nii.gz", right, reference)
        return subprocess.CompletedProcess(args=command, returncode=0, stderr="")

    manifest = run_lung_segmentation(
        input_path=input_path,
        input_type="nifti",
        modality="mr",
        case_id="case_a",
        output_root=output_root,
        dry_run=False,
        runner=runner,
    )

    assert manifest["mask_voxel_count"] == 2
    assert (output_root / "case_a" / "lungs" / "combined_lungs.nii.gz").exists()
    assert (output_root / "case_a" / "manifest.json").exists()


def test_reject_output_root_under_data_raw():
    unsafe = Path(__file__).resolve().parents[1] / "data" / "raw" / "case"

    with pytest.raises(ValueError, match="data/raw"):
        make_plan(
            input_path=Path("input.nii.gz"),
            modality="ct",
            case_id="case",
            output_root=unsafe,
            device="cpu",
            fast=False,
        )

