from pathlib import Path
import plistlib
import subprocess
import sys

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import viewer_launcher  # noqa: E402
from viewer_launcher import (  # noqa: E402
    ITKSnapInstallation,
    build_itksnap_launch_plan,
    check_itksnap_health,
    detect_itksnap,
    launch_plan,
)


def make_bundle(root: Path, *, executable_name="ITK-SNAP", executable=True) -> Path:
    bundle = root / "ITK-SNAP.app"
    contents = bundle / "Contents"
    binary = contents / "MacOS" / executable_name
    binary.parent.mkdir(parents=True)
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump(
            {
                "CFBundleExecutable": executable_name,
                "CFBundleShortVersionString": "4.4.0",
                "CFBundleVersion": "59",
            },
            stream,
        )
    binary.write_text("fake")
    binary.chmod(0o755 if executable else 0o644)
    return bundle


def available_installation(executable: Path) -> ITKSnapInstallation:
    return ITKSnapInstallation(
        status="available", bundle_path=executable.parents[2], executable_path=executable
    )


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_system_application_detection_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer_launcher.platform, "system", lambda: "Darwin")
    bundle = make_bundle(tmp_path / "Applications")

    result = detect_itksnap(application_paths=[bundle])

    assert result.status == "available"
    assert result.executable_path == bundle / "Contents/MacOS/ITK-SNAP"
    assert result.short_version == "4.4.0"
    assert result.bundle_version == "59"


def test_user_application_fallback_and_system_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer_launcher.platform, "system", lambda: "Darwin")
    system = tmp_path / "system" / "ITK-SNAP.app"
    user = make_bundle(tmp_path / "user")
    assert detect_itksnap(application_paths=[system, user]).bundle_path == user
    system = make_bundle(tmp_path / "system")
    assert detect_itksnap(application_paths=[system, user]).bundle_path == system


def test_missing_application(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer_launcher.platform, "system", lambda: "Darwin")
    result = detect_itksnap(application_paths=[tmp_path / "missing.app"])
    assert result.status == "not_found"


@pytest.mark.parametrize("problem", ["malformed", "missing_key", "missing_binary", "not_executable"])
def test_unhealthy_bundles(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(viewer_launcher.platform, "system", lambda: "Darwin")
    bundle = make_bundle(tmp_path)
    plist = bundle / "Contents/Info.plist"
    binary = bundle / "Contents/MacOS/ITK-SNAP"
    if problem == "malformed":
        plist.write_text("not a plist")
    elif problem == "missing_key":
        with plist.open("wb") as stream:
            plistlib.dump({}, stream)
    elif problem == "missing_binary":
        binary.unlink()
    else:
        binary.chmod(0o644)

    assert detect_itksnap(application_paths=[bundle]).status == "unhealthy"


def test_health_check_success_failure_and_timeout(tmp_path):
    executable = touch(tmp_path / "ITK-SNAP")
    installation = available_installation(executable)
    calls = []

    def healthy(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="help", stderr="")

    assert check_itksnap_health(installation, runner=healthy).status == "available"
    assert calls[0][0] == [str(executable), "--help"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["capture_output"] is True

    def failed(command, **kwargs):
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="failed")

    failed_result = check_itksnap_health(installation, runner=failed)
    assert failed_result.status == "unhealthy"
    assert "status 7" in failed_result.message

    def timed_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    timeout_result = check_itksnap_health(installation, runner=timed_out, timeout=0.25)
    assert timeout_result.status == "unhealthy"
    assert "timed out" in timeout_result.message


def test_direct_command_with_multiple_inputs_and_spaces(tmp_path):
    executable = touch(tmp_path / "Fake App.app/Contents/MacOS/ITK-SNAP")
    image = touch(tmp_path / "case with spaces/main image.nii.gz")
    seg1 = touch(tmp_path / "case with spaces/left lung.nii")
    seg2 = touch(tmp_path / "case with spaces/right lung.nii.gz")
    labels = touch(tmp_path / "case with spaces/labels.txt")
    overlay1 = touch(tmp_path / "case with spaces/overlay one.nii.gz")
    overlay2 = touch(tmp_path / "case with spaces/overlay two.nii.gz")

    plan = build_itksnap_launch_plan(
        image_path=image,
        segmentation_paths=[seg1, seg2],
        label_description_path=labels,
        overlay_paths=[overlay1, overlay2],
        installation=available_installation(executable),
    )

    assert plan.commands == [[
        str(executable), "-g", str(image), "-s", str(seg1), str(seg2),
        "-l", str(labels), "-o", str(overlay1), str(overlay2),
    ]]


def test_workspace_command_and_direct_mode_exclusion(tmp_path):
    executable = touch(tmp_path / "ITK-SNAP")
    workspace = touch(tmp_path / "case.itksnap")
    image = touch(tmp_path / "image.nii.gz")
    installation = available_installation(executable)
    plan = build_itksnap_launch_plan(
        workspace_path=workspace, installation=installation
    )
    assert plan.commands == [[str(executable), "-w", str(workspace)]]
    with pytest.raises(ValueError, match="cannot be combined"):
        build_itksnap_launch_plan(
            workspace_path=workspace, image_path=image, installation=installation
        )


def test_input_validation(tmp_path):
    executable = touch(tmp_path / "ITK-SNAP")
    installation = available_installation(executable)
    seg = touch(tmp_path / "seg.nii.gz")
    with pytest.raises(ValueError, match="require a main image"):
        build_itksnap_launch_plan(segmentation_paths=[seg], installation=installation)
    with pytest.raises(FileNotFoundError, match="Main image"):
        build_itksnap_launch_plan(
            image_path=tmp_path / "missing.nii.gz", installation=installation
        )
    image = touch(tmp_path / "image.nii.gz")
    bad_seg = touch(tmp_path / "seg.nrrd")
    with pytest.raises(ValueError, match=".nii or .nii.gz"):
        build_itksnap_launch_plan(
            image_path=image, segmentation_paths=[bad_seg], installation=installation
        )


def test_dry_run_does_not_execute_and_real_launch_uses_injected_runner(tmp_path):
    executable = touch(tmp_path / "ITK-SNAP")
    image = touch(tmp_path / "image.nii.gz")
    plan = build_itksnap_launch_plan(
        image_path=image, installation=available_installation(executable)
    )
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0)

    assert launch_plan(plan, dry_run=True, runner=runner)["status"] == "dry-run"
    assert calls == []
    assert launch_plan(plan, runner=runner)["status"] == "launched"
    assert calls[0][1] == {"shell": False, "check": True}
