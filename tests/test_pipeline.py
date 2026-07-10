from argparse import Namespace
from pathlib import Path
import sys

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import pipeline  # noqa: E402
import viewer_launcher  # noqa: E402
from version import PIPELINE_VERSION  # noqa: E402


def test_version_output(capsys):
    with pytest.raises(SystemExit):
        pipeline.build_parser().parse_args(["--version"])
    assert PIPELINE_VERSION in capsys.readouterr().out


def test_unified_cli_argument_parsing():
    args = pipeline.build_parser().parse_args(
        [
            "bbox-qc",
            "--image",
            "image.nii.gz",
            "--mask",
            "mask.nii.gz",
            "--bbox-json",
            "bbox.json",
            "--case-id",
            "case",
            "--modality",
            "mr",
            "--dry-run",
        ]
    )
    assert args.command == "bbox-qc"
    assert args.dry_run


def test_configuration_precedence(monkeypatch):
    captured = {}

    def fake_segment(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(pipeline, "run_lung_segmentation", fake_segment)
    args = Namespace(
        image="image.nii.gz",
        input_type="nifti",
        modality="mr",
        case_id="case",
        output_root=None,
        device=None,
        fast=None,
        overwrite=False,
        dry_run=True,
    )
    config = {"output_root": "custom_outputs", "totalsegmentator": {"device": "cpu", "fast": True}}

    pipeline.command_segment(args, config)

    assert captured["output_root"] == Path("custom_outputs") / "totalsegmentator"
    assert captured["device"] == "cpu"
    assert captured["fast"] is True


def test_run_all_stage_order(tmp_path):
    calls = []

    def stage(name):
        def _inner(*args, **kwargs):
            calls.append(name)
            return {"status": "ok"}
        return _inner

    args = run_all_args(tmp_path)
    pipeline.run_all(
        args,
        {"output_root": str(tmp_path / "outputs"), "launch": {"enabled": False}},
        stage_functions={
            "inspect": stage("inspect"),
            "segment": stage("segment"),
            "segmentation-qc": stage("segmentation-qc"),
            "bbox": stage("bbox"),
            "bbox-qc": stage("bbox-qc"),
            "export": stage("export"),
            "launch": stage("launch"),
        },
    )

    assert calls == ["inspect", "segment", "segmentation-qc", "bbox", "bbox-qc", "export"]


def test_run_all_resume_reuses_complete_outputs(tmp_path):
    output_root = tmp_path / "outputs"
    paths = pipeline._run_all_paths(output_root, "case", "mr")
    for key in ["segment_manifest", "combined_mask", "seg_qc_json", "bbox_json", "image_crop", "combined_crop", "bbox_qc_json", "export_report"]:
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_text("{}")
    calls = []
    args = run_all_args(tmp_path, resume=True)

    pipeline.run_all(
        args,
        {"output_root": str(output_root), "launch": {"enabled": False}},
        stage_functions={
            "inspect": lambda args: calls.append("inspect") or {},
            "segment": lambda *a: calls.append("segment") or {},
            "segmentation-qc": lambda args: calls.append("segmentation-qc") or {},
            "bbox": lambda *a: calls.append("bbox") or {},
            "bbox-qc": lambda args: calls.append("bbox-qc") or {},
            "export": lambda *a: calls.append("export") or {},
            "launch": lambda args: calls.append("launch") or {},
        },
    )

    assert calls == ["inspect"]


def test_run_all_failure_propagates_and_logs(tmp_path):
    args = run_all_args(tmp_path)

    with pytest.raises(RuntimeError):
        pipeline.run_all(
            args,
            {"output_root": str(tmp_path / "outputs")},
            stage_functions={
                "inspect": lambda args: {},
                "segment": lambda *a: (_ for _ in ()).throw(ValueError("boom")),
                "segmentation-qc": lambda args: {},
                "bbox": lambda *a: {},
                "bbox-qc": lambda args: {},
                "export": lambda *a: {},
                "launch": lambda args: {},
            },
        )

    log = tmp_path / "outputs" / "logs" / "case" / "pipeline_run.json"
    assert log.exists()
    assert "patient" not in log.read_text().lower()


def test_viewer_detection_and_dry_run(monkeypatch):
    monkeypatch.setattr(viewer_launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(viewer_launcher, "detect_viewers", lambda: {"osirix-lite": True, "slicer": False})

    plan = viewer_launcher.build_launch_plan(
        dicom_dir="dicom",
        rtstruct_path="rt.dcm",
        viewer="osirix-lite",
    )
    result = viewer_launcher.launch_plan(plan, dry_run=True)

    assert result["status"] == "dry-run"
    assert result["commands"][0][:3] == ["open", "-a", "OsiriX Lite"]


def test_fallback_package_launch_behavior(monkeypatch):
    monkeypatch.setattr(viewer_launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(viewer_launcher, "detect_viewers", lambda: {"horos": True, "slicer": False})

    plan = viewer_launcher.build_launch_plan(
        dicom_dir="dicom",
        fallback_dir="outputs/osirix/case/review_package",
    )

    assert any(command[0] == "open" for command in plan.commands)
    assert any("Fallback" in warning for warning in plan.warnings)


def test_non_macos_launch_failure(monkeypatch):
    monkeypatch.setattr(viewer_launcher.platform, "system", lambda: "Linux")

    with pytest.raises(RuntimeError):
        viewer_launcher.build_launch_plan(dicom_dir="dicom")


def run_all_args(tmp_path, resume=False):
    return Namespace(
        command="run-all",
        image="image.nii.gz",
        input_type="nifti",
        modality="mr",
        case_id="case",
        dicom_dir="dicom",
        output_root=str(tmp_path / "outputs"),
        device="cpu",
        fast=False,
        padding_mm=[5.0],
        overwrite=False,
        resume=resume,
        no_launch=True,
        viewer=None,
        dry_run=True,
    )
