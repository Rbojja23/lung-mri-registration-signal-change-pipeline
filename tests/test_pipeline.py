from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import pipeline  # noqa: E402
import viewer_launcher  # noqa: E402
from version import PIPELINE_VERSION  # noqa: E402


def test_version_output(capsys):
    with pytest.raises(SystemExit):
        pipeline.build_parser().parse_args(["--version"])
    assert PIPELINE_VERSION == "1.1.0"
    assert capsys.readouterr().out.strip() == "1.1.0"


def test_top_level_help_presents_itksnap_primary(capsys):
    with pytest.raises(SystemExit):
        pipeline.build_parser().parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "pipeline v1.1.0" in help_text
    assert "prepare-itksnap" in help_text
    assert "primary ITK-SNAP" in help_text


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

    args = run_all_args(tmp_path, launch=True, no_launch=False, dry_run=False)
    pipeline.run_all(
        args,
        {"output_root": str(tmp_path / "outputs"), "launch": {"enabled": False}},
        stage_functions={
            "inspect": stage("inspect"),
            "segment": stage("segment"),
            "segmentation-qc": stage("segmentation-qc"),
            "bbox": stage("bbox"),
            "bbox-qc": stage("bbox-qc"),
            "prepare-itksnap": stage("prepare-itksnap"),
            "export": stage("export"),
            "launch": stage("launch"),
        },
    )

    assert calls == [
        "inspect", "segment", "segmentation-qc", "bbox", "bbox-qc",
        "prepare-itksnap", "launch",
    ]


def test_default_run_all_does_not_export_osirix(tmp_path):
    calls = []
    inspected = {}
    funcs = stage_functions(calls)
    funcs["inspect"] = lambda args: inspected.update(vars(args)) or {"status": "ok"}
    args = run_all_args(tmp_path, dry_run=False)
    result = pipeline.run_all(
        args,
        {"output_root": str(tmp_path / "outputs"), "launch": {"enabled": False}},
        stage_functions=funcs,
    )
    assert "export" not in calls
    assert "prepare-itksnap" in calls
    assert inspected["dicom_dir"] is None
    assert result["viewer"] == "itksnap"
    assert result["launch_status"] == "disabled"


def test_default_configuration_selects_itksnap():
    config = pipeline.load_config("config/default.yaml")
    assert pipeline.cfg(config, "launch.viewer") == "itksnap"


def test_optional_osirix_export_is_explicit(tmp_path):
    calls = []
    args = run_all_args(tmp_path, dry_run=False, export_rtstruct=True, dicom_dir="dicom")
    pipeline.run_all(
        args,
        {"output_root": str(tmp_path / "outputs"), "launch": {"enabled": False}},
        stage_functions=stage_functions(calls),
    )
    assert calls.index("prepare-itksnap") < calls.index("export")


def test_run_all_resume_reuses_complete_outputs(tmp_path):
    output_root = tmp_path / "outputs"
    paths = pipeline._run_all_paths(output_root, "case", "mr")
    image_path, geometry = write_reference_image(tmp_path / "image.nii.gz")
    for key in ["segment_manifest", "combined_mask", "seg_qc_json", "bbox_json", "image_crop", "combined_crop", "bbox_qc_json"]:
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_text("{}")
    for path in paths["individual_masks"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mask")
    write_valid_itksnap_package(paths, image_path, geometry)
    calls = []
    args = run_all_args(tmp_path, resume=True, image=str(image_path))

    result = pipeline.run_all(
        args,
        {"output_root": str(output_root), "launch": {"enabled": False}},
        stage_functions=stage_functions(calls),
    )

    assert calls == ["inspect", "launch"]
    assert result["prepare-itksnap"]["status"] == "reused"
    assert result["stage_timings_seconds"]["prepare-itksnap"] == 0.0


def test_resume_rejects_partial_itksnap_package(tmp_path):
    calls = []
    captured = {}
    output_root = tmp_path / "outputs"
    paths = pipeline._run_all_paths(output_root, "case", "mr")
    paths["itksnap_lungs"].parent.mkdir(parents=True)
    paths["itksnap_lungs"].touch()
    funcs = stage_functions(calls)
    funcs["prepare-itksnap"] = (
        lambda args, config: captured.update(vars(args)) or calls.append("prepare-itksnap") or {}
    )
    args = run_all_args(tmp_path, resume=True, dry_run=False)
    pipeline.run_all(
        args,
        {"output_root": str(output_root), "launch": {"enabled": False}},
        stage_functions=funcs,
    )
    assert "prepare-itksnap" in calls
    assert captured["overwrite"] is True


@pytest.mark.parametrize("field", ["label_mapping", "bbox_convention"])
def test_resume_rejects_stale_itksnap_semantics(tmp_path, field):
    paths = pipeline._run_all_paths(tmp_path / "outputs", "case", "mr")
    image_path, geometry = write_reference_image(tmp_path / "image.nii.gz")
    write_valid_itksnap_package(paths, image_path, geometry)
    manifest = json.loads(paths["itksnap_manifest"].read_text())
    manifest[field] = {} if field == "label_mapping" else "closed xyz"
    paths["itksnap_manifest"].write_text(json.dumps(manifest))
    assert not pipeline._valid_itksnap_package(paths, image_path)


def test_mr_semantic_mask_resolution():
    paths = pipeline._run_all_paths(Path("outputs"), "case", "mr")
    assert [path.name for path in paths["individual_masks"]] == [
        "lung_left.nii.gz", "lung_right.nii.gz"
    ]


def test_missing_mr_semantic_mask_fails_clearly(tmp_path):
    funcs = stage_functions([])
    funcs["prepare-itksnap"] = pipeline.command_prepare_itksnap
    image_path, _ = write_reference_image(tmp_path / "image.nii.gz")
    args = run_all_args(tmp_path, dry_run=False, image=str(image_path))
    with pytest.raises(RuntimeError, match="Left lung mask does not exist"):
        pipeline.run_all(
            args,
            {"output_root": str(tmp_path / "outputs"), "launch": {"enabled": False}},
            stage_functions=funcs,
        )


def test_ct_viewer_preparation_fails_with_clear_message(tmp_path):
    args = run_all_args(tmp_path, modality="ct")
    with pytest.raises(ValueError, match="CT label preparation not yet configured"):
        pipeline.run_all(args, {"output_root": str(tmp_path / "outputs")})


def test_run_all_failure_propagates_and_logs(tmp_path):
    args = run_all_args(tmp_path, dry_run=False)

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
                "prepare-itksnap": lambda *a: {},
                "export": lambda *a: {},
                "launch": lambda args: {},
            },
        )

    log = tmp_path / "outputs" / "logs" / "case" / "pipeline_run.json"
    assert log.exists()
    assert "patient" not in log.read_text().lower()


def test_run_all_dry_run_plans_products_and_launch_without_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "command_inspect", lambda args: {"status": "dry-run"})
    monkeypatch.setattr(pipeline, "command_segment", lambda args, config: {"status": "dry-run"})
    args = run_all_args(tmp_path, dry_run=True, no_launch=True)

    result = pipeline.run_all(
        args,
        {"output_root": str(tmp_path / "outputs"), "launch": {"enabled": False, "viewer": "itksnap"}},
    )

    assert result["prepare-itksnap"]["completion_status"] == "dry-run"
    assert result["launch"]["argument_roles"]["main_image"] == "image.nii.gz"
    assert result["launch"]["external_application_executed"] is False
    assert result["run_log"]["status"] == "not-written-during-dry-run"
    assert not (tmp_path / "outputs").exists()


def test_no_launch_still_prepares_products(tmp_path):
    calls = []
    args = run_all_args(tmp_path, dry_run=False, no_launch=True)
    result = pipeline.run_all(
        args,
        {"output_root": str(tmp_path / "outputs"), "launch": {"enabled": True}},
        stage_functions=stage_functions(calls),
    )
    assert "prepare-itksnap" in calls
    assert "launch" not in calls
    assert result["launch"]["status"] == "disabled"


def test_launch_uses_full_image_combined_labels_and_descriptions(tmp_path):
    captured = {}
    funcs = stage_functions([])
    funcs["launch"] = lambda args: captured.update(vars(args)) or {"status": "launched"}
    args = run_all_args(tmp_path, dry_run=False, launch=True, no_launch=False)
    result = pipeline.run_all(
        args,
        {"output_root": str(tmp_path / "outputs"), "launch": {"viewer": "itksnap"}},
        stage_functions=funcs,
    )
    assert captured["image"] == args.image
    assert captured["segmentation"] == [
        str(tmp_path / "outputs/itksnap/case/lungs_and_bbox_labels.nii.gz")
    ]
    assert captured["labels"] == str(tmp_path / "outputs/itksnap/case/label_descriptions.txt")
    assert result["launch_status"] == "launched"


def test_overwrite_is_scoped_to_selected_itksnap_case(tmp_path):
    captured = {}
    funcs = stage_functions([])
    funcs["prepare-itksnap"] = lambda args, config: captured.update(vars(args)) or {"status": "ok"}
    args = run_all_args(tmp_path, dry_run=False, overwrite=True)
    pipeline.run_all(
        args,
        {"output_root": str(tmp_path / "outputs"), "launch": {"enabled": False}},
        stage_functions=funcs,
    )
    assert captured["overwrite"] is True
    assert captured["output_root"] == str(tmp_path / "outputs/itksnap")
    assert captured["left_lung"].endswith("/case/lungs/lung_left.nii.gz")


def test_pipeline_log_records_viewer_products_and_launch_status(tmp_path):
    args = run_all_args(tmp_path, dry_run=False)
    pipeline.run_all(
        args,
        {"output_root": str(tmp_path / "outputs"), "launch": {"enabled": False}},
        stage_functions=stage_functions([]),
    )
    data = json.loads((tmp_path / "outputs/logs/case/pipeline_run.json").read_text())
    assert data["viewer"] == "itksnap"
    assert data["launch_status"] == "disabled"
    assert "prepare-itksnap" in data["completed_stages"]
    assert len(data["viewer_product_paths"]) == 5
    assert str(tmp_path) not in json.dumps(data)


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


def test_launch_rejects_mixed_viewer_arguments():
    args = Namespace(
        viewer="itksnap", dicom_dir="dicom", rtstruct=None, fallback_dir=None,
        nifti=[], image=None, segmentation=[], labels=None, overlay=[],
        workspace=None, health_check=False, dry_run=True,
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        pipeline.command_launch(args)

    args.viewer = "osirix"
    args.dicom_dir = None
    args.image = "image.nii.gz"
    with pytest.raises(ValueError, match="cannot be combined"):
        pipeline.command_launch(args)


def stage_functions(calls):
    def stage(name):
        return lambda *args, **kwargs: calls.append(name) or {"status": "ok"}
    return {
        "inspect": stage("inspect"),
        "segment": stage("segment"),
        "segmentation-qc": stage("segmentation-qc"),
        "bbox": stage("bbox"),
        "bbox-qc": stage("bbox-qc"),
        "prepare-itksnap": stage("prepare-itksnap"),
        "export": stage("export"),
        "launch": stage("launch"),
    }


def write_reference_image(path):
    image = sitk.GetImageFromArray(np.zeros((2, 3, 4), dtype=np.int16))
    image.SetSpacing((1.0, 2.0, 3.0))
    image.SetOrigin((4.0, 5.0, 6.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    geometry = pipeline.summarize_geometry(image)
    return path, geometry


def write_valid_itksnap_package(paths, image_path, geometry):
    for path in paths["itksnap_required"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    paths["itksnap_manifest"].write_text(json.dumps({
        "completion_status": "complete",
        "image_path": pipeline.display_path(image_path),
        "label_mapping": {
            str(value): name for value, name in pipeline.LABEL_MAPPING.items()
        },
        "bbox_convention": "half-open NumPy z,y,x [min:max); shell faces use min and max-1",
        "image_geometry": {
            "size_xyz": list(geometry.size_xyz),
            "spacing_xyz": list(geometry.spacing_xyz),
            "origin_xyz": list(geometry.origin_xyz),
            "direction": list(geometry.direction),
        },
    }))


def run_all_args(
    tmp_path, resume=False, dry_run=True, no_launch=True, launch=False,
    export_rtstruct=False, dicom_dir="dicom", image="image.nii.gz", modality="mr",
    overwrite=False,
):
    return Namespace(
        command="run-all",
        image=image,
        input_type="nifti",
        modality=modality,
        case_id="case",
        dicom_dir=dicom_dir,
        output_root=str(tmp_path / "outputs"),
        device="cpu",
        fast=False,
        padding_mm=[5.0],
        overwrite=overwrite,
        resume=resume,
        no_launch=no_launch,
        launch=launch,
        viewer=None,
        export_rtstruct=export_rtstruct,
        dry_run=dry_run,
    )
