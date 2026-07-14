from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any, Callable

import yaml

from bbox_qc import run_bbox_qc
from bounding_box import compute_bbox_and_crops
from dicom_reference import inspect_reference_series
from medical_image_io import load_medical_image
from osirix_export import export_osirix_package
from project_paths import DEFAULT_OUTPUTS_DIR, display_path
from segmentation_qc import IndividualMask, run_segmentation_qc
from totalseg_lungs import run_lung_segmentation
from version import PIPELINE_VERSION
from viewer_launcher import (
    build_itksnap_launch_plan,
    build_launch_plan,
    check_itksnap_health,
    detect_itksnap,
    launch_plan,
)


STAGE_NAMES = ["inspect", "segment", "segmentation-qc", "bbox", "bbox-qc", "export", "launch"]


def load_config(path: str | Path | None) -> dict[str, Any]:
    path = Path(path or "config/default.yaml")
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def cfg(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CT/MRI lung bbox pipeline v1.0")
    parser.add_argument("--version", action="version", version=PIPELINE_VERSION)
    parser.add_argument("--config", default="config/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect")
    inspect_p.add_argument("--input")
    inspect_p.add_argument("--input-type", choices=["auto", "dicom", "nifti"], default="auto")
    inspect_p.add_argument("--modality", choices=["auto", "ct", "mr"], default="auto")
    inspect_p.add_argument("--dicom-dir")

    segment = sub.add_parser("segment")
    _add_image_case(segment)
    segment.add_argument("--input-type", choices=["auto", "dicom", "nifti"], default="auto")
    segment.add_argument("--output-root")
    segment.add_argument("--device")
    segment.add_argument("--fast", action="store_true", default=None)
    _add_common_runtime(segment)

    seg_qc = sub.add_parser("segmentation-qc")
    _add_qc_inputs(seg_qc)
    seg_qc.add_argument("--individual-mask", action="append", default=[])
    _add_common_runtime(seg_qc)

    bbox = sub.add_parser("bbox")
    _add_image_mask_case(bbox)
    bbox.add_argument("--padding-mm", type=float, nargs="+")
    bbox.add_argument("--padding-voxels", type=int, nargs="+")
    bbox.add_argument("--crop-individual-masks", action="store_true")
    bbox.add_argument("--individual-mask", action="append", default=[])
    bbox.add_argument("--output-root")
    _add_common_runtime(bbox)

    bbox_qc = sub.add_parser("bbox-qc")
    _add_image_mask_case(bbox_qc)
    bbox_qc.add_argument("--bbox-json", required=True)
    bbox_qc.add_argument("--crop-image")
    bbox_qc.add_argument("--crop-mask")
    bbox_qc.add_argument("--output-dir")
    _add_common_runtime(bbox_qc)

    export = sub.add_parser("export")
    _add_export_args(export)
    _add_common_runtime(export)

    launch = sub.add_parser("launch")
    launch.add_argument("--dicom-dir")
    launch.add_argument("--rtstruct")
    launch.add_argument("--fallback-dir")
    launch.add_argument("--nifti", action="append", default=[])
    launch.add_argument("--viewer", default="itksnap")
    launch.add_argument("--image")
    launch.add_argument("--segmentation", action="append", default=[])
    launch.add_argument("--labels")
    launch.add_argument("--overlay", action="append", default=[])
    launch.add_argument("--workspace")
    launch.add_argument("--health-check", action="store_true")
    launch.add_argument("--dry-run", action="store_true")

    run_all = sub.add_parser("run-all")
    _add_image_case(run_all)
    run_all.add_argument("--input-type", choices=["auto", "dicom", "nifti"], default="nifti")
    run_all.add_argument("--dicom-dir")
    run_all.add_argument("--output-root")
    run_all.add_argument("--device")
    run_all.add_argument("--fast", action="store_true", default=None)
    run_all.add_argument("--padding-mm", type=float, nargs="+")
    run_all.add_argument("--resume", action="store_true")
    run_all.add_argument("--no-launch", action="store_true")
    run_all.add_argument("--viewer")
    _add_common_runtime(run_all)
    return parser


def _add_image_case(parser):
    parser.add_argument("--image", required=True)
    parser.add_argument("--modality", choices=["ct", "mr"], required=True)
    parser.add_argument("--case-id", required=True)


def _add_image_mask_case(parser):
    _add_image_case(parser)
    parser.add_argument("--mask", required=True)


def _add_qc_inputs(parser):
    _add_image_mask_case(parser)
    parser.add_argument("--output-dir")


def _add_export_args(parser):
    parser.add_argument("--dicom-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--individual-mask", action="append", default=[])
    parser.add_argument("--bbox-json")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--format", choices=["auto", "rtstruct", "dicom-seg"], default="auto")


def _add_common_runtime(parser):
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        result = dispatch(args, config)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2


def dispatch(args, config: dict[str, Any]) -> dict[str, Any]:
    if args.command == "inspect":
        return command_inspect(args)
    if args.command == "segment":
        return command_segment(args, config)
    if args.command == "segmentation-qc":
        return command_segmentation_qc(args)
    if args.command == "bbox":
        return command_bbox(args, config)
    if args.command == "bbox-qc":
        return command_bbox_qc(args)
    if args.command == "export":
        return command_export(args, config)
    if args.command == "launch":
        return command_launch(args)
    if args.command == "run-all":
        return run_all(args, config)
    raise ValueError(f"Unsupported command: {args.command}")


def command_inspect(args) -> dict[str, Any]:
    result: dict[str, Any] = {"pipeline_version": PIPELINE_VERSION}
    if args.input:
        loaded = load_medical_image(args.input, input_type=args.input_type, modality=args.modality)
        result["image"] = loaded.safe_summary()
    if args.dicom_dir:
        result["dicom_reference"] = inspect_reference_series(args.dicom_dir).safe_summary()
    return result


def command_segment(args, config) -> dict[str, Any]:
    output_root = args.output_root or Path(cfg(config, "output_root", "outputs")) / "totalsegmentator"
    device = args.device or cfg(config, "totalsegmentator.device", "auto")
    fast = cfg(config, "totalsegmentator.fast", False) if args.fast is None else args.fast
    return run_lung_segmentation(
        input_path=args.image,
        input_type=args.input_type,
        modality=args.modality,
        case_id=args.case_id,
        output_root=output_root,
        device=device,
        fast=fast,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def command_segmentation_qc(args) -> dict[str, Any]:
    individuals = [_parse_individual_mask(value) for value in args.individual_mask]
    return run_segmentation_qc(
        image_path=args.image,
        mask_path=args.mask,
        modality=args.modality,
        case_id=args.case_id,
        output_dir=args.output_dir,
        individual_masks=individuals,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def command_bbox(args, config) -> dict[str, Any]:
    output_root = args.output_root or cfg(config, "output_root", "outputs")
    padding_mm = args.padding_mm if args.padding_mm is not None else cfg(config, "bbox.padding_mm", 0)
    return compute_bbox_and_crops(
        image_path=args.image,
        mask_path=args.mask,
        case_id=args.case_id,
        padding_mm=padding_mm,
        padding_voxels=args.padding_voxels,
        individual_mask_paths=args.individual_mask if args.crop_individual_masks else [],
        output_root=output_root,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def command_bbox_qc(args) -> dict[str, Any]:
    return run_bbox_qc(
        image_path=args.image,
        mask_path=args.mask,
        bbox_json_path=args.bbox_json,
        case_id=args.case_id,
        modality=args.modality,
        crop_image_path=args.crop_image,
        crop_mask_path=args.crop_mask,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def command_export(args, config) -> dict[str, Any]:
    output_root = args.output_root or Path(cfg(config, "output_root", "outputs")) / "osirix"
    return export_osirix_package(
        dicom_dir=args.dicom_dir,
        image_path=args.image,
        mask_path=args.mask,
        individual_masks=[_parse_named_path(value) for value in args.individual_mask],
        bbox_json=args.bbox_json,
        case_id=args.case_id,
        output_root=output_root,
        export_format=args.format,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def command_launch(args) -> dict[str, Any]:
    if args.viewer == "itksnap":
        installation = detect_itksnap()
        if getattr(args, "health_check", False):
            checked = check_itksnap_health(installation)
            return {
                "viewer": "itksnap",
                "status": checked.status,
                "bundle": display_path(checked.bundle_path) if checked.bundle_path else None,
                "executable": display_path(checked.executable_path) if checked.executable_path else None,
                "version": checked.short_version,
                "message": checked.message,
            }
        plan = build_itksnap_launch_plan(
            image_path=getattr(args, "image", None),
            segmentation_paths=getattr(args, "segmentation", ()),
            label_description_path=getattr(args, "labels", None),
            overlay_paths=getattr(args, "overlay", ()),
            workspace_path=getattr(args, "workspace", None),
            installation=installation,
        )
        return launch_plan(plan, dry_run=args.dry_run)
    plan = build_launch_plan(
        dicom_dir=args.dicom_dir,
        rtstruct_path=args.rtstruct,
        fallback_dir=args.fallback_dir,
        nifti_paths=args.nifti,
        viewer=args.viewer,
    )
    return launch_plan(plan, dry_run=args.dry_run)


def run_all(args, config: dict[str, Any], stage_functions: dict[str, Callable] | None = None) -> dict[str, Any]:
    funcs = stage_functions or _default_stage_functions()
    output_root = Path(args.output_root or cfg(config, "output_root", "outputs"))
    paths = _run_all_paths(output_root, args.case_id, args.modality)
    completed: list[str] = []
    timings: dict[str, float] = {}
    warnings: list[str] = []
    start_total = time.perf_counter()

    def run_stage(index: int, name: str, exists_path: Path | list[Path] | None, call: Callable[[], Any]):
        print(f"[{index}/7] {name}")
        stage_start = time.perf_counter()
        if args.resume and _complete(exists_path):
            completed.append(name)
            timings[name] = 0.0
            print(f"[{index}/7] {name}: reused")
            return {"status": "reused"}
        result = call()
        completed.append(name)
        timings[name] = time.perf_counter() - stage_start
        print(f"[{index}/7] {name}: ok ({timings[name]:.1f}s)")
        return result

    results: dict[str, Any] = {"pipeline_version": PIPELINE_VERSION, "case_id": args.case_id}
    try:
        results["inspect"] = run_stage(1, "inspect", None, lambda: funcs["inspect"](args))
        results["segment"] = run_stage(
            2,
            "segment",
            [paths["segment_manifest"], paths["combined_mask"]],
            lambda: funcs["segment"](args, config),
        )
        seg_qc_args = _namespace_from(args, mask=str(paths["combined_mask"]), output_dir=str(paths["figures_dir"]))
        results["segmentation-qc"] = run_stage(
            3,
            "segmentation-qc",
            paths["seg_qc_json"],
            lambda: funcs["segmentation-qc"](seg_qc_args),
        )
        bbox_args = _namespace_from(
            args,
            mask=str(paths["combined_mask"]),
            output_root=str(output_root),
            padding_mm=args.padding_mm if args.padding_mm is not None else cfg(config, "bbox.padding_mm", 0),
            padding_voxels=None,
            crop_individual_masks=True,
            individual_mask=[str(p) for p in paths["individual_masks"] if p.exists()],
        )
        results["bbox"] = run_stage(
            4,
            "bbox",
            [paths["bbox_json"], paths["image_crop"], paths["combined_crop"]],
            lambda: funcs["bbox"](bbox_args, config),
        )
        bbox_qc_args = _namespace_from(
            args,
            mask=str(paths["combined_mask"]),
            bbox_json=str(paths["bbox_json"]),
            crop_image=str(paths["image_crop"]),
            crop_mask=str(paths["combined_crop"]),
            output_dir=str(paths["figures_dir"]),
        )
        results["bbox-qc"] = run_stage(5, "bbox-qc", paths["bbox_qc_json"], lambda: funcs["bbox-qc"](bbox_qc_args))
        if args.dicom_dir:
            export_args = _namespace_from(
                args,
                dicom_dir=args.dicom_dir,
                mask=str(paths["combined_mask"]),
                bbox_json=str(paths["bbox_json"]),
                output_root=str(output_root / "osirix"),
                format="auto",
                individual_mask=[f"{p.stem}={p}" for p in paths["individual_masks"] if p.exists()],
            )
            results["export"] = run_stage(6, "export", paths["export_report"], lambda: funcs["export"](export_args, config))
        if not args.no_launch and not args.dry_run and cfg(config, "launch.enabled", False):
            launch_args = _namespace_from(
                args,
                rtstruct=str(paths["rtstruct"]) if paths["rtstruct"].exists() else None,
                fallback_dir=str(paths["fallback_dir"]) if paths["fallback_dir"].exists() else None,
                nifti=[args.image, str(paths["combined_mask"])],
            )
            results["launch"] = run_stage(7, "launch", None, lambda: funcs["launch"](launch_args))
    except Exception as exc:
        log = _write_run_log(output_root, args.case_id, args.modality, completed, timings, warnings, "failed", str(exc))
        raise RuntimeError(f"run-all failed: {exc}. Log: {display_path(log)}") from exc

    total_runtime = time.perf_counter() - start_total
    log = _write_run_log(output_root, args.case_id, args.modality, completed, timings, warnings, "success", None)
    results["completed_stages"] = completed
    results["stage_timings_seconds"] = timings
    results["total_runtime_seconds"] = total_runtime
    results["run_log"] = display_path(log)
    return results


def _complete(path_or_paths: Path | list[Path] | None) -> bool:
    if path_or_paths is None:
        return False
    if isinstance(path_or_paths, list):
        return all(path.exists() for path in path_or_paths)
    return path_or_paths.exists()


def _default_stage_functions() -> dict[str, Callable]:
    return {
        "inspect": lambda args: command_inspect(_namespace_from(args, input=args.image, dicom_dir=args.dicom_dir)),
        "segment": command_segment,
        "segmentation-qc": command_segmentation_qc,
        "bbox": command_bbox,
        "bbox-qc": command_bbox_qc,
        "export": command_export,
        "launch": command_launch,
    }


def _run_all_paths(output_root: Path, case_id: str, modality: str) -> dict[str, Any]:
    ts = output_root / "totalsegmentator" / case_id
    lungs = ts / "lungs"
    individual_names = ["lung_left", "lung_right"] if modality == "mr" else [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ]
    return {
        "segment_manifest": ts / "manifest.json",
        "combined_mask": lungs / "combined_lungs.nii.gz",
        "individual_masks": [lungs / f"{name}.nii.gz" for name in individual_names],
        "figures_dir": output_root / "figures" / case_id,
        "seg_qc_json": output_root / "figures" / case_id / "segmentation_qc.json",
        "bbox_json": output_root / "bbox" / case_id / "bbox.json",
        "image_crop": output_root / "crops" / case_id / "image_crop.nii.gz",
        "combined_crop": output_root / "crops" / case_id / "combined_lungs_crop.nii.gz",
        "bbox_qc_json": output_root / "figures" / case_id / "bounding_box_qc.json",
        "export_report": output_root / "osirix" / case_id / "export_report.json",
        "rtstruct": output_root / "osirix" / case_id / "lung_segmentation_and_bbox_rtstruct.dcm",
        "fallback_dir": output_root / "osirix" / case_id / "review_package",
    }


def _write_run_log(output_root: Path, case_id: str, modality: str, completed, timings, warnings, status, error):
    log_dir = output_root / "logs" / case_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline_run.json"
    versions = {}
    for package in ["numpy", "SimpleITK", "pydicom", "TotalSegmentator", "PyYAML"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    log_path.write_text(json.dumps({
        "pipeline_version": PIPELINE_VERSION,
        "case_id": case_id,
        "modality": modality,
        "completed_stages": completed,
        "stage_timings_seconds": timings,
        "software_versions": versions,
        "warnings": warnings,
        "errors": [error] if error else [],
        "completion_status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    return log_path


def _parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("Expected name=path.")
    name, path = value.split("=", 1)
    return name, path


def _parse_individual_mask(value: str) -> IndividualMask:
    name, path = _parse_named_path(value)
    return IndividualMask(label=name, path=Path(path))


def _namespace_from(source, **updates):
    data = vars(source).copy()
    data.update(updates)
    return argparse.Namespace(**data)


if __name__ == "__main__":
    raise SystemExit(main())
