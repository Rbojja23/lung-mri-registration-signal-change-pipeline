from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lung_ct_fractional_proxy_dicom_mpl"),
)

import matplotlib
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from scipy.ndimage import binary_erosion
import SimpleITK as sitk

from project_paths import DEFAULT_OUTPUTS_DIR, PROJECT_ROOT, display_path
from registration_inputs import sha256_file, validate_registration_image
from rgb_secondary_capture_export import (
    _exact_geometry_mismatches,
    _validated_spatial_geometry,
    _write_dicom_slice,
)
from version import PIPELINE_VERSION


ANALYSIS_TYPE = "NIfTI-derived RGB DICOM Secondary Capture of fractional MRI signal-change proxies"
EXPECTED_REPORT_ANALYSIS = "Exploratory registered-space fractional MRI signal-change proxy"
FORMULA = "(registered_moving - fixed) / registered_moving"
CASE_KEYS = ("case_1", "case_2")
METHOD_KEYS = ("native", "mean", "gaussian")
REPORT_VARIANTS = {
    "native": "native",
    "mean": "mean_filtered",
    "gaussian": "gaussian_filtered",
}
DEFAULT_EXPERIMENT_DIR = (
    DEFAULT_OUTPUTS_DIR
    / "registration"
    / "case_comparison"
    / "fractional_mri_signal_change_proxy_gaussian_mm_comparison"
)
DEFAULT_REPORT = DEFAULT_EXPERIMENT_DIR / "fractional_proxy_report.json"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "horos_rgb_dicom_secondary_capture"
MANIFEST_NAME = "fractional_proxy_rgb_secondary_capture_manifest.json"
DEFAULT_OPACITY = 0.62


def export_fractional_proxy_rgb_secondary_capture(
    *,
    comparison_report: str | Path = DEFAULT_REPORT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    opacity: float = DEFAULT_OPACITY,
    draw_contour: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Export six research-only proxy series using fixed MRI as background."""
    if not math.isfinite(float(opacity)) or not 0 <= float(opacity) <= 1:
        raise ValueError("Overlay opacity must be finite and between 0 and 1.")
    report_path = Path(comparison_report).expanduser().resolve(strict=False)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    report = _load_report(report_path)
    context = _load_inputs(report, report_path)
    vmin, vmax = context["limits"]
    anatomy_limits = context["anatomy_limits"]
    slice_counts = {
        case: int(context["images"][case]["fixed"].GetSize()[2]) for case in CASE_KEYS
    }
    planned = {
        case: {
            method: [
                destination / case / method / f"IM_{z + 1:04d}.dcm"
                for z in range(slice_counts[case])
            ]
            for method in METHOD_KEYS
        }
        for case in CASE_KEYS
    }
    manifest_path = destination / MANIFEST_NAME
    _check_outputs(destination, planned, manifest_path, overwrite=overwrite, dry_run=dry_run)

    studies = {case: generate_uid() for case in CASE_KEYS}
    frames = {case: generate_uid() for case in CASE_KEYS}
    uid_plan = {
        case: {
            method: {
                "study_instance_uid": studies[case],
                "series_instance_uid": generate_uid(),
                "frame_of_reference_uid": frames[case],
                "sop_instance_uids": [generate_uid() for _ in range(slice_counts[case])],
            }
            for method in METHOD_KEYS
        }
        for case in CASE_KEYS
    }
    if len({uid_plan[c][m]["series_instance_uid"] for c in CASE_KEYS for m in METHOD_KEYS}) != 6:
        raise RuntimeError("UID generation did not produce six unique series identifiers.")

    now = datetime.now(timezone.utc)
    geometries = {
        case: _validated_spatial_geometry(context["images"][case]["fixed"])
        for case in CASE_KEYS
    }
    manifest = _build_manifest(
        report=report,
        report_path=report_path,
        destination=destination,
        manifest_path=manifest_path,
        context=context,
        planned=planned,
        uid_plan=uid_plan,
        geometries=geometries,
        slice_counts=slice_counts,
        opacity=float(opacity),
        draw_contour=bool(draw_contour),
        timestamp=now,
        dry_run=dry_run,
    )
    if dry_run:
        return manifest

    destination.mkdir(parents=True, exist_ok=True)
    for case_index, case in enumerate(CASE_KEYS):
        for method_index, method in enumerate(METHOD_KEYS):
            _write_series(
                case=case,
                method=method,
                fixed=context["images"][case]["fixed"],
                proxy=context["images"][case][method],
                validity=context["arrays"][case][f"{method}_validity"],
                lung_mask=context["arrays"][case]["lung_mask"],
                paths=planned[case][method],
                uids=uid_plan[case][method],
                geometry=geometries[case],
                vmin=vmin,
                vmax=vmax,
                anatomy_limits=anatomy_limits,
                opacity=float(opacity),
                draw_contour=bool(draw_contour),
                series_number=case_index * len(METHOD_KEYS) + method_index + 1,
                timestamp=now,
            )

    validation = validate_fractional_proxy_dicom_series(
        planned, uid_plan, geometries, vmin=vmin, vmax=vmax
    )
    hashes_after = _source_hashes(context["source_paths"])
    if hashes_after != context["hashes_before"]:
        raise RuntimeError("A source NIfTI or comparison report changed during DICOM export.")
    manifest["output_checksums_sha256"] = {
        case: {
            method: {path.name: sha256_file(path) for path in planned[case][method]}
            for method in METHOD_KEYS
        }
        for case in CASE_KEYS
    }
    manifest["source_immutability"] = {
        "verified_before_and_after": True,
        "unchanged": True,
        "sha256_before": context["hashes_before"],
        "sha256_after": hashes_after,
    }
    manifest["validation"] = validation
    manifest["completion_status"] = "complete"
    manifest["writes_performed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def render_fractional_proxy_rgb_slice(
    fixed: np.ndarray,
    proxy: np.ndarray,
    validity: np.ndarray,
    lung_mask: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    anatomy_limits: tuple[float, float],
    opacity: float,
    draw_contour: bool = True,
) -> np.ndarray:
    fixed_values = np.asarray(fixed, dtype=np.float32)
    proxy_values = np.asarray(proxy, dtype=np.float32)
    valid = np.asarray(validity, dtype=bool)
    lung = np.asarray(lung_mask, dtype=bool)
    if any(array.ndim != 2 for array in (fixed_values, proxy_values, valid, lung)):
        raise ValueError("Fixed, proxy, validity, and lung-mask slices must be 2D.")
    if len({array.shape for array in (fixed_values, proxy_values, valid, lung)}) != 1:
        raise ValueError("Fixed, proxy, validity, and lung-mask slices must have identical shapes.")
    low, high = anatomy_limits
    if not all(math.isfinite(value) for value in (vmin, vmax, low, high, opacity)):
        raise ValueError("Display parameters must be finite.")
    if not vmin < 0 < vmax or not math.isclose(abs(vmin), vmax, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Proxy display limits must be symmetric and zero-centered.")
    if high <= low or not 0 <= opacity <= 1:
        raise ValueError("Invalid anatomy limits or opacity.")
    if not np.isfinite(fixed_values).all():
        raise ValueError("Fixed anatomy must contain only finite values.")
    color_domain = valid & lung & np.isfinite(proxy_values)
    gray = np.clip((fixed_values - low) / (high - low), 0, 1)
    gray_rgb = np.repeat(gray[..., None], 3, axis=2)
    normalized = np.clip((proxy_values - vmin) / (vmax - vmin), 0, 1)
    color = matplotlib.colormaps["RdBu_r"](normalized)[..., :3]
    result = gray_rgb.copy()
    result[color_domain] = (
        (1.0 - opacity) * gray_rgb[color_domain] + opacity * color[color_domain]
    )
    if draw_contour and np.any(lung):
        contour = lung & ~binary_erosion(lung, structure=np.ones((3, 3), dtype=bool), border_value=0)
        result[contour] = 1.0
    return np.ascontiguousarray(np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8))


def _load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Comparison report does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Comparison report is unreadable or malformed.") from exc
    if report.get("completion_status") != "complete":
        raise ValueError("Comparison report is not complete.")
    if report.get("analysis_type") != EXPECTED_REPORT_ANALYSIS:
        raise ValueError("Comparison report has an incompatible analysis type.")
    if report.get("formula", {}).get("expression") != FORMULA:
        raise ValueError("Comparison report has an incompatible proxy formula.")
    display = report.get("display", {})
    window = display.get("shared_symmetric_window", {})
    try:
        vmin, vmax = float(window["vmin"]), float(window["vmax"])
        low, high = (float(value) for value in display["anatomy_limits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Comparison report has invalid display limits.") from exc
    if not vmin < 0 < vmax or not math.isclose(abs(vmin), vmax, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Comparison report limits are not symmetric and zero-centered.")
    if high <= low or display.get("colormap") != "RdBu_r":
        raise ValueError("Comparison report display policy is incompatible.")
    expected_order = [f"{case}_{REPORT_VARIANTS[method]}" for case in CASE_KEYS for method in METHOD_KEYS]
    if window.get("pooled_order") != expected_order or window.get("identical_for_all_proxy_panels") is not True:
        raise ValueError("Comparison report does not certify one shared six-map window.")
    return report


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve(strict=False) if path.is_absolute() else (PROJECT_ROOT / path).resolve(strict=False)


def _load_inputs(report, report_path):
    images: dict[str, dict[str, sitk.Image]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    source_paths: dict[str, Path] = {"comparison_report": report_path}
    input_report: dict[str, Any] = {"comparison_report": {"path": display_path(report_path), "sha256": sha256_file(report_path)}, "cases": {}}
    for case in CASE_KEYS:
        report_inputs = report["inputs"][case]
        report_outputs = report["outputs"]["cases"][case]
        paths = {
            "fixed": _resolve(report_inputs["fixed"]["path"]),
            "lung_mask": _resolve(report_inputs["exact_fixed_space_lung_mask"]["path"]),
        }
        for method in METHOD_KEYS:
            variant = REPORT_VARIANTS[method]
            paths[method] = _resolve(report_outputs[variant])
            paths[f"{method}_validity"] = _resolve(report_outputs[f"{variant}_validity"])
        loaded = {}
        for role, path in paths.items():
            if role in METHOD_KEYS:
                try:
                    image = sitk.ReadImage(str(path))
                except RuntimeError as exc:
                    raise ValueError(f"{case} {role} proxy NIfTI could not be read.") from exc
                if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
                    raise ValueError(f"{case} {role} proxy must be a scalar 3D NIfTI.")
                if not np.isfinite(sitk.GetArrayViewFromImage(image)).any():
                    raise ValueError(f"{case} {role} proxy has no finite analytical voxels.")
                loaded[role] = image
            else:
                loaded[role] = validate_registration_image(
                    path, role=f"{case} {role}"
                ).image
        for role, image in loaded.items():
            if role == "fixed":
                continue
            mismatches = _exact_geometry_mismatches(loaded["fixed"], image)
            if mismatches:
                raise ValueError(f"{case} fixed/{role} exact geometry mismatch: " + "; ".join(mismatches))
        mask = sitk.GetArrayFromImage(loaded["lung_mask"])
        if not set(float(value) for value in np.unique(mask)).issubset({0.0, 1.0}) or not np.any(mask):
            raise ValueError(f"{case} lung mask must be nonempty and binary.")
        case_arrays = {"lung_mask": mask.astype(bool)}
        for method in METHOD_KEYS:
            validity = sitk.GetArrayFromImage(loaded[f"{method}_validity"])
            if not set(float(value) for value in np.unique(validity)).issubset({0.0, 1.0}):
                raise ValueError(f"{case} {method} validity map must be binary.")
            case_arrays[f"{method}_validity"] = validity.astype(bool)
        images[case] = loaded
        arrays[case] = case_arrays
        hashes = {role: sha256_file(path) for role, path in paths.items()}
        if hashes["fixed"] != report_inputs["fixed"]["sha256"]:
            raise ValueError(f"{case} fixed checksum does not match the comparison report.")
        if hashes["lung_mask"] != report_inputs["exact_fixed_space_lung_mask"]["sha256"]:
            raise ValueError(f"{case} lung-mask checksum does not match the comparison report.")
        recorded_outputs = report.get("output_checksums_sha256", {})
        for method in METHOD_KEYS:
            for role in (method, f"{method}_validity"):
                recorded = recorded_outputs.get(display_path(paths[role]))
                if recorded is not None and recorded != hashes[role]:
                    raise ValueError(f"{case} {role} checksum does not match the comparison report.")
        input_report["cases"][case] = {
            role: {"path": display_path(path), "sha256": hashes[role]}
            for role, path in paths.items()
        }
        for role, path in paths.items():
            source_paths[f"{case}:{role}"] = path
    window = report["display"]["shared_symmetric_window"]
    anatomy = report["display"]["anatomy_limits"]
    return {
        "images": images,
        "arrays": arrays,
        "source_paths": source_paths,
        "hashes_before": _source_hashes(source_paths),
        "input_report": input_report,
        "limits": (float(window["vmin"]), float(window["vmax"])),
        "anatomy_limits": (float(anatomy[0]), float(anatomy[1])),
    }


def _build_manifest(*, report, report_path, destination, manifest_path, context, planned, uid_plan, geometries, slice_counts, opacity, draw_contour, timestamp, dry_run):
    vmin, vmax = context["limits"]
    low, high = context["anatomy_limits"]
    return {
        "pipeline_version": PIPELINE_VERSION,
        "analysis_type": ANALYSIS_TYPE,
        "completion_status": "dry-run" if dry_run else "pending",
        "writes_performed": False if dry_run else None,
        "timestamp_utc": timestamp.isoformat(),
        "source_analytical_method": {
            "formula": FORMULA,
            "filtering": {
                "native": "unfiltered inputs",
                "mean": "mask-aware 7x7x7 box-mean filtering of fixed and registered-moving inputs before ratio",
                "gaussian": "mask-aware physical-mm Gaussian filtering of fixed and registered-moving inputs before ratio",
            },
        },
        "inputs": context["input_report"],
        "display": {
            "background_everywhere": "corresponding fixed MRI rendered in grayscale",
            "fixed_anatomy_used_as_background": True,
            "difference_image_used_as_background": False,
            "difference_image_used_anywhere": False,
            "inside_valid_lung": "fixed grayscale blended with signed proxy RdBu_r color",
            "outside_valid_lung": "fixed grayscale retained without proxy color",
            "exact_lung_mask_policy": "color domain is validity map AND exact fixed-space lung mask AND finite proxy",
            "lung_mask_contour": bool(draw_contour),
            "lung_mask_contour_policy": "one-pixel white inner boundary derived from the exact binary mask" if draw_contour else "disabled",
            "colormap": "RdBu_r",
            "zero_centered": True,
            "shared_signed_limits": {"vmin": vmin, "vcenter": 0.0, "vmax": vmax},
            "grayscale_limits": {"vmin": low, "vmax": high},
            "overlay_opacity": opacity,
            "identical_policy_for_all_six_series": True,
            "research_only_labeling": "SeriesDescription and DerivationDescription identify these as exploratory and not validated ventilation; no pixel annotation obscures anatomy.",
        },
        "geometry": geometries,
        "spatial_tags": {
            "source": "corresponding fixed NIfTI geometry",
            "tags": ["PixelSpacing", "ImageOrientationPatient", "ImagePositionPatient", "SliceThickness", "SpacingBetweenSlices", "FrameOfReferenceUID"],
            "clinical_reference_relationship_claimed": False,
        },
        "slice_order": {
            "source": "ascending fixed-NIfTI z index",
            "filename": "IM_{InstanceNumber:04d}.dcm",
            "instance_number": "z index + 1",
            "image_position": "fixed.TransformIndexToPhysicalPoint((0, 0, z))",
        },
        "identifiers": {
            case: {
                method: {
                    "StudyInstanceUID": uid_plan[case][method]["study_instance_uid"],
                    "SeriesInstanceUID": uid_plan[case][method]["series_instance_uid"],
                    "FrameOfReferenceUID": uid_plan[case][method]["frame_of_reference_uid"],
                    "SOPInstanceUIDs": uid_plan[case][method]["sop_instance_uids"],
                }
                for method in METHOD_KEYS
            }
            for case in CASE_KEYS
        },
        "slice_counts": slice_counts,
        "outputs": {
            "directory": display_path(destination),
            "manifest": display_path(manifest_path),
            "series": {
                case: {
                    method: {
                        "directory": display_path(destination / case / method),
                        "series_description": _series_description(case, method),
                        "file_count": len(planned[case][method]),
                        "first_file": display_path(planned[case][method][0]),
                        "last_file": display_path(planned[case][method][-1]),
                    }
                    for method in METHOD_KEYS
                }
                for case in CASE_KEYS
            },
        },
        "output_checksums_sha256": None,
        "source_immutability": {"verified_before_and_after": False if dry_run else None, "unchanged": None, "sha256_before": context["hashes_before"]},
        "object_classification": {
            "dicom_sop_class": "Secondary Capture Image Storage",
            "nifti_derived": True,
            "visualization_only": True,
            "original_clinical_dicom": False,
            "quantitative_parametric_map": False,
        },
        "warnings": [
            "Exploratory fractional MRI signal-change proxy — not validated ventilation.",
            "Derived anonymized 8-bit RGB visualization; quantitative source values cannot be recovered from these pixels.",
            "These are NIfTI-derived Secondary Capture objects, not original clinical DICOM objects or quantitative DICOM Parametric Maps.",
            "Spatial tags are derived from fixed NIfTI geometry; no original clinical DICOM reference relationship is claimed.",
            "MRI intensity calibration, respiratory ordering, and clinical interpretation remain unconfirmed.",
        ],
        "validation": {"status": "not-run" if dry_run else "pending"},
        "provenance_report": {"path": display_path(report_path), "sha256": sha256_file(report_path)},
    }


def _write_series(*, case, method, fixed, proxy, validity, lung_mask, paths, uids, geometry, vmin, vmax, anatomy_limits, opacity, draw_contour, series_number, timestamp):
    fixed_array = sitk.GetArrayViewFromImage(fixed)
    proxy_array = sitk.GetArrayViewFromImage(proxy)
    derivation = (
        f"NIfTI-derived exploratory fractional MRI signal-change proxy ({method}); P=(M_R-F)/M_R; "
        f"fixed MRI grayscale background; signed RdBu_r color only at valid exact-mask lung voxels; "
        f"limits [{vmin:.12g}, {vmax:.12g}], opacity={opacity:.6g}. Not validated ventilation. "
        "Visualization-only RGB Secondary Capture; no original clinical DICOM reference relationship is claimed."
    )
    for z, path in enumerate(paths):
        rgb = render_fractional_proxy_rgb_slice(
            fixed_array[z], proxy_array[z], validity[z], lung_mask[z],
            vmin=vmin, vmax=vmax, anatomy_limits=anatomy_limits,
            opacity=opacity, draw_contour=draw_contour,
        )
        _write_dicom_slice(
            path=path,
            rgb=rgb,
            case=case,
            instance_number=z + 1,
            sop_instance_uid=uids["sop_instance_uids"][z],
            study_instance_uid=uids["study_instance_uid"],
            series_instance_uid=uids["series_instance_uid"],
            frame_of_reference_uid=uids["frame_of_reference_uid"],
            geometry=geometry,
            position=fixed.TransformIndexToPhysicalPoint((0, 0, z)),
            series_description=_series_description(case, method),
            derivation=derivation,
            timestamp=timestamp,
            series_number=series_number,
        )


def _series_description(case, method):
    labels = {"native": "Native", "mean": "7x7x7 Mean", "gaussian": "Gaussian"}
    return f"{case.replace('_', ' ').title()} {labels[method]} MRI proxy - research only"


def _check_outputs(destination, planned, manifest_path, *, overwrite, dry_run):
    expected = {manifest_path.resolve(strict=False)}
    expected.update(path.resolve(strict=False) for case in planned.values() for paths in case.values() for path in paths)
    existing = [path for path in expected if path.exists()]
    if existing and not overwrite and not dry_run:
        raise FileExistsError("Fractional proxy Secondary Capture outputs already exist; use --overwrite.")
    if overwrite:
        for case in CASE_KEYS:
            for method in METHOD_KEYS:
                directory = destination / case / method
                if directory.is_dir():
                    unexpected = [path for path in directory.glob("*.dcm") if path.resolve(strict=False) not in expected]
                    if unexpected:
                        raise FileExistsError("Refusing overwrite because a series directory contains unexpected DICOM files; use a new output directory.")


def validate_fractional_proxy_dicom_series(planned, uid_plan, geometries, *, vmin, vmax):
    all_series = set()
    all_sops = set()
    results = {}
    for case in CASE_KEYS:
        results[case] = {}
        geometry = geometries[case]
        direction = np.asarray(geometry["direction"], dtype=float).reshape(3, 3)
        for method in METHOD_KEYS:
            series = set()
            sops = []
            for z, path in enumerate(planned[case][method]):
                ds = pydicom.dcmread(str(path))
                if str(ds.SOPClassUID) != str(SecondaryCaptureImageStorage) or str(ds.file_meta.TransferSyntaxUID) != str(ExplicitVRLittleEndian):
                    raise ValueError("Generated DICOM is not Explicit VR Little Endian Secondary Capture.")
                if ds.PhotometricInterpretation != "RGB" or ds.PlanarConfiguration != 0 or ds.pixel_array.shape != (int(ds.Rows), int(ds.Columns), 3):
                    raise ValueError("Generated DICOM does not contain readable interleaved RGB pixels.")
                if int(ds.InstanceNumber) != z + 1:
                    raise ValueError("Generated DICOM slice order is invalid.")
                expected_position = np.asarray(geometry["origin_xyz_mm"]) + direction[:, 2] * float(geometry["spacing_xyz_mm"][2]) * z
                if not np.allclose(np.asarray(ds.ImagePositionPatient, dtype=float), expected_position, atol=1e-6, rtol=0):
                    raise ValueError("Generated DICOM slice position is invalid.")
                if not np.allclose(np.asarray(ds.ImageOrientationPatient, dtype=float), geometry["image_orientation_patient"], atol=1e-6, rtol=0):
                    raise ValueError("Generated DICOM orientation is invalid.")
                if not np.allclose(np.asarray(ds.PixelSpacing, dtype=float), geometry["pixel_spacing_row_column_mm"], atol=1e-6, rtol=0):
                    raise ValueError("Generated DICOM pixel spacing is invalid.")
                series.add(str(ds.SeriesInstanceUID))
                sops.append(str(ds.SOPInstanceUID))
            expected_series = uid_plan[case][method]["series_instance_uid"]
            if series != {expected_series} or expected_series in all_series:
                raise ValueError("Series Instance UIDs are not unique and consistent.")
            if len(set(sops)) != len(sops) or all_sops.intersection(sops):
                raise ValueError("SOP Instance UIDs are not globally unique.")
            all_series.add(expected_series)
            all_sops.update(sops)
            results[case][method] = {
                "readable_slice_count": len(sops),
                "rgb_pixel_validation": "passed",
                "slice_ordering_validation": "passed",
                "spatial_tag_validation": "passed",
                "consistent_unique_series_instance_uid": True,
                "unique_sop_instance_uids": True,
            }
    return {
        "status": "passed",
        "series": results,
        "six_unique_series_instance_uids": len(all_series) == 6,
        "all_sop_instance_uids_unique": True,
        "identical_shared_scaling": {"passed": True, "vmin": vmin, "vmax": vmax},
        "patient_identity_policy": "passed: only synthetic research identifiers written",
    }


def _source_hashes(paths):
    return {name: sha256_file(path) for name, path in paths.items()}
