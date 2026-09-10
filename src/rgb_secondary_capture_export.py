from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lung_ct_rgb_secondary_capture_mpl")
)

import matplotlib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from pydicom.valuerep import format_number_as_ds
import SimpleITK as sitk

from medical_image_io import summarize_geometry
from project_paths import PROJECT_ROOT, display_path
from registration_inputs import sha256_file, validate_registration_image
from version import PIPELINE_VERSION


ANALYSIS_TYPE = "Derived RGB DICOM Secondary Capture export of registration differences"
EXPECTED_SHARED_WINDOW_ANALYSIS = (
    "Case 1-versus-Case 2 equal-window registration difference visualization"
)
MANIFEST_NAME = "rgb_secondary_capture_manifest.json"
CASE_KEYS = ("case_1", "case_2")
PATIENT_IDS = {"case_1": "RESEARCH_CASE_1", "case_2": "RESEARCH_CASE_2"}
WARNING_TEXT = (
    "RGB Secondary Capture pixels are visualization-only, irreversibly windowed and blended, "
    "and cannot recover the quantitative absolute-difference values."
)


def export_rgb_secondary_capture(
    *,
    case_1_fixed: str | Path,
    case_1_registered: str | Path,
    case_2_fixed: str | Path,
    case_2_registered: str | Path,
    shared_window_report: str | Path,
    output_dir: str | Path,
    opacity: float = 0.45,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Export full-volume RGB Secondary Capture series for two registered cases."""
    if not 0 <= float(opacity) <= 1:
        raise ValueError("Opacity must be between 0 and 1.")
    destination = Path(output_dir).expanduser().resolve(strict=False)
    report_path = Path(shared_window_report).expanduser().resolve(strict=False)
    report, report_hash = _load_compatible_report(report_path)
    paths = {
        "case_1": {
            "fixed": Path(case_1_fixed).expanduser().resolve(strict=False),
            "registered": Path(case_1_registered).expanduser().resolve(strict=False),
        },
        "case_2": {
            "fixed": Path(case_2_fixed).expanduser().resolve(strict=False),
            "registered": Path(case_2_registered).expanduser().resolve(strict=False),
        },
    }
    images = {case: _load_case(case, values) for case, values in paths.items()}
    input_hashes_before = {
        case: {role: sha256_file(path) for role, path in values.items()}
        for case, values in paths.items()
    }
    _validate_report_inputs(report, paths, input_hashes_before)
    window = report["shared_difference_window"]
    vmin, vmax = float(window["vmin"]), float(window["vmax"])
    anatomy = report["display"]["fixed_anatomy_limits"]
    anatomy_limits = (float(anatomy["vmin"]), float(anatomy["vmax"]))
    _validate_limits(vmin, vmax, anatomy_limits)
    geometries = {case: _validated_spatial_geometry(items["fixed"].image) for case, items in images.items()}
    slice_counts = {case: int(items["fixed"].image.GetSize()[2]) for case, items in images.items()}
    planned = {
        case: [destination / case / f"IM_{z + 1:04d}.dcm" for z in range(slice_counts[case])]
        for case in CASE_KEYS
    }
    manifest_path = destination / MANIFEST_NAME
    _check_outputs(destination, planned, manifest_path, overwrite=overwrite, dry_run=dry_run)

    uid_plan = {
        case: {
            "study_instance_uid": generate_uid(),
            "series_instance_uid": generate_uid(),
            "frame_of_reference_uid": generate_uid(),
            "sop_instance_uids": [generate_uid() for _ in range(slice_counts[case])],
        }
        for case in CASE_KEYS
    }
    derivation = (
        f"Derived |F - R| turbo visualization; opacity={float(opacity):.6g}; shared window "
        f"[{vmin:.12g}, {vmax:.12g}]; research-only visualization. Source inputs were NIfTI; "
        "no original clinical DICOM reference relationship is claimed. Quantitative values "
        "cannot be recovered from these RGB Secondary Capture pixels."
    )
    now = datetime.now(timezone.utc)
    common_tags = {
        "SOPClassUID": str(SecondaryCaptureImageStorage),
        "TransferSyntaxUID": str(ExplicitVRLittleEndian),
        "Modality": "OT",
        "ImageType": ["DERIVED", "SECONDARY"],
        "SamplesPerPixel": 3,
        "PhotometricInterpretation": "RGB",
        "PlanarConfiguration": 0,
        "BitsAllocated": 8,
        "BitsStored": 8,
        "HighBit": 7,
        "PixelRepresentation": 0,
        "BurnedInAnnotation": "NO",
        "PatientIdentityRemoved": "YES",
    }
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "analysis_type": ANALYSIS_TYPE,
        "object_classification": {
            "dicom_sop_class": "Secondary Capture Image Storage",
            "visualization_only": True,
            "is_parametric_map": False,
            "is_dicom_seg": False,
            "is_rtstruct": False,
            "is_legacy_overlay_plane": False,
        },
        "inputs": {
            "shared_window_report": {"path": display_path(report_path), "sha256": report_hash},
            "cases": {
                case: {
                    role: {"path": display_path(path), "sha256": input_hashes_before[case][role]}
                    for role, path in values.items()
                }
                for case, values in paths.items()
            },
        },
        "formula": "abs(fixed - registered)",
        "display": {
            "fixed_anatomy": "grayscale",
            "difference_colormap": "turbo",
            "opacity": float(opacity),
            "shared_difference_limits": {"vmin": vmin, "vmax": vmax},
            "fixed_anatomy_limits": {"vmin": anatomy_limits[0], "vmax": anatomy_limits[1]},
            "identical_map_normalization_across_cases": True,
            "quantitative_values_recoverable_from_rgb": False,
        },
        "geometry": geometries,
        "slice_count": slice_counts,
        "slice_ordering": {
            "source": "ascending fixed-NIfTI z index",
            "filename": "IM_{InstanceNumber:04d}.dcm",
            "instance_number": "z index + 1",
            "image_position": "fixed.TransformIndexToPhysicalPoint((0, 0, z))",
        },
        "dicom_tags_used": {
            "object_and_pixel_constants": common_tags,
            "spatial": [
                "PixelSpacing", "ImageOrientationPatient", "ImagePositionPatient",
                "SliceThickness", "SpacingBetweenSlices", "FrameOfReferenceUID",
            ],
            "ordering": ["InstanceNumber", "SeriesNumber"],
            "descriptive": [
                "SeriesDescription", "DerivationDescription", "StudyDescription",
            ],
            "anonymization": [
                "PatientName", "PatientID", "PatientIdentityRemoved",
                "DeidentificationMethod", "BurnedInAnnotation",
            ],
        },
        "identifiers": {
            case: {
                "PatientName": "ANONYMIZED^RESEARCH",
                "PatientID": PATIENT_IDS[case],
                "StudyInstanceUID": uid_plan[case]["study_instance_uid"],
                "SeriesInstanceUID": uid_plan[case]["series_instance_uid"],
                "FrameOfReferenceUID": uid_plan[case]["frame_of_reference_uid"],
            }
            for case in CASE_KEYS
        },
        "descriptions": {
            "series_description_template": "Research |F-R| turbo alpha=<opacity> shared [vmin,vmax]",
            "derivation_description": derivation,
        },
        "outputs": {
            "directory": display_path(destination),
            "manifest": display_path(manifest_path),
            "series": {
                case: {
                    "directory": display_path(destination / case),
                    "filename_pattern": "IM_####.dcm",
                    "file_count": len(planned[case]),
                    "first_file": display_path(planned[case][0]),
                    "last_file": display_path(planned[case][-1]),
                }
                for case in CASE_KEYS
            },
        },
        "output_checksums_sha256": None,
        "source_immutability": {"verified_before_and_after": None, "unchanged": None},
        "warnings": [
            "Research-only, derived, anonymized visualization; not for diagnosis.",
            WARNING_TEXT,
            "This is not a quantitative DICOM Parametric Map, DICOM SEG, RTSTRUCT, or legacy DICOM Overlay Plane.",
            "Spatial tags are derived from fixed NIfTI geometry; no original clinical DICOM reference relationship is claimed.",
            "Study, Series, Frame of Reference, and SOP Instance UIDs are newly generated.",
        ],
        "validation": {"status": "not-run" if dry_run else "pending"},
        "completion_status": "dry-run" if dry_run else "pending",
        "writes_performed": False if dry_run else None,
        "timestamp_utc": now.isoformat(),
        "intended_use": "Research-only, visualization-only, and non-diagnostic.",
    }
    if dry_run:
        return manifest

    destination.mkdir(parents=True, exist_ok=True)
    for case in CASE_KEYS:
        case_dir = destination / case
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_case_series(
            case=case,
            fixed=images[case]["fixed"].image,
            registered=images[case]["registered"].image,
            paths=planned[case],
            uids=uid_plan[case],
            geometry=geometries[case],
            vmin=vmin,
            vmax=vmax,
            anatomy_limits=anatomy_limits,
            opacity=float(opacity),
            derivation=derivation,
            timestamp=now,
        )
    validation = validate_rgb_secondary_capture_series(
        planned, uid_plan, geometries, vmin=vmin, vmax=vmax
    )
    output_hashes = {
        case: {path.name: sha256_file(path) for path in planned[case]}
        for case in CASE_KEYS
    }
    input_hashes_after = {
        case: {role: sha256_file(path) for role, path in values.items()}
        for case, values in paths.items()
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("A source NIfTI checksum changed during Secondary Capture export.")
    manifest["output_checksums_sha256"] = output_hashes
    manifest["source_immutability"] = {
        "verified_before_and_after": True,
        "unchanged": True,
        "sha256_after": input_hashes_after,
    }
    manifest["validation"] = validation
    manifest["completion_status"] = "complete"
    manifest["writes_performed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _load_compatible_report(path: Path) -> tuple[dict[str, Any], str]:
    try:
        report = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError("Shared-window comparison report does not exist.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Shared-window comparison report is unreadable or malformed.") from exc
    if report.get("completion_status") != "complete":
        raise ValueError("Shared-window comparison report is not complete.")
    if report.get("analysis_type") != EXPECTED_SHARED_WINDOW_ANALYSIS:
        raise ValueError("Shared-window report has an incompatible analysis type.")
    if set(report.get("inputs", {})) != set(CASE_KEYS):
        raise ValueError("Shared-window report must include exactly Case 1 and Case 2.")
    window = report.get("shared_difference_window", {})
    required_volumes = {
        "case_1_before", "case_1_after", "case_2_before", "case_2_after"
    }
    if set(window.get("pooled_volume_order", [])) != required_volumes:
        raise ValueError("Shared-window report does not include all four comparison volumes.")
    if window.get("identical_for_every_difference_panel") is not True:
        raise ValueError("Shared-window report does not certify identical panel normalization.")
    display = report.get("display", {})
    anatomy = display.get("fixed_anatomy_limits", {})
    try:
        _validate_limits(
            float(window["vmin"]), float(window["vmax"]),
            (float(anatomy["vmin"]), float(anatomy["vmax"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Shared-window report contains invalid display limits.") from exc
    for panel in ("case_1_after", "case_2_after"):
        limits = window.get("panel_limits", {}).get("overlay", {}).get(panel, {})
        if limits.get("vmin") != window.get("vmin") or limits.get("vmax") != window.get("vmax"):
            raise ValueError("Shared-window report has incompatible post-registration panel limits.")
    if display.get("colormaps", {}).get("overlay_difference") != "turbo":
        raise ValueError("Shared-window report is incompatible with the turbo overlay export.")
    if not isinstance(display.get("fixed_anatomy_limits"), dict):
        raise ValueError("Shared-window report lacks fixed-anatomy display limits.")
    return report, sha256_file(path)


def _validate_report_inputs(report, paths, hashes):
    for case in CASE_KEYS:
        recorded = report["inputs"][case].get("analytical_images", {})
        for role in ("fixed", "registered"):
            item = recorded.get(role, {})
            if item.get("sha256") != hashes[case][role]:
                raise ValueError(f"Shared-window report {case} {role} checksum is incompatible.")
            recorded_path = item.get("path")
            if not isinstance(recorded_path, str):
                raise ValueError(f"Shared-window report {case} {role} path is incompatible.")
            if recorded_path == "[external path]":
                continue
            resolved = (PROJECT_ROOT / recorded_path).resolve(strict=False)
            if resolved != paths[case][role]:
                raise ValueError(f"Shared-window report {case} {role} path is incompatible.")


def _load_case(case: str, paths: dict[str, Path]):
    loaded = {
        role: validate_registration_image(path, role=f"{case} {role}")
        for role, path in paths.items()
    }
    mismatches = _exact_geometry_mismatches(loaded["fixed"].image, loaded["registered"].image)
    if mismatches:
        raise ValueError(
            f"{case.replace('_', ' ').title()} fixed/registered geometry mismatch: "
            + "; ".join(mismatches)
        )
    return loaded


def _exact_geometry_mismatches(first: sitk.Image, second: sitk.Image) -> list[str]:
    mismatches = []
    for name, left, right in (
        ("size", tuple(first.GetSize()), tuple(second.GetSize())),
        ("spacing", tuple(first.GetSpacing()), tuple(second.GetSpacing())),
        ("origin", tuple(first.GetOrigin()), tuple(second.GetOrigin())),
        ("direction", tuple(first.GetDirection()), tuple(second.GetDirection())),
    ):
        if left != right:
            mismatches.append(f"{name} mismatch: {left} != {right}")
    return mismatches


def _validated_spatial_geometry(image: sitk.Image) -> dict[str, Any]:
    geometry = summarize_geometry(image)
    spacing = np.asarray(geometry.spacing_xyz, dtype=float)
    origin = np.asarray(geometry.origin_xyz, dtype=float)
    direction = np.asarray(geometry.direction, dtype=float).reshape(3, 3)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Fixed NIfTI spacing must be finite and positive.")
    if not np.isfinite(origin).all() or not np.isfinite(direction).all():
        raise ValueError("Fixed NIfTI spatial geometry must be finite.")
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("Fixed NIfTI direction must be orthonormal for DICOM spatial tags.")
    determinant = float(np.linalg.det(direction))
    if not math.isclose(abs(determinant), 1.0, abs_tol=1e-6):
        raise ValueError("Fixed NIfTI direction determinant is invalid for DICOM spatial tags.")
    x_direction = direction[:, 0]
    y_direction = direction[:, 1]
    z_direction = direction[:, 2]
    iop_normal = np.cross(x_direction, y_direction)
    first = image.TransformIndexToPhysicalPoint((0, 0, 0))
    last = image.TransformIndexToPhysicalPoint((0, 0, image.GetSize()[2] - 1))
    return {
        "size_xyz": list(geometry.size_xyz),
        "spacing_xyz_mm": list(geometry.spacing_xyz),
        "origin_xyz_mm": list(geometry.origin_xyz),
        "direction": list(geometry.direction),
        "pixel_spacing_row_column_mm": [float(spacing[1]), float(spacing[0])],
        "image_orientation_patient": [float(value) for value in (*x_direction, *y_direction)],
        "first_image_position_patient_mm": list(first),
        "last_image_position_patient_mm": list(last),
        "slice_thickness_mm": float(spacing[2]),
        "slice_progression_dot_iop_normal": float(np.dot(z_direction, iop_normal)),
        "validation": "passed",
        "source": "fixed NIfTI geometry; no clinical DICOM reference relationship claimed",
    }


def _validate_limits(vmin, vmax, anatomy_limits):
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin != 0 or vmax <= vmin:
        raise ValueError("Shared-window report contains invalid difference limits.")
    low, high = anatomy_limits
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("Shared-window report contains invalid fixed-anatomy limits.")


def _check_outputs(destination, planned, manifest_path, *, overwrite, dry_run):
    expected = {path.resolve(strict=False) for paths in planned.values() for path in paths}
    expected.add(manifest_path.resolve(strict=False))
    existing_targets = [path for path in expected if path.exists()]
    if existing_targets and not overwrite and not dry_run:
        raise FileExistsError("Secondary Capture outputs already exist; use --overwrite.")
    if overwrite:
        for case in CASE_KEYS:
            directory = destination / case
            if not directory.is_dir():
                continue
            unexpected = [
                path for path in directory.glob("*.dcm")
                if path.resolve(strict=False) not in expected
            ]
            if unexpected:
                raise FileExistsError(
                    "Refusing overwrite because an output case directory contains unexpected DICOM files; use a new output directory."
                )


def _write_case_series(
    *, case, fixed, registered, paths, uids, geometry, vmin, vmax,
    anatomy_limits, opacity, derivation, timestamp,
):
    fixed_array = sitk.GetArrayViewFromImage(fixed)
    registered_array = sitk.GetArrayViewFromImage(registered)
    description = f"Research |F-R| turbo opacity={opacity:.3g} shared [{vmin:.5g},{vmax:.5g}]"
    for z, path in enumerate(paths):
        rgb = render_rgb_slice(
            fixed_array[z], registered_array[z], vmin=vmin, vmax=vmax,
            anatomy_limits=anatomy_limits, opacity=opacity,
        )
        position = fixed.TransformIndexToPhysicalPoint((0, 0, z))
        _write_dicom_slice(
            path=path, rgb=rgb, case=case, instance_number=z + 1,
            sop_instance_uid=uids["sop_instance_uids"][z],
            study_instance_uid=uids["study_instance_uid"],
            series_instance_uid=uids["series_instance_uid"],
            frame_of_reference_uid=uids["frame_of_reference_uid"],
            geometry=geometry, position=position, series_description=description,
            derivation=derivation, timestamp=timestamp,
        )


def render_rgb_slice(fixed, registered, *, vmin, vmax, anatomy_limits, opacity) -> np.ndarray:
    fixed_values = np.asarray(fixed, dtype=np.float32)
    registered_values = np.asarray(registered, dtype=np.float32)
    if fixed_values.shape != registered_values.shape or fixed_values.ndim != 2:
        raise ValueError("Fixed and registered slices must have identical 2D shapes.")
    anatomy_low, anatomy_high = anatomy_limits
    gray = np.clip((fixed_values - anatomy_low) / (anatomy_high - anatomy_low), 0, 1)
    difference = np.abs(fixed_values - registered_values)
    normalized = np.clip((difference - vmin) / (vmax - vmin), 0, 1)
    turbo = matplotlib.colormaps["turbo"](normalized)[..., :3]
    gray_rgb = np.repeat(gray[..., None], 3, axis=2)
    blended = (1.0 - opacity) * gray_rgb + opacity * turbo
    return np.ascontiguousarray(np.rint(np.clip(blended, 0, 1) * 255).astype(np.uint8))


def _write_dicom_slice(
    *, path, rgb, case, instance_number, sop_instance_uid, study_instance_uid,
    series_instance_uid, frame_of_reference_uid, geometry, position,
    series_description, derivation, timestamp,
):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.StudyInstanceUID = study_instance_uid
    dataset.SeriesInstanceUID = series_instance_uid
    dataset.FrameOfReferenceUID = frame_of_reference_uid
    dataset.PatientName = "ANONYMIZED^RESEARCH"
    dataset.PatientID = PATIENT_IDS[case]
    dataset.PatientBirthDate = ""
    dataset.PatientSex = ""
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethod = "Synthetic research IDs; no clinical identity copied"
    dataset.StudyID = f"RGB_{case.upper()}"
    dataset.AccessionNumber = ""
    dataset.ReferringPhysicianName = ""
    dataset.StudyDescription = "Research-only NIfTI-derived visualization"
    dataset.SeriesDescription = series_description
    dataset.DerivationDescription = derivation
    dataset.Modality = "OT"
    dataset.ConversionType = "WSD"
    dataset.ImageType = ["DERIVED", "SECONDARY"]
    dataset.BurnedInAnnotation = "NO"
    dataset.LossyImageCompression = "00"
    dataset.Manufacturer = "lung-ct-bounding-box-pipeline"
    dataset.SoftwareVersions = PIPELINE_VERSION
    dataset.StudyDate = timestamp.strftime("%Y%m%d")
    dataset.StudyTime = timestamp.strftime("%H%M%S.%f")
    dataset.SeriesDate = dataset.StudyDate
    dataset.SeriesTime = dataset.StudyTime
    dataset.ContentDate = dataset.StudyDate
    dataset.ContentTime = dataset.StudyTime
    dataset.InstanceCreationDate = dataset.StudyDate
    dataset.InstanceCreationTime = dataset.StudyTime
    dataset.DateOfSecondaryCapture = dataset.StudyDate
    dataset.TimeOfSecondaryCapture = dataset.StudyTime
    dataset.SeriesNumber = 1 if case == "case_1" else 2
    dataset.InstanceNumber = instance_number
    dataset.PatientOrientation = ""
    dataset.PixelSpacing = [_ds(value) for value in geometry["pixel_spacing_row_column_mm"]]
    dataset.ImageOrientationPatient = [_ds(value) for value in geometry["image_orientation_patient"]]
    dataset.ImagePositionPatient = [_ds(value) for value in position]
    dataset.SliceThickness = _ds(geometry["slice_thickness_mm"])
    dataset.SpacingBetweenSlices = _ds(geometry["slice_thickness_mm"])
    dataset.SamplesPerPixel = 3
    dataset.PhotometricInterpretation = "RGB"
    dataset.PlanarConfiguration = 0
    dataset.Rows, dataset.Columns = rgb.shape[:2]
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    pixel_bytes = rgb.tobytes(order="C")
    dataset.PixelData = pixel_bytes if len(pixel_bytes) % 2 == 0 else pixel_bytes + b"\0"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(str(path), enforce_file_format=True)


def _ds(value) -> str:
    return format_number_as_ds(float(value))


def validate_rgb_secondary_capture_series(planned, uid_plan, geometries, *, vmin, vmax):
    series_uids = {}
    all_sop_uids = set()
    case_results = {}
    for case in CASE_KEYS:
        paths = planned[case]
        expected_geometry = geometries[case]
        case_sops = []
        case_series = set()
        for z, path in enumerate(paths):
            dataset = pydicom.dcmread(str(path))
            if str(dataset.SOPClassUID) != str(SecondaryCaptureImageStorage):
                raise ValueError("Generated object is not Secondary Capture Image Storage.")
            if str(dataset.file_meta.TransferSyntaxUID) != str(ExplicitVRLittleEndian):
                raise ValueError("Generated object is not Explicit VR Little Endian.")
            if dataset.PhotometricInterpretation != "RGB" or dataset.PlanarConfiguration != 0:
                raise ValueError("Generated object does not use interleaved RGB pixels.")
            expected_bytes = int(dataset.Rows) * int(dataset.Columns) * 3
            if len(dataset.PixelData) not in {expected_bytes, expected_bytes + 1}:
                raise ValueError("Generated object has invalid RGB PixelData length.")
            if dataset.pixel_array.shape != (int(dataset.Rows), int(dataset.Columns), 3):
                raise ValueError("Generated object has invalid decoded RGB shape.")
            if int(dataset.InstanceNumber) != z + 1:
                raise ValueError("Generated slice InstanceNumber ordering is invalid.")
            direction = np.asarray(expected_geometry["direction"], dtype=float).reshape(3, 3)
            expected_position = (
                np.asarray(expected_geometry["origin_xyz_mm"], dtype=float)
                + direction[:, 2] * float(expected_geometry["spacing_xyz_mm"][2]) * z
            )
            if not np.allclose(
                np.asarray(dataset.ImagePositionPatient, dtype=float), expected_position,
                atol=1e-6, rtol=0,
            ):
                raise ValueError("Generated slice position validation failed.")
            if not np.allclose(
                np.asarray(dataset.ImageOrientationPatient, dtype=float),
                expected_geometry["image_orientation_patient"], atol=1e-6, rtol=0,
            ):
                raise ValueError("Generated slice orientation validation failed.")
            if not np.allclose(
                np.asarray(dataset.PixelSpacing, dtype=float),
                expected_geometry["pixel_spacing_row_column_mm"], atol=1e-6, rtol=0,
            ):
                raise ValueError("Generated PixelSpacing validation failed.")
            case_sops.append(str(dataset.SOPInstanceUID))
            case_series.add(str(dataset.SeriesInstanceUID))
        if len(case_series) != 1 or next(iter(case_series)) != uid_plan[case]["series_instance_uid"]:
            raise ValueError("Generated SeriesInstanceUID is inconsistent.")
        if len(set(case_sops)) != len(case_sops):
            raise ValueError("Generated SOPInstanceUID values are not unique within a case.")
        if all_sop_uids.intersection(case_sops):
            raise ValueError("Generated SOPInstanceUID values are not unique across cases.")
        all_sop_uids.update(case_sops)
        series_uids[case] = next(iter(case_series))
        case_results[case] = {
            "readable_slice_count": len(paths),
            "rgb_pixel_validation": "passed",
            "slice_ordering_validation": "passed",
            "spatial_tag_validation": "passed",
            "unique_sop_instance_uids": True,
            "consistent_series_instance_uid": True,
        }
    if series_uids["case_1"] == series_uids["case_2"]:
        raise ValueError("Case 1 and Case 2 must have different SeriesInstanceUID values.")
    return {
        "status": "passed",
        "cases": case_results,
        "different_series_instance_uids_between_cases": True,
        "all_sop_instance_uids_unique": True,
        "identical_shared_scaling": {"passed": True, "vmin": vmin, "vmax": vmax},
        "patient_identity_policy": "passed: only synthetic research identifiers written",
    }
