from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import SimpleITK as sitk

from lung_masked_difference_visualization import (
    CASE_KEYS,
    _mask_and_clipping_statistics,
    _source_hashes,
    prepare_lung_masked_inputs,
    render_lung_masked_turbo_rgb,
)
from project_paths import display_path
from registration_inputs import sha256_file
from rgb_secondary_capture_export import (
    PATIENT_IDS,
    _check_outputs,
    _validated_spatial_geometry,
    _write_dicom_slice,
    validate_rgb_secondary_capture_series,
)
from version import PIPELINE_VERSION


ANALYSIS_TYPE = "Derived RGB DICOM Secondary Capture of lung-masked turbo differences"
MANIFEST_NAME = "lung_masked_rgb_secondary_capture_manifest.json"
WARNING_TEXT = (
    "RGB Secondary Capture pixels are visualization-only, irreversibly windowed and blended, "
    "and cannot recover quantitative absolute-difference values."
)


def export_lung_masked_rgb_secondary_capture(
    *,
    case_1_dir: str | Path,
    case_2_dir: str | Path,
    case_1_lung_mask: str | Path,
    case_2_lung_mask: str | Path,
    shared_window_report: str | Path,
    output_dir: str | Path,
    opacity: float = 0.45,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Export post-registration masked-color difference as RGB Secondary Capture."""
    context = prepare_lung_masked_inputs(
        case_1_dir=case_1_dir,
        case_2_dir=case_2_dir,
        case_1_lung_mask=case_1_lung_mask,
        case_2_lung_mask=case_2_lung_mask,
        shared_window_report=shared_window_report,
    )
    if not 0 <= float(opacity) <= 1:
        raise ValueError("Opacity must be between 0 and 1.")
    destination = Path(output_dir).expanduser().resolve(strict=False)
    vmin, vmax = context["limits"]
    geometries = {
        case: _validated_spatial_geometry(context["cases"][case]["fixed"].image)
        for case in CASE_KEYS
    }
    slice_counts = {
        case: int(context["cases"][case]["fixed"].image.GetSize()[2]) for case in CASE_KEYS
    }
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
        f"Derived |F - R| visualization: grayscale absolute difference outside the fixed-space "
        f"lung mask; turbo absolute difference restricted to the lung mask and blended with "
        f"grayscale difference at opacity={float(opacity):.6g}; shared window "
        f"[{vmin:.12g}, {vmax:.12g}]. Background is difference, not fixed anatomy. "
        "Research-only visualization from NIfTI sources; no original clinical DICOM reference "
        "relationship is claimed. Quantitative values cannot be recovered from RGB pixels."
    )
    now = datetime.now(timezone.utc)
    clipping = _mask_and_clipping_statistics(context["arrays"], context["masks"], vmax)
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
            "is_clinical_overlay": False,
        },
        "inputs": context["input_report"],
        "formula": "abs(fixed - registered)",
        "display": {
            "background_everywhere": "grayscale abs(fixed - registered)",
            "inside_lung_mask": "(1-opacity)*grayscale difference + opacity*turbo difference",
            "outside_lung_mask": "grayscale abs(fixed - registered), retained and not zeroed",
            "fixed_anatomy_used_as_background": False,
            "lung_mask_restricts_analysis": False,
            "difference_colormap_inside_mask": "turbo",
            "opacity": float(opacity),
            "shared_difference_limits": {"vmin": vmin, "vmax": vmax},
            "identical_map_normalization_across_cases": True,
            "quantitative_values_recoverable_from_rgb": False,
        },
        "geometry_validation": context["geometry_report"],
        "dicom_geometry": geometries,
        "mask_and_clipping_statistics": clipping,
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
            "descriptive": ["SeriesDescription", "DerivationDescription", "StudyDescription"],
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
            "series_description": "Research |F-R| lung-mask turbo; gray difference outside",
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
            "This is not a quantitative DICOM Parametric Map, DICOM SEG, RTSTRUCT, legacy DICOM Overlay Plane, or clinical overlay.",
            "Spatial tags are derived from fixed NIfTI geometry; no original clinical DICOM reference relationship is claimed.",
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
        _write_case_series(
            case=case,
            fixed=context["cases"][case]["fixed"].image,
            registered=context["cases"][case]["registered"].image,
            mask=context["masks"][case],
            paths=planned[case],
            uids=uid_plan[case],
            geometry=geometries[case],
            vmin=vmin,
            vmax=vmax,
            opacity=float(opacity),
            derivation=derivation,
            timestamp=now,
        )
    validation = validate_rgb_secondary_capture_series(
        planned, uid_plan, geometries, vmin=vmin, vmax=vmax
    )
    output_hashes = {
        case: {path.name: sha256_file(path) for path in planned[case]} for case in CASE_KEYS
    }
    hashes_after = _source_hashes(context["source_paths"])
    if hashes_after != context["hashes_before"]:
        raise RuntimeError("A source NIfTI, mask, manifest, or shared report changed during export.")
    manifest["output_checksums_sha256"] = output_hashes
    manifest["source_immutability"] = {
        "verified_before_and_after": True,
        "unchanged": True,
        "sha256_after": hashes_after,
    }
    manifest["validation"] = validation
    manifest["completion_status"] = "complete"
    manifest["writes_performed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _write_case_series(
    *, case, fixed, registered, mask, paths, uids, geometry, vmin, vmax,
    opacity, derivation, timestamp,
):
    fixed_array = sitk.GetArrayViewFromImage(fixed)
    registered_array = sitk.GetArrayViewFromImage(registered)
    description = "Research |F-R| lung-mask turbo; gray difference outside"
    for z, path in enumerate(paths):
        difference = np.abs(
            fixed_array[z].astype(np.float32) - registered_array[z].astype(np.float32)
        )
        rgb_float = render_lung_masked_turbo_rgb(
            difference, mask[z], vmin=vmin, vmax=vmax, opacity=opacity
        )
        rgb = np.ascontiguousarray(np.rint(rgb_float * 255).astype(np.uint8))
        position = fixed.TransformIndexToPhysicalPoint((0, 0, z))
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
            position=position,
            series_description=description,
            derivation=derivation,
            timestamp=timestamp,
        )
