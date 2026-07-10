from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import ExplicitVRLittleEndian, RTStructureSetStorage, generate_uid
import SimpleITK as sitk

from bounding_box import BBoxZYX
from dicom_reference import (
    DicomReferenceSeries,
    DicomSliceRef,
    inspect_reference_series,
    map_image_slices_to_dicom,
    reference_sitk_geometry,
)
from medical_image_io import compare_image_geometry, summarize_geometry
from project_paths import DEFAULT_OUTPUTS_DIR, display_path
from version import PIPELINE_VERSION


@dataclass(frozen=True)
class ROIInput:
    name: str
    mask_path: Path | None = None
    bbox: BBoxZYX | None = None


@dataclass(frozen=True)
class ExportResult:
    report: dict[str, Any]
    output_path: Path | None


def validate_dicom_image_alignment(image: sitk.Image, series: DicomReferenceSeries) -> list[DicomSliceRef]:
    ref = reference_sitk_geometry(series)
    geom = summarize_geometry(image)
    failures = []
    if geom.size_xyz != ref["size_xyz"]:
        failures.append(f"size mismatch: {geom.size_xyz} != {ref['size_xyz']}")
    if not np.allclose(geom.spacing_xyz, ref["spacing_xyz"], atol=1e-3, rtol=0):
        failures.append(f"spacing mismatch: {geom.spacing_xyz} != {ref['spacing_xyz']}")
    if not np.allclose(geom.origin_xyz, ref["origin_xyz"], atol=1e-3, rtol=0):
        failures.append(f"origin mismatch: {geom.origin_xyz} != {ref['origin_xyz']}")
    if not np.allclose(geom.direction, ref["direction"], atol=1e-3, rtol=0):
        failures.append("direction mismatch between NIfTI image and DICOM series")
    if failures:
        raise ValueError("DICOM/NIfTI geometry mismatch: " + "; ".join(failures))
    return map_image_slices_to_dicom(image, series)


def mask_to_contours_by_slice(mask_image: sitk.Image, slice_mapping: Sequence[DicomSliceRef]) -> dict[int, list[list[float]]]:
    mask = sitk.GetArrayFromImage(mask_image) > 0
    contours: dict[int, list[list[float]]] = {}
    for z in np.flatnonzero(mask.any(axis=(1, 2))):
        slice_contours = []
        for loop in binary_slice_to_index_loops(mask[z]):
            physical = []
            for x, y in loop:
                point = mask_image.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
                physical.extend([float(point[0]), float(point[1]), float(point[2])])
            if _valid_flat_contour(physical):
                slice_contours.append(physical)
        if slice_contours:
            contours[int(z)] = slice_contours
    return contours


def bbox_to_contours_by_slice(image: sitk.Image, bbox: BBoxZYX) -> dict[int, list[list[float]]]:
    contours = {}
    for z in range(bbox.z_min, bbox.z_max):
        corners = [
            (bbox.x_min, bbox.y_min, z),
            (bbox.x_max, bbox.y_min, z),
            (bbox.x_max, bbox.y_max, z),
            (bbox.x_min, bbox.y_max, z),
            (bbox.x_min, bbox.y_min, z),
        ]
        flat = []
        for x, y, z_index in corners:
            point = image.TransformContinuousIndexToPhysicalPoint((x, y, z_index))
            flat.extend([float(point[0]), float(point[1]), float(point[2])])
        contours[z] = [flat]
    return contours


def binary_slice_to_index_loops(mask_2d: np.ndarray) -> list[list[tuple[float, float]]]:
    mask = mask_2d.astype(bool)
    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    rows, cols = mask.shape
    for y in range(rows):
        for x in range(cols):
            if not mask[y, x]:
                continue
            left, right = x - 0.5, x + 0.5
            top, bottom = y - 0.5, y + 0.5
            if y == 0 or not mask[y - 1, x]:
                edges.append(((left, top), (right, top)))
            if x == cols - 1 or not mask[y, x + 1]:
                edges.append(((right, top), (right, bottom)))
            if y == rows - 1 or not mask[y + 1, x]:
                edges.append(((right, bottom), (left, bottom)))
            if x == 0 or not mask[y, x - 1]:
                edges.append(((left, bottom), (left, top)))
    return _stitch_edges(edges)


def _stitch_edges(edges):
    unused = edges[:]
    loops = []
    while unused:
        start, end = unused.pop(0)
        loop = [start, end]
        while loop[-1] != loop[0]:
            idx = next((i for i, edge in enumerate(unused) if edge[0] == loop[-1]), None)
            if idx is None:
                idx = next((i for i, edge in enumerate(unused) if edge[1] == loop[-1]), None)
                if idx is None:
                    break
                a, b = unused.pop(idx)
                loop.append(a)
            else:
                a, b = unused.pop(idx)
                loop.append(b)
        if loop[-1] == loop[0] and len(set(loop[:-1])) >= 3:
            loops.append(loop)
    return loops


def write_rtstruct(
    *,
    series: DicomReferenceSeries,
    roi_contours: dict[str, dict[int, list[list[float]]]],
    slice_mapping: Sequence[DicomSliceRef],
    output_path: Path,
) -> Path:
    first = pydicom.dcmread(str(series.slices[0].path), stop_before_pixels=True)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTStructureSetStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = RTStructureSetStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = series.study_instance_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = series.frame_of_reference_uid
    ds.Modality = "RTSTRUCT"
    ds.StructureSetLabel = "LUNG_BBOX_QC"
    ds.StructureSetName = "Lung segmentation and bbox"
    ds.StructureSetDate = datetime.now().strftime("%Y%m%d")
    ds.StructureSetTime = datetime.now().strftime("%H%M%S")
    ds.PatientName = getattr(first, "PatientName", "")
    ds.PatientID = getattr(first, "PatientID", "")
    ds.PatientBirthDate = getattr(first, "PatientBirthDate", "")
    ds.PatientSex = getattr(first, "PatientSex", "")
    ds.StudyDate = getattr(first, "StudyDate", "")
    ds.StudyTime = getattr(first, "StudyTime", "")
    ds.Manufacturer = "lung-ct-bounding-box-pipeline"

    ds.ReferencedFrameOfReferenceSequence = DicomSequence([
        _referenced_frame_of_reference(series)
    ])
    ds.StructureSetROISequence = DicomSequence()
    ds.ROIContourSequence = DicomSequence()
    ds.RTROIObservationsSequence = DicomSequence()
    colors = [(255, 80, 80), (80, 200, 255), (255, 200, 80), (120, 255, 120)]
    sop_by_z = {z: ref.sop_instance_uid for z, ref in enumerate(slice_mapping)}

    for idx, (name, contour_by_slice) in enumerate(roi_contours.items(), start=1):
        roi = Dataset()
        roi.ROINumber = idx
        roi.ReferencedFrameOfReferenceUID = series.frame_of_reference_uid
        roi.ROIName = name
        roi.ROIGenerationAlgorithm = "AUTOMATIC"
        ds.StructureSetROISequence.append(roi)

        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = idx
        roi_contour.ROIDisplayColor = list(colors[(idx - 1) % len(colors)])
        roi_contour.ContourSequence = DicomSequence()
        for z, contours in sorted(contour_by_slice.items()):
            for contour in contours:
                item = Dataset()
                item.ContourGeometricType = "CLOSED_PLANAR"
                item.NumberOfContourPoints = len(contour) // 3
                item.ContourData = [float(v) for v in contour]
                ref = Dataset()
                ref.ReferencedSOPClassUID = series.slices[z].sop_class_uid
                ref.ReferencedSOPInstanceUID = sop_by_z[z]
                item.ContourImageSequence = DicomSequence([ref])
                roi_contour.ContourSequence.append(item)
        ds.ROIContourSequence.append(roi_contour)

        obs = Dataset()
        obs.ObservationNumber = idx
        obs.ReferencedROINumber = idx
        obs.RTROIInterpretedType = "ORGAN"
        obs.ROIInterpreter = ""
        ds.RTROIObservationsSequence.append(obs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path), write_like_original=False)
    validate_rtstruct(output_path, {s.sop_instance_uid for s in series.slices})
    return output_path


def validate_rtstruct(path: Path, source_sop_uids: set[str]) -> dict[str, Any]:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    if str(ds.SOPClassUID) != str(RTStructureSetStorage):
        raise ValueError("Generated DICOM is not RT Structure Set Storage.")
    referenced = set()
    contour_counts = {}
    for roi_contour in ds.ROIContourSequence:
        count = 0
        for contour in getattr(roi_contour, "ContourSequence", []):
            coords = np.array(contour.ContourData, dtype=float).reshape(-1, 3)
            if coords.shape[0] < 3 or len(np.unique(coords, axis=0)) < 3:
                raise ValueError("Generated RTSTRUCT contains a degenerate contour.")
            if not np.isfinite(coords).all():
                raise ValueError("Generated RTSTRUCT contains non-finite contour coordinates.")
            for ref in contour.ContourImageSequence:
                referenced.add(str(ref.ReferencedSOPInstanceUID))
            count += 1
        contour_counts[int(roi_contour.ReferencedROINumber)] = count
    missing = referenced - source_sop_uids
    if missing:
        raise ValueError("Generated RTSTRUCT references SOP Instance UIDs outside the source series.")
    return {"referenced_sop_count": len(referenced), "contour_counts_by_roi_number": contour_counts}


def export_osirix_package(
    *,
    dicom_dir: str | Path,
    image_path: str | Path,
    mask_path: str | Path,
    case_id: str,
    bbox_json: str | Path | None,
    individual_masks: Sequence[tuple[str, str | Path]] = (),
    output_root: str | Path = DEFAULT_OUTPUTS_DIR / "osirix",
    export_format: str = "auto",
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve(strict=False)
    case_dir = output_root / case_id
    report_path = case_dir / "export_report.json"
    rt_path = case_dir / "lung_segmentation_and_bbox_rtstruct.dcm"
    if case_dir.exists() and any(case_dir.iterdir()) and not overwrite and not dry_run:
        raise FileExistsError(f"Refusing to overwrite existing OsiriX export: {display_path(case_dir)}")

    series = inspect_reference_series(dicom_dir)
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if compare_image_geometry(image, mask):
        raise ValueError("Image and combined mask geometry do not match.")

    selected_format = "rtstruct" if export_format in {"auto", "rtstruct"} and series.report.rtstruct_compatible else "fallback-package"
    warnings = list(series.report.warnings)
    if export_format == "dicom-seg":
        raise NotImplementedError(
            "DICOM SEG export is not implemented in this stage; use --format auto or rtstruct."
        )
    if export_format == "rtstruct" and not series.report.rtstruct_compatible:
        raise ValueError("RTSTRUCT requested but reference series is not compatible: " + "; ".join(series.report.failures))

    contour_counts: dict[str, int] = {}
    referenced_count = 0
    roi_names: list[str] = []
    if selected_format == "rtstruct":
        slice_mapping = validate_dicom_image_alignment(image, series)
        roi_contours: dict[str, dict[int, list[list[float]]]] = {
            "Whole Lung Segmentation": mask_to_contours_by_slice(mask, slice_mapping)
        }
        for label, path in individual_masks:
            indiv = sitk.ReadImage(str(path))
            if compare_image_geometry(image, indiv):
                raise ValueError(f"Individual mask geometry mismatch: {label}")
            roi_contours[label.replace("_", " ").title()] = mask_to_contours_by_slice(indiv, slice_mapping)
        if bbox_json:
            bbox = _bbox_from_json(Path(bbox_json))
            roi_contours["Lung Bounding Box"] = bbox_to_contours_by_slice(image, bbox)
        roi_names = list(roi_contours)
        contour_counts = {name: sum(len(v) for v in by_slice.values()) for name, by_slice in roi_contours.items()}
        referenced_count = len({series.slices[z].sop_instance_uid for by_slice in roi_contours.values() for z in by_slice})
        if not dry_run:
            if case_dir.exists() and overwrite:
                shutil.rmtree(case_dir)
            write_rtstruct(series=series, roi_contours=roi_contours, slice_mapping=slice_mapping, output_path=rt_path)
    else:
        roi_names = ["fallback review package"]
        warnings.append("RTSTRUCT export was not possible; generated a non-DICOM fallback review package.")
        if not dry_run:
            if case_dir.exists() and overwrite:
                shutil.rmtree(case_dir)
            _write_fallback_package(case_dir, image_path, mask_path, bbox_json)

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": case_id,
        "selected_export_format": selected_format,
        "capability_classification": series.report.classification,
        "source_sop_class_category": series.report.source_sop_class_category,
        "source_image_count": series.report.image_count,
        "frame_of_reference_available": series.report.frame_of_reference_available,
        "geometry_validation_result": "passed" if selected_format != "fallback-package" else "not-applicable",
        "image_to_dicom_slice_mapping_result": "passed" if selected_format != "fallback-package" else "not-applicable",
        "roi_names": roi_names,
        "contour_counts_per_roi": contour_counts,
        "referenced_sop_instance_count": referenced_count,
        "output_filename": rt_path.name if selected_format == "rtstruct" else "review_package",
        "warnings": warnings,
        "completion_status": "dry-run" if dry_run else "success",
        "software_versions": {"pydicom": pydicom.__version__},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not dry_run:
        case_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        _write_export_readme(case_dir, selected_format)
    return report


def _referenced_frame_of_reference(series: DicomReferenceSeries) -> Dataset:
    frame = Dataset()
    frame.FrameOfReferenceUID = series.frame_of_reference_uid
    study = Dataset()
    study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.2"
    study.ReferencedSOPInstanceUID = series.study_instance_uid
    series_ref = Dataset()
    series_ref.SeriesInstanceUID = series.series_instance_uid
    series_ref.ContourImageSequence = DicomSequence()
    for s in series.slices:
        item = Dataset()
        item.ReferencedSOPClassUID = s.sop_class_uid
        item.ReferencedSOPInstanceUID = s.sop_instance_uid
        series_ref.ContourImageSequence.append(item)
    study.RTReferencedSeriesSequence = DicomSequence([series_ref])
    frame.RTReferencedStudySequence = DicomSequence([study])
    return frame


def _valid_flat_contour(contour: list[float]) -> bool:
    if len(contour) < 9 or len(contour) % 3 != 0:
        return False
    arr = np.array(contour, dtype=float).reshape(-1, 3)
    return np.isfinite(arr).all() and len(np.unique(arr, axis=0)) >= 3


def _bbox_from_json(path: Path) -> BBoxZYX:
    data = json.loads(path.read_text())
    values = data["voxel_bbox_half_open_zyx"]
    return BBoxZYX(
        z_min=int(values["z_min"]),
        z_max=int(values["z_max"]),
        y_min=int(values["y_min"]),
        y_max=int(values["y_max"]),
        x_min=int(values["x_min"]),
        x_max=int(values["x_max"]),
    )


def _write_fallback_package(case_dir: Path, image_path, mask_path, bbox_json) -> None:
    review = case_dir / "review_package"
    review.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, review / "image.nii.gz")
    shutil.copy2(mask_path, review / "combined_lungs.nii.gz")
    if bbox_json:
        shutil.copy2(bbox_json, review / "bbox.json")
    figures_dir = DEFAULT_OUTPUTS_DIR / "figures" / case_dir.name
    for figure_name in ["segmentation_overlay.png", "segmentation_contours.png"]:
        figure_path = figures_dir / figure_name
        if figure_path.exists():
            shutil.copy2(figure_path, review / figure_name)
    (review / "README.txt").write_text(
        "Fallback review package only. A standards-compliant referenced DICOM ROI object could not be guaranteed.\n"
        "Open the original DICOM series separately and use these files for manual research review only.\n"
    )


def _write_export_readme(case_dir: Path, selected_format: str) -> None:
    (case_dir / "README.txt").write_text(
        f"OsiriX research export format: {selected_format}.\n"
        "Generated DICOM-derived objects remain private medical data and must stay local.\n"
        "Import the original DICOM series and this exported object manually into OsiriX.\n"
    )
