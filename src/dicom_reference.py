from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pydicom
from pydicom.uid import SecondaryCaptureImageStorage


STRUCTURAL_TAGS = [
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "FrameOfReferenceUID",
    "Modality",
    "Rows",
    "Columns",
    "PixelSpacing",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "SliceThickness",
    "SpacingBetweenSlices",
    "InstanceNumber",
]


@dataclass(frozen=True)
class DicomSliceRef:
    path: Path
    sop_class_uid: str
    sop_instance_uid: str
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str | None
    modality: str | None
    rows: int
    columns: int
    pixel_spacing: tuple[float, float] | None
    orientation: tuple[float, ...] | None
    position: tuple[float, ...] | None
    slice_thickness: float | None
    instance_number: int | None
    slice_coordinate: float | None


@dataclass(frozen=True)
class DicomReferenceReport:
    classification: str
    source_sop_class_category: str
    image_count: int
    frame_of_reference_available: bool
    rtstruct_compatible: bool
    dicom_seg_compatible: bool
    image_viewing_only: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "source_sop_class_category": self.source_sop_class_category,
            "source_image_count": self.image_count,
            "frame_of_reference_available": self.frame_of_reference_available,
            "rtstruct_compatible": self.rtstruct_compatible,
            "dicom_seg_compatible": self.dicom_seg_compatible,
            "image_viewing_only": self.image_viewing_only,
            "failed_criteria": list(self.failures),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class DicomReferenceSeries:
    dicom_dir: Path
    slices: tuple[DicomSliceRef, ...]
    report: DicomReferenceReport

    @property
    def study_instance_uid(self) -> str:
        return self.slices[0].study_instance_uid

    @property
    def series_instance_uid(self) -> str:
        return self.slices[0].series_instance_uid

    @property
    def frame_of_reference_uid(self) -> str | None:
        return self.slices[0].frame_of_reference_uid

    @property
    def rows(self) -> int:
        return self.slices[0].rows

    @property
    def columns(self) -> int:
        return self.slices[0].columns

    @property
    def pixel_spacing(self) -> tuple[float, float] | None:
        return self.slices[0].pixel_spacing

    @property
    def orientation(self) -> tuple[float, ...] | None:
        return self.slices[0].orientation

    def safe_summary(self) -> dict[str, Any]:
        summary = self.report.to_safe_dict()
        summary["series_uid_hash"] = _hash_uid(self.series_instance_uid)
        summary["study_uid_hash"] = _hash_uid(self.study_instance_uid)
        summary["frame_of_reference_uid_hash"] = (
            _hash_uid(self.frame_of_reference_uid) if self.frame_of_reference_uid else None
        )
        return summary


def inspect_reference_series(dicom_dir: str | Path) -> DicomReferenceSeries:
    root = Path(dicom_dir).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ValueError("DICOM reference input must be a directory.")

    files = sorted(path for path in root.glob("*") if path.is_file())
    slices = []
    for path in files:
        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
                specific_tags=STRUCTURAL_TAGS,
            )
        except Exception:
            continue
        if not hasattr(dataset, "SOPInstanceUID"):
            continue
        slices.append(_slice_from_dataset(path, dataset))

    if not slices:
        raise ValueError("No readable DICOM image files found in the supplied directory.")

    sorted_slices, warnings = sort_slices(slices)
    report = classify_reference_series(sorted_slices, warnings=warnings)
    return DicomReferenceSeries(
        dicom_dir=root,
        slices=tuple(sorted_slices),
        report=report,
    )


def classify_reference_series(
    slices: Sequence[DicomSliceRef], *, warnings: Sequence[str] = ()
) -> DicomReferenceReport:
    failures = []
    warnings = list(warnings)

    if len({s.series_instance_uid for s in slices}) != 1:
        failures.append("Files contain multiple SeriesInstanceUID values.")
    if len({s.study_instance_uid for s in slices}) != 1:
        failures.append("Files contain multiple StudyInstanceUID values.")
    sop_uids = [s.sop_instance_uid for s in slices]
    if len(set(sop_uids)) != len(sop_uids):
        failures.append("SOPInstanceUID values are not unique.")
    if len({s.rows for s in slices}) != 1 or len({s.columns for s in slices}) != 1:
        failures.append("Rows/Columns are inconsistent.")
    if len({s.pixel_spacing for s in slices}) != 1:
        failures.append("PixelSpacing is missing or inconsistent.")
    if len({s.orientation for s in slices}) != 1:
        failures.append("ImageOrientationPatient is missing or inconsistent.")
    if any(s.position is None for s in slices):
        failures.append("ImagePositionPatient is missing for one or more slices.")

    frame_values = {s.frame_of_reference_uid for s in slices}
    frame_available = None not in frame_values and len(frame_values) == 1
    if not frame_available:
        failures.append("FrameOfReferenceUID is missing or inconsistent.")

    positions = [s.slice_coordinate for s in slices]
    if any(value is None for value in positions):
        failures.append("Slice positions are not usable for one-to-one mapping.")
    elif len(set(round(float(value), 6) for value in positions)) != len(positions):
        failures.append("Slice positions are not unique.")

    sop_classes = {s.sop_class_uid for s in slices}
    source_category = (
        "secondary-capture"
        if sop_classes == {str(SecondaryCaptureImageStorage)}
        else "image-storage"
    )
    if source_category == "secondary-capture":
        warnings.append(
            "Reference images use Secondary Capture SOP Class; RTSTRUCT export is allowed only if geometry and frame references are complete."
        )

    rt_ok = not failures
    classification = "RTSTRUCT-compatible" if rt_ok else "image-viewing-only"
    return DicomReferenceReport(
        classification=classification,
        source_sop_class_category=source_category,
        image_count=len(slices),
        frame_of_reference_available=frame_available,
        rtstruct_compatible=rt_ok,
        dicom_seg_compatible=rt_ok,
        image_viewing_only=not rt_ok,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def sort_slices(slices: Sequence[DicomSliceRef]) -> tuple[list[DicomSliceRef], list[str]]:
    warnings: list[str] = []
    if all(s.slice_coordinate is not None for s in slices):
        return sorted(slices, key=lambda item: float(item.slice_coordinate)), warnings
    if all(s.instance_number is not None for s in slices):
        warnings.append("Sorted slices by InstanceNumber because slice positions were incomplete.")
        return sorted(slices, key=lambda item: int(item.instance_number)), warnings
    warnings.append("Sorted slices by filename because spatial ordering metadata were incomplete.")
    return sorted(slices, key=lambda item: item.path.name), warnings


def reference_sitk_geometry(series: DicomReferenceSeries) -> dict[str, Any]:
    orientation = np.array(series.orientation, dtype=float)
    row_dir = orientation[:3]
    col_dir = orientation[3:]
    normal = np.cross(row_dir, col_dir)
    if len(series.slices) > 1:
        slice_spacing = abs(
            float(series.slices[1].slice_coordinate) - float(series.slices[0].slice_coordinate)
        )
    else:
        slice_spacing = float(series.slices[0].slice_thickness or 1.0)
    row_spacing, col_spacing = series.pixel_spacing
    return {
        "size_xyz": (series.columns, series.rows, len(series.slices)),
        "spacing_xyz": (float(col_spacing), float(row_spacing), float(slice_spacing)),
        "origin_xyz": tuple(float(v) for v in series.slices[0].position),
        "direction": tuple(float(v) for v in (*row_dir, *col_dir, *normal)),
    }


def map_image_slices_to_dicom(image, series: DicomReferenceSeries, tolerance_mm: float = 1e-3) -> list[DicomSliceRef]:
    import SimpleITK as sitk

    size = image.GetSize()
    if size[2] != len(series.slices):
        raise ValueError(
            f"Image slice count {size[2]} does not match DICOM slice count {len(series.slices)}."
        )
    mapping = []
    positions = np.array([s.position for s in series.slices], dtype=float)
    for z in range(size[2]):
        point = np.array(image.TransformIndexToPhysicalPoint((0, 0, z)), dtype=float)
        distances = np.linalg.norm(positions - point, axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] > tolerance_mm:
            raise ValueError(
                f"NIfTI slice {z} does not map to a DICOM slice within tolerance."
            )
        mapping.append(series.slices[idx])
    if len({s.sop_instance_uid for s in mapping}) != len(mapping):
        raise ValueError("Image-to-DICOM mapping is not one-to-one.")
    return mapping


def _slice_from_dataset(path: Path, dataset) -> DicomSliceRef:
    orientation = _float_tuple(getattr(dataset, "ImageOrientationPatient", None))
    position = _float_tuple(getattr(dataset, "ImagePositionPatient", None))
    coordinate = None
    if orientation is not None and position is not None:
        row = np.array(orientation[:3], dtype=float)
        col = np.array(orientation[3:], dtype=float)
        normal = np.cross(row, col)
        coordinate = float(np.dot(np.array(position, dtype=float), normal))
    return DicomSliceRef(
        path=path,
        sop_class_uid=str(getattr(dataset, "SOPClassUID", "")),
        sop_instance_uid=str(getattr(dataset, "SOPInstanceUID", "")),
        study_instance_uid=str(getattr(dataset, "StudyInstanceUID", "")),
        series_instance_uid=str(getattr(dataset, "SeriesInstanceUID", "")),
        frame_of_reference_uid=_optional_str(getattr(dataset, "FrameOfReferenceUID", None)),
        modality=_optional_str(getattr(dataset, "Modality", None)),
        rows=int(getattr(dataset, "Rows")),
        columns=int(getattr(dataset, "Columns")),
        pixel_spacing=_float_tuple(getattr(dataset, "PixelSpacing", None)),
        orientation=orientation,
        position=position,
        slice_thickness=_optional_float(getattr(dataset, "SliceThickness", None)),
        instance_number=_optional_int(getattr(dataset, "InstanceNumber", None)),
        slice_coordinate=coordinate,
    )


def _float_tuple(value) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(float(v) for v in value)
    except Exception:
        return None


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value) -> int | None:
    if value is None:
        return None


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _hash_uid(uid: str | None) -> str | None:
    if uid is None:
        return None
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()[:12]
