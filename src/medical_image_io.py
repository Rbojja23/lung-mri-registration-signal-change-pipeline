from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import SimpleITK as sitk

from project_paths import DEFAULT_RAW_DATA_DIR, display_path, project_root

try:
    import pydicom
except ImportError:  # pragma: no cover - exercised only in incomplete envs.
    pydicom = None


SUPPORTED_MODALITIES = {"CT", "MR"}
NIFTI_EXTENSIONS = (".nii", ".nii.gz")
DICOM_EXTENSIONS = (".dcm", ".dicom")


@dataclass(frozen=True)
class ImageGeometry:
    """Geometry summary using SimpleITK's x, y, z image-index convention."""

    size_xyz: tuple[int, int, int]
    array_shape_zyx: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]


@dataclass(frozen=True)
class LoadedMedicalImage:
    """A loaded CT or MR image with non-identifying provenance.

    SimpleITK image indices are ordered x, y, z. Arrays returned by
    sitk.GetArrayFromImage are ordered z, y, x.
    """

    image: sitk.Image
    modality: str
    input_type: str
    source_path: Path
    size_xyz: tuple[int, int, int]
    array_shape_zyx: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]
    dicom_series_id: str | None = None
    loader_warnings: tuple[str, ...] = ()

    def safe_summary(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "input_type": self.input_type,
            "source_path": display_path(self.source_path),
            "image_size_xyz": list(self.size_xyz),
            "numpy_array_shape_zyx": list(self.array_shape_zyx),
            "spacing_xyz_mm": list(self.spacing_xyz),
            "origin_xyz_mm": list(self.origin_xyz),
            "direction": list(self.direction),
            "dicom_series_id": self.dicom_series_id,
            "coordinate_conventions": {
                "simpleitk_index_order": "x, y, z",
                "numpy_array_order": "z, y, x",
                "physical_points": "millimeters in the image patient/world coordinate frame",
            },
            "loader_warnings": list(self.loader_warnings),
        }


@dataclass(frozen=True)
class DicomFileInfo:
    path: Path
    series_instance_uid: str | None
    modality: str | None
    rows: int | None
    columns: int | None
    image_orientation_patient: tuple[float, ...] | None
    image_position_patient: tuple[float, ...] | None
    instance_number: int | None


def summarize_geometry(image: sitk.Image) -> ImageGeometry:
    """Return size/spacing/origin/direction without changing image data.

    SimpleITK reports image size, spacing, origin, and indices in x, y, z
    order. The NumPy array shape is z, y, x because sitk.GetArrayFromImage
    reverses axis order for NumPy.
    """

    return ImageGeometry(
        size_xyz=tuple(int(v) for v in image.GetSize()),
        array_shape_zyx=tuple(int(v) for v in sitk.GetArrayViewFromImage(image).shape),
        spacing_xyz=tuple(float(v) for v in image.GetSpacing()),
        origin_xyz=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
    )


def sitk_index_xyz_to_numpy_index_zyx(index_xyz: Sequence[int]) -> tuple[int, int, int]:
    """Convert a SimpleITK voxel index (x, y, z) to a NumPy index (z, y, x)."""

    if len(index_xyz) != 3:
        raise ValueError("Expected a 3D index in x, y, z order.")
    x, y, z = index_xyz
    return int(z), int(y), int(x)


def numpy_index_zyx_to_sitk_index_xyz(index_zyx: Sequence[int]) -> tuple[int, int, int]:
    """Convert a NumPy index (z, y, x) to a SimpleITK voxel index (x, y, z)."""

    if len(index_zyx) != 3:
        raise ValueError("Expected a 3D index in z, y, x order.")
    z, y, x = index_zyx
    return int(x), int(y), int(z)


def voxel_index_xyz_to_physical_point(
    image: sitk.Image, index_xyz: Sequence[int]
) -> tuple[float, float, float]:
    """Convert a SimpleITK voxel index (x, y, z) to a physical point in mm."""

    return tuple(float(v) for v in image.TransformIndexToPhysicalPoint(tuple(index_xyz)))


def physical_point_to_voxel_index_xyz(
    image: sitk.Image, point_xyz_mm: Sequence[float]
) -> tuple[int, int, int]:
    """Convert a physical point in mm to the nearest SimpleITK index x, y, z."""

    return tuple(int(v) for v in image.TransformPhysicalPointToIndex(tuple(point_xyz_mm)))


def compare_image_geometry(
    image_a: sitk.Image,
    image_b: sitk.Image,
    *,
    spacing_tolerance: float = 1e-5,
    origin_tolerance: float = 1e-5,
    direction_tolerance: float = 1e-5,
) -> list[str]:
    """Return non-identifying geometry mismatch messages for two images."""

    geom_a = summarize_geometry(image_a)
    geom_b = summarize_geometry(image_b)
    mismatches: list[str] = []

    if geom_a.size_xyz != geom_b.size_xyz:
        mismatches.append(f"size mismatch: {geom_a.size_xyz} != {geom_b.size_xyz}")
    if not np.allclose(
        geom_a.spacing_xyz, geom_b.spacing_xyz, atol=spacing_tolerance, rtol=0
    ):
        mismatches.append(
            f"spacing mismatch: {geom_a.spacing_xyz} != {geom_b.spacing_xyz}"
        )
    if not np.allclose(geom_a.origin_xyz, geom_b.origin_xyz, atol=origin_tolerance, rtol=0):
        mismatches.append(f"origin mismatch: {geom_a.origin_xyz} != {geom_b.origin_xyz}")
    if not np.allclose(
        geom_a.direction, geom_b.direction, atol=direction_tolerance, rtol=0
    ):
        mismatches.append(f"direction mismatch: {geom_a.direction} != {geom_b.direction}")

    return mismatches


def geometry_matches(image_a: sitk.Image, image_b: sitk.Image, **kwargs: Any) -> bool:
    """Return True when size, spacing, origin, and direction match."""

    return not compare_image_geometry(image_a, image_b, **kwargs)


def infer_input_type(path: str | Path) -> str:
    candidate = Path(path)
    name = candidate.name.lower()
    if candidate.is_dir():
        return "dicom"
    if name.endswith(NIFTI_EXTENSIONS):
        return "nifti"
    raise ValueError("Could not infer input type. Use --input-type dicom or nifti.")


def normalize_modality(
    modality: str | None,
    *,
    allow_auto: bool = False,
    context: str = "modality",
) -> str:
    if modality is None:
        if allow_auto:
            return "auto"
        raise ValueError(f"{context} is required.")

    normalized = modality.strip().upper()
    if normalized == "MRI":
        normalized = "MR"
    if normalized == "AUTO" and allow_auto:
        return "auto"
    if normalized not in SUPPORTED_MODALITIES:
        raise ValueError(f"Unsupported {context}: {modality}. Expected CT or MR.")
    return normalized


def load_medical_image(
    input_path: str | Path,
    *,
    input_type: str = "auto",
    modality: str = "auto",
    series_id: str | None = None,
    recursive: bool = False,
    force_modality: bool = False,
) -> LoadedMedicalImage:
    """Load a CT/MR DICOM directory or NIfTI file without altering image data."""

    path = Path(input_path).expanduser().resolve(strict=False)
    resolved_input_type = input_type.lower()
    if resolved_input_type == "auto":
        resolved_input_type = infer_input_type(path)
    if resolved_input_type == "nifti":
        return load_nifti(path, modality=modality)
    if resolved_input_type == "dicom":
        return load_dicom_directory(
            path,
            modality=modality,
            series_id=series_id,
            recursive=recursive,
            force_modality=force_modality,
        )
    raise ValueError("input_type must be one of auto, dicom, or nifti.")


def load_nifti(input_path: str | Path, *, modality: str) -> LoadedMedicalImage:
    """Load .nii or .nii.gz with an explicit CT/MR modality.

    NIfTI headers do not reliably encode clinical modality. This loader never
    guesses from voxel values and does not reorient, resample, or normalize.
    """

    path = Path(input_path).expanduser().resolve(strict=False)
    if not path.name.lower().endswith(NIFTI_EXTENSIONS):
        raise ValueError("NIfTI input must end with .nii or .nii.gz.")
    normalized_modality = normalize_modality(modality, allow_auto=False)
    image = sitk.ReadImage(str(path))
    geom = summarize_geometry(image)
    return LoadedMedicalImage(
        image=image,
        modality=normalized_modality,
        input_type="nifti",
        source_path=path,
        size_xyz=geom.size_xyz,
        array_shape_zyx=geom.array_shape_zyx,
        spacing_xyz=geom.spacing_xyz,
        origin_xyz=geom.origin_xyz,
        direction=geom.direction,
    )


def load_dicom_directory(
    input_dir: str | Path,
    *,
    modality: str = "auto",
    series_id: str | None = None,
    recursive: bool = False,
    force_modality: bool = False,
) -> LoadedMedicalImage:
    """Load a DICOM image series from a directory.

    The primary loader uses SimpleITK/GDCM series discovery. If GDCM finds no
    series, a conservative fallback groups direct DICOM files by structural,
    non-identifying tags and orders slices by spatial position, instance number,
    or filename as a last resort.
    """

    path = Path(input_dir).expanduser().resolve(strict=False)
    if not path.is_dir():
        raise ValueError("DICOM input must be a directory.")

    series_ids: Sequence[str] = ()
    if not recursive:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or ()

    if series_ids:
        return _load_dicom_with_gdcm(
            path,
            series_ids=series_ids,
            modality=modality,
            series_id=series_id,
            force_modality=force_modality,
        )

    return _load_dicom_with_metadata_fallback(
        path,
        modality=modality,
        series_id=series_id,
        recursive=recursive,
        force_modality=force_modality,
    )


def _load_dicom_with_gdcm(
    input_dir: Path,
    *,
    series_ids: Sequence[str],
    modality: str,
    series_id: str | None,
    force_modality: bool,
) -> LoadedMedicalImage:
    counts = {
        current_id: len(
            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(input_dir), current_id)
        )
        for current_id in series_ids
    }

    if series_id is None:
        if len(series_ids) != 1:
            raise ValueError(
                "Multiple DICOM series found. Provide --series-id. "
                f"Available non-identifying series counts: {counts}"
            )
        selected_series_id = series_ids[0]
    else:
        if series_id not in series_ids:
            raise ValueError(
                f"Requested series ID was not found. Available series IDs: {list(series_ids)}"
            )
        selected_series_id = series_id

    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(input_dir), selected_series_id
    )
    detected_modality = _detect_dicom_modality(Path(file_names[0]))
    normalized_modality = _resolve_detected_modality(
        requested_modality=modality,
        detected_modality=detected_modality,
        force_modality=force_modality,
    )

    image = _read_dicom_files(file_names)
    geom = summarize_geometry(image)
    return LoadedMedicalImage(
        image=image,
        modality=normalized_modality,
        input_type="dicom",
        source_path=input_dir,
        size_xyz=geom.size_xyz,
        array_shape_zyx=geom.array_shape_zyx,
        spacing_xyz=geom.spacing_xyz,
        origin_xyz=geom.origin_xyz,
        direction=geom.direction,
        dicom_series_id=selected_series_id,
    )


def _load_dicom_with_metadata_fallback(
    input_dir: Path,
    *,
    modality: str,
    series_id: str | None,
    recursive: bool,
    force_modality: bool,
) -> LoadedMedicalImage:
    files = _find_dicom_files(input_dir, recursive=recursive)
    if not files:
        raise ValueError("No DICOM files found in the supplied directory.")

    infos = [_read_dicom_file_info(path) for path in files]
    groups = _group_by_series_uid(infos)

    if series_id is None:
        if len(groups) != 1:
            counts = {key: len(value) for key, value in groups.items()}
            raise ValueError(
                "Multiple incompatible DICOM series found. Provide --series-id. "
                f"Available non-identifying series counts: {counts}"
            )
        selected_series_id, selected_infos = next(iter(groups.items()))
    else:
        if series_id not in groups:
            raise ValueError(
                f"Requested series ID was not found. Available series IDs: {list(groups)}"
            )
        selected_series_id = series_id
        selected_infos = groups[series_id]

    warnings = ["GDCM series discovery found no series; used metadata fallback loader."]
    _validate_dicom_infos(selected_infos)
    detected_modality = _single_detected_modality(selected_infos)
    normalized_modality = _resolve_detected_modality(
        requested_modality=modality,
        detected_modality=detected_modality,
        force_modality=force_modality,
    )

    ordered_infos, sort_warning = sort_dicom_files_for_loading(selected_infos)
    if sort_warning:
        warnings.append(sort_warning)

    image = _read_dicom_files([str(info.path) for info in ordered_infos])
    geom = summarize_geometry(image)
    return LoadedMedicalImage(
        image=image,
        modality=normalized_modality,
        input_type="dicom",
        source_path=input_dir,
        size_xyz=geom.size_xyz,
        array_shape_zyx=geom.array_shape_zyx,
        spacing_xyz=geom.spacing_xyz,
        origin_xyz=geom.origin_xyz,
        direction=geom.direction,
        dicom_series_id=selected_series_id if selected_series_id != "__missing__" else None,
        loader_warnings=tuple(warnings),
    )


def sort_dicom_files_for_loading(
    infos: Sequence[DicomFileInfo],
) -> tuple[list[DicomFileInfo], str | None]:
    """Sort DICOM slices for loading using non-identifying structural metadata."""

    if all(
        info.image_orientation_patient is not None
        and info.image_position_patient is not None
        for info in infos
    ):
        orientation = np.array(infos[0].image_orientation_patient, dtype=float)
        row = orientation[:3]
        col = orientation[3:]
        normal = np.cross(row, col)
        if np.linalg.norm(normal) > 0:
            ordered = sorted(
                infos,
                key=lambda info: float(
                    np.dot(np.array(info.image_position_patient, dtype=float), normal)
                ),
            )
            return ordered, None

    if all(info.instance_number is not None for info in infos):
        return sorted(infos, key=lambda info: int(info.instance_number)), None

    return (
        sorted(infos, key=lambda info: info.path.name),
        "Slice ordering used deterministic filename order because spatial metadata "
        "and InstanceNumber were incomplete.",
    )


def save_summary_json(
    loaded: LoadedMedicalImage,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    path = _prepare_output_path(output_path, overwrite=overwrite)
    path.write_text(json.dumps(loaded.safe_summary(), indent=2) + "\n")
    return path


def write_nifti(
    loaded: LoadedMedicalImage,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    path = _prepare_output_path(output_path, overwrite=overwrite)
    if not path.name.lower().endswith(NIFTI_EXTENSIONS):
        raise ValueError("Converted NIfTI output must end with .nii or .nii.gz.")
    sitk.WriteImage(loaded.image, str(path))
    return path


def _prepare_output_path(output_path: str | Path, *, overwrite: bool) -> Path:
    path = Path(output_path).expanduser().resolve(strict=False)
    _reject_raw_data_write(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {display_path(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _reject_raw_data_write(path: Path) -> None:
    raw_root = DEFAULT_RAW_DATA_DIR.resolve(strict=False)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(raw_root)
    except ValueError:
        return
    raise ValueError("Refusing to write outputs under data/raw/.")


def _read_dicom_files(file_names: Iterable[str]) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(name) for name in file_names])
    return reader.Execute()


def _find_dicom_files(input_dir: Path, *, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.name.lower().endswith(DICOM_EXTENSIONS)
    )


def _read_dicom_file_info(path: Path) -> DicomFileInfo:
    if pydicom is None:
        raise ImportError("pydicom is required for fallback DICOM metadata loading.")

    specific_tags = [
        "SeriesInstanceUID",
        "Modality",
        "Rows",
        "Columns",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "InstanceNumber",
    ]
    dataset = pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
        force=True,
        specific_tags=specific_tags,
    )
    return DicomFileInfo(
        path=path,
        series_instance_uid=_optional_str(getattr(dataset, "SeriesInstanceUID", None)),
        modality=_optional_str(getattr(dataset, "Modality", None)),
        rows=_optional_int(getattr(dataset, "Rows", None)),
        columns=_optional_int(getattr(dataset, "Columns", None)),
        image_orientation_patient=_optional_float_tuple(
            getattr(dataset, "ImageOrientationPatient", None)
        ),
        image_position_patient=_optional_float_tuple(
            getattr(dataset, "ImagePositionPatient", None)
        ),
        instance_number=_optional_int(getattr(dataset, "InstanceNumber", None)),
    )


def _detect_dicom_modality(path: Path) -> str | None:
    if pydicom is None:
        return None
    dataset = pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
        force=True,
        specific_tags=["Modality"],
    )
    return normalize_modality(
        _optional_str(getattr(dataset, "Modality", None)),
        allow_auto=False,
        context="DICOM Modality",
    )


def _resolve_detected_modality(
    *,
    requested_modality: str,
    detected_modality: str | None,
    force_modality: bool,
) -> str:
    requested = normalize_modality(requested_modality, allow_auto=True)
    if detected_modality is None:
        if requested == "auto":
            raise ValueError("DICOM Modality tag is missing; provide --modality CT or MR.")
        return normalize_modality(requested)

    detected = normalize_modality(detected_modality, allow_auto=False, context="DICOM Modality")
    if requested == "auto":
        return detected
    requested_normalized = normalize_modality(requested, allow_auto=False)
    if requested_normalized != detected and not force_modality:
        raise ValueError(
            "Requested modality conflicts with DICOM metadata. "
            "Use --force-modality only if this is intentional."
        )
    return requested_normalized


def _group_by_series_uid(infos: Sequence[DicomFileInfo]) -> dict[str, list[DicomFileInfo]]:
    groups: dict[str, list[DicomFileInfo]] = {}
    for info in infos:
        key = info.series_instance_uid or "__missing__"
        groups.setdefault(key, []).append(info)
    return groups


def _validate_dicom_infos(infos: Sequence[DicomFileInfo]) -> None:
    if not infos:
        raise ValueError("No DICOM files selected for loading.")

    rows = {info.rows for info in infos}
    columns = {info.columns for info in infos}
    modalities = {info.modality for info in infos}
    series_ids = {info.series_instance_uid for info in infos}

    if len(series_ids) > 1:
        raise ValueError("Selected DICOM files contain multiple SeriesInstanceUID values.")
    if len(rows) != 1 or None in rows:
        raise ValueError("Selected DICOM files have incompatible or missing row counts.")
    if len(columns) != 1 or None in columns:
        raise ValueError("Selected DICOM files have incompatible or missing column counts.")
    if len(modalities) != 1:
        raise ValueError("Selected DICOM files have incompatible modalities.")

    orientations = [
        info.image_orientation_patient
        for info in infos
        if info.image_orientation_patient is not None
    ]
    if orientations and not all(np.allclose(orientations[0], value) for value in orientations):
        raise ValueError("Selected DICOM files have incompatible image orientations.")


def _single_detected_modality(infos: Sequence[DicomFileInfo]) -> str | None:
    modalities = {info.modality for info in infos}
    if len(modalities) != 1:
        raise ValueError("Selected DICOM files have incompatible modalities.")
    modality = next(iter(modalities))
    if modality is None:
        return None
    return normalize_modality(modality, allow_auto=False, context="DICOM Modality")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def image_summary_without_pixels(loaded: LoadedMedicalImage) -> dict[str, Any]:
    """Return the safe JSON-ready summary for callers that do not write files."""

    return loaded.safe_summary()
