from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from export_registration_difference_dicom import main as cli_main  # noqa: E402
from rgb_secondary_capture_export import (  # noqa: E402
    MANIFEST_NAME,
    PATIENT_IDS,
    export_rgb_secondary_capture,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(
    path: Path,
    array: np.ndarray,
    *,
    spacing=(1.2, 1.5, 2.0),
    origin=(10.0, 20.0, 30.0),
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk.WriteImage(image, str(path))
    return path


def _fixtures(tmp_path: Path):
    inputs = {}
    base = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
    for number, offset in ((1, 2.0), (2, 2.0)):
        case = f"case_{number}"
        fixed = _write_image(tmp_path / case / "fixed.nii.gz", base)
        registered = _write_image(tmp_path / case / "registered.nii.gz", base + offset)
        inputs[case] = {"fixed": fixed, "registered": registered}
    report = {
        "analysis_type": "Case 1-versus-Case 2 equal-window registration difference visualization",
        "completion_status": "complete",
        "inputs": {
            case: {
                "analytical_images": {
                    role: {"path": "[external path]", "sha256": _sha(path)}
                    for role, path in values.items()
                }
            }
            for case, values in inputs.items()
        },
        "shared_difference_window": {
            "pooled_volume_order": [
                "case_1_before", "case_1_after", "case_2_before", "case_2_after"
            ],
            "vmin": 0.0,
            "vmax": 10.0,
            "identical_for_every_difference_panel": True,
            "panel_limits": {
                "overlay": {
                    "case_1_after": {"vmin": 0.0, "vmax": 10.0},
                    "case_2_after": {"vmin": 0.0, "vmax": 10.0},
                }
            },
        },
        "display": {
            "colormaps": {"overlay_difference": "turbo"},
            "fixed_anatomy_limits": {"vmin": 0.0, "vmax": 100.0},
        },
    }
    report_path = tmp_path / "shared.json"
    report_path.write_text(json.dumps(report))
    return inputs, report_path


def _export(tmp_path: Path, **kwargs):
    inputs, report = _fixtures(tmp_path)
    output = tmp_path / "dicom"
    result = export_rgb_secondary_capture(
        case_1_fixed=inputs["case_1"]["fixed"],
        case_1_registered=inputs["case_1"]["registered"],
        case_2_fixed=inputs["case_2"]["fixed"],
        case_2_registered=inputs["case_2"]["registered"],
        shared_window_report=report,
        output_dir=output,
        **kwargs,
    )
    return result, inputs, report, output


def test_dicom_series_are_readable_rgb_unique_and_correctly_grouped(tmp_path):
    manifest, _, _, output = _export(tmp_path)
    series_by_case = {}
    studies = set()
    all_sops = set()
    for case in ("case_1", "case_2"):
        datasets = [pydicom.dcmread(str(path)) for path in sorted((output / case).glob("*.dcm"))]
        assert len(datasets) == 3 == manifest["slice_count"][case]
        assert [int(ds.InstanceNumber) for ds in datasets] == [1, 2, 3]
        assert all(str(ds.SOPClassUID) == str(SecondaryCaptureImageStorage) for ds in datasets)
        assert all(str(ds.file_meta.TransferSyntaxUID) == str(ExplicitVRLittleEndian) for ds in datasets)
        assert all(ds.PhotometricInterpretation == "RGB" and ds.PlanarConfiguration == 0 for ds in datasets)
        assert all(ds.pixel_array.shape == (4, 6, 3) for ds in datasets)
        assert all(len(ds.PixelData) == 4 * 6 * 3 for ds in datasets)
        sops = {str(ds.SOPInstanceUID) for ds in datasets}
        assert len(sops) == len(datasets)
        assert not all_sops.intersection(sops)
        all_sops.update(sops)
        series = {str(ds.SeriesInstanceUID) for ds in datasets}
        assert len(series) == 1
        series_by_case[case] = next(iter(series))
        studies.update(str(ds.StudyInstanceUID) for ds in datasets)
    assert series_by_case["case_1"] != series_by_case["case_2"]
    assert len(studies) == 2
    assert manifest["validation"]["status"] == "passed"


def test_spatial_tags_slice_order_and_shared_scaling_are_consistent(tmp_path):
    manifest, _, _, output = _export(tmp_path)
    decoded = {}
    for case in ("case_1", "case_2"):
        datasets = [pydicom.dcmread(str(path)) for path in sorted((output / case).glob("*.dcm"))]
        assert all(
            np.allclose(np.asarray(ds.PixelSpacing, dtype=float), [1.5, 1.2], atol=1e-6)
            for ds in datasets
        )
        assert all(
            np.allclose(
                np.asarray(ds.ImageOrientationPatient, dtype=float),
                [1, 0, 0, 0, 1, 0],
                atol=1e-6,
            )
            for ds in datasets
        )
        positions = [list(np.asarray(ds.ImagePositionPatient, dtype=float)) for ds in datasets]
        assert positions == [[10, 20, 30], [10, 20, 32], [10, 20, 34]]
        for ds in datasets:
            for values in (
                ds.PixelSpacing,
                ds.ImageOrientationPatient,
                ds.ImagePositionPatient,
                [ds.SliceThickness],
                [ds.SpacingBetweenSlices],
            ):
                assert all(len(str(value)) <= 16 for value in values)
        decoded[case] = np.stack([ds.pixel_array for ds in datasets])
    assert np.array_equal(decoded["case_1"], decoded["case_2"])
    scaling = manifest["validation"]["identical_shared_scaling"]
    assert scaling == {"passed": True, "vmin": 0.0, "vmax": 10.0}


def test_only_anonymized_research_identity_and_no_reference_claim(tmp_path):
    manifest, _, _, output = _export(tmp_path)
    for case in ("case_1", "case_2"):
        dataset = pydicom.dcmread(str(next((output / case).glob("*.dcm"))))
        assert str(dataset.PatientName) == "ANONYMIZED^RESEARCH"
        assert dataset.PatientID == PATIENT_IDS[case]
        assert dataset.PatientIdentityRemoved == "YES"
        assert dataset.BurnedInAnnotation == "NO"
        assert dataset.ImageType == ["DERIVED", "SECONDARY"]
        assert not hasattr(dataset, "SourceImageSequence")
        assert not hasattr(dataset, "ReferencedSeriesSequence")
        assert "no original clinical DICOM reference relationship is claimed" in dataset.DerivationDescription
        assert str(dataset.ReferringPhysicianName) == ""
        for keyword in ("InstitutionName", "OperatorsName"):
            assert not hasattr(dataset, keyword)
    assert manifest["object_classification"]["is_parametric_map"] is False
    assert manifest["display"]["quantitative_values_recoverable_from_rgb"] is False


def test_invalid_geometry_is_refused_before_output(tmp_path):
    inputs, report = _fixtures(tmp_path)
    path = inputs["case_2"]["registered"]
    image = sitk.ReadImage(str(path))
    image.SetSpacing((1.2, 1.5, 2.1))
    sitk.WriteImage(image, str(path))
    output = tmp_path / "dicom"
    with pytest.raises(ValueError, match="geometry mismatch"):
        export_rgb_secondary_capture(
            case_1_fixed=inputs["case_1"]["fixed"],
            case_1_registered=inputs["case_1"]["registered"],
            case_2_fixed=inputs["case_2"]["fixed"],
            case_2_registered=path,
            shared_window_report=report,
            output_dir=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda report: report.update(completion_status="pending"), "not complete"),
        (lambda report: report["inputs"].pop("case_2"), "Case 1 and Case 2"),
        (lambda report: report["shared_difference_window"].update(vmax=-1), "invalid.*limits"),
        (lambda report: report["display"]["colormaps"].update(overlay_difference="magma"), "incompatible"),
    ],
)
def test_incomplete_or_incompatible_report_is_refused(tmp_path, mutation, match):
    inputs, report_path = _fixtures(tmp_path)
    report = json.loads(report_path.read_text())
    mutation(report)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        export_rgb_secondary_capture(
            case_1_fixed=inputs["case_1"]["fixed"],
            case_1_registered=inputs["case_1"]["registered"],
            case_2_fixed=inputs["case_2"]["fixed"],
            case_2_registered=inputs["case_2"]["registered"],
            shared_window_report=report_path,
            output_dir=tmp_path / "dicom",
        )


def test_source_niftis_unchanged_dry_run_and_overwrite_protection(tmp_path, capsys):
    inputs, report = _fixtures(tmp_path)
    before = {path: _sha(path) for values in inputs.values() for path in values.values()}
    output = tmp_path / "dicom"
    args = [
        "--case-1-fixed", str(inputs["case_1"]["fixed"]),
        "--case-1-registered", str(inputs["case_1"]["registered"]),
        "--case-2-fixed", str(inputs["case_2"]["fixed"]),
        "--case-2-registered", str(inputs["case_2"]["registered"]),
        "--shared-window-report", str(report),
        "--output-dir", str(output),
    ]
    assert cli_main([*args, "--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out
    assert not output.exists()
    manifest = export_rgb_secondary_capture(
        case_1_fixed=inputs["case_1"]["fixed"],
        case_1_registered=inputs["case_1"]["registered"],
        case_2_fixed=inputs["case_2"]["fixed"],
        case_2_registered=inputs["case_2"]["registered"],
        shared_window_report=report,
        output_dir=output,
    )
    assert manifest["source_immutability"]["unchanged"] is True
    assert all(_sha(path) == digest for path, digest in before.items())
    with pytest.raises(FileExistsError, match="--overwrite"):
        export_rgb_secondary_capture(
            case_1_fixed=inputs["case_1"]["fixed"],
            case_1_registered=inputs["case_1"]["registered"],
            case_2_fixed=inputs["case_2"]["fixed"],
            case_2_registered=inputs["case_2"]["registered"],
            shared_window_report=report,
            output_dir=output,
        )
