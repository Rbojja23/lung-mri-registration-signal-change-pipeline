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


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from export_fractional_proxy_dicom import main as cli_main  # noqa: E402
from fractional_proxy_rgb_secondary_capture import (  # noqa: E402
    FORMULA,
    MANIFEST_NAME,
    export_fractional_proxy_rgb_secondary_capture,
    render_fractional_proxy_rgb_slice,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(array))
    image.SetSpacing((1.2, 1.5, 2.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image, str(path))
    return path


def _fixtures(tmp_path):
    outputs = {"cases": {}}
    inputs = {}
    output_hashes = {}
    all_sources = []
    shape = (3, 5, 7)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 20
    mask = np.zeros(shape, np.uint8)
    mask[:, 1:4, 1:6] = 1
    for number in (1, 2):
        case = f"case_{number}"
        case_dir = tmp_path / case
        fixed = _write(case_dir / "fixed.nii.gz", base + number)
        lung = _write(case_dir / "lung.nii.gz", mask)
        inputs[case] = {
            "fixed": {"path": str(fixed), "sha256": _sha(fixed)},
            "exact_fixed_space_lung_mask": {"path": str(lung), "sha256": _sha(lung)},
        }
        outputs["cases"][case] = {}
        all_sources.extend([fixed, lung])
        for method_index, variant in enumerate(("native", "mean_filtered", "gaussian_filtered"), 1):
            values = np.full(shape, np.nan, np.float32)
            values[mask.astype(bool)] = (-0.4 + 0.2 * method_index) * (1 if number == 1 else -1)
            validity = mask.copy()
            proxy = _write(case_dir / f"{variant}.nii.gz", values)
            valid = _write(case_dir / f"{variant}_validity.nii.gz", validity)
            outputs["cases"][case][variant] = str(proxy)
            outputs["cases"][case][f"{variant}_validity"] = str(valid)
            output_hashes[str(proxy)] = _sha(proxy)
            output_hashes[str(valid)] = _sha(valid)
            all_sources.extend([proxy, valid])
    report = {
        "analysis_type": "Exploratory registered-space fractional MRI signal-change proxy",
        "completion_status": "complete",
        "formula": {"expression": FORMULA},
        "inputs": inputs,
        "outputs": outputs,
        "output_checksums_sha256": output_hashes,
        "display": {
            "colormap": "RdBu_r",
            "anatomy_limits": [0.0, 150.0],
            "shared_symmetric_window": {
                "pooled_order": [
                    "case_1_native", "case_1_mean_filtered", "case_1_gaussian_filtered",
                    "case_2_native", "case_2_mean_filtered", "case_2_gaussian_filtered",
                ],
                "vmin": -0.6633644104003906,
                "vcenter": 0.0,
                "vmax": 0.6633644104003906,
                "identical_for_all_proxy_panels": True,
            },
        },
    }
    report_path = tmp_path / "fractional_proxy_report.json"
    report_path.write_text(json.dumps(report))
    all_sources.append(report_path)
    return report_path, all_sources


def _export(tmp_path, **kwargs):
    report, sources = _fixtures(tmp_path)
    output = tmp_path / "dicom"
    result = export_fractional_proxy_rgb_secondary_capture(
        comparison_report=report, output_dir=output, **kwargs
    )
    return result, report, sources, output


def test_renderer_uses_fixed_grayscale_and_confines_proxy_color_to_valid_lung():
    fixed = np.array([[0, 25, 50], [75, 100, 50], [25, 50, 75]], np.float32)
    proxy = np.full((3, 3), 0.5, np.float32)
    lung = np.zeros((3, 3), bool)
    lung[1, 1] = True
    valid = lung.copy()
    rgb = render_fractional_proxy_rgb_slice(
        fixed, proxy, valid, lung, vmin=-1, vmax=1,
        anatomy_limits=(0, 100), opacity=1.0, draw_contour=False,
    )
    gray = np.rint(np.clip(fixed / 100, 0, 1) * 255).astype(np.uint8)
    expected_outside = np.repeat(gray[..., None], 3, axis=2)[~lung]
    assert np.array_equal(rgb[~lung], expected_outside)
    assert not np.array_equal(rgb[1, 1], [gray[1, 1]] * 3)


def test_six_series_source_selection_case_separation_readability_uids_and_geometry(tmp_path):
    manifest, _, _, output = _export(tmp_path, draw_contour=False)
    all_series = set()
    all_sops = set()
    representative = {}
    for case in ("case_1", "case_2"):
        for method in ("native", "mean", "gaussian"):
            datasets = [pydicom.dcmread(str(path)) for path in sorted((output / case / method).glob("*.dcm"))]
            assert len(datasets) == 3
            assert [int(ds.InstanceNumber) for ds in datasets] == [1, 2, 3]
            assert [list(np.asarray(ds.ImagePositionPatient, float)) for ds in datasets] == [
                [10, 20, 30], [10, 20, 32], [10, 20, 34]
            ]
            assert all(np.allclose(np.asarray(ds.PixelSpacing, float), [1.5, 1.2]) for ds in datasets)
            assert all(str(ds.SOPClassUID) == str(SecondaryCaptureImageStorage) for ds in datasets)
            assert all(str(ds.file_meta.TransferSyntaxUID) == str(ExplicitVRLittleEndian) for ds in datasets)
            assert all(ds.pixel_array.shape == (5, 7, 3) for ds in datasets)
            series = {str(ds.SeriesInstanceUID) for ds in datasets}
            sops = {str(ds.SOPInstanceUID) for ds in datasets}
            assert len(series) == 1 and not all_series.intersection(series)
            assert len(sops) == 3 and not all_sops.intersection(sops)
            all_series.update(series)
            all_sops.update(sops)
            assert case.replace("_", " ").title() in datasets[0].SeriesDescription
            assert method.lower() in datasets[0].SeriesDescription.lower() or method == "mean"
            representative[(case, method)] = datasets[1].pixel_array[2, 3].copy()
    assert len(all_series) == 6 and len(all_sops) == 18
    assert not np.array_equal(representative[("case_1", "native")], representative[("case_1", "gaussian")])
    assert not np.array_equal(representative[("case_1", "native")], representative[("case_2", "native")])
    assert manifest["validation"]["status"] == "passed"


def test_manifest_records_fixed_background_no_difference_shared_limits_and_sources(tmp_path):
    manifest, _, _, output = _export(tmp_path)
    saved = json.loads((output / MANIFEST_NAME).read_text())
    assert saved["source_analytical_method"]["formula"] == FORMULA
    assert saved["display"]["fixed_anatomy_used_as_background"] is True
    assert saved["display"]["difference_image_used_anywhere"] is False
    assert saved["display"]["colormap"] == "RdBu_r"
    assert saved["display"]["shared_signed_limits"] == {
        "vmin": -0.6633644104003906, "vcenter": 0.0, "vmax": 0.6633644104003906
    }
    assert saved["display"]["identical_policy_for_all_six_series"] is True
    assert set(saved["outputs"]["series"]["case_1"]) == {"native", "mean", "gaussian"}
    assert saved["object_classification"]["quantitative_parametric_map"] is False
    assert saved["spatial_tags"]["clinical_reference_relationship_claimed"] is False
    assert saved["source_immutability"]["unchanged"] is True
    assert manifest["completion_status"] == "complete"


def test_dry_run_source_immutability_cli_and_overwrite_protection(tmp_path, capsys):
    report, sources = _fixtures(tmp_path)
    output = tmp_path / "dicom"
    before = {path: _sha(path) for path in sources}
    args = ["--comparison-report", str(report), "--output-dir", str(output)]
    assert cli_main([*args, "--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out
    assert not output.exists()
    export_fractional_proxy_rgb_secondary_capture(comparison_report=report, output_dir=output)
    assert {path: _sha(path) for path in sources} == before
    with pytest.raises(FileExistsError, match="--overwrite"):
        export_fractional_proxy_rgb_secondary_capture(comparison_report=report, output_dir=output)
    result = export_fractional_proxy_rgb_secondary_capture(
        comparison_report=report, output_dir=output, overwrite=True
    )
    assert result["completion_status"] == "complete"
