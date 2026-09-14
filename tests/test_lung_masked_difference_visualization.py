from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from create_lung_masked_difference_visualization import main as figure_cli_main  # noqa: E402
from export_lung_masked_difference_dicom import main as dicom_cli_main  # noqa: E402
from lung_masked_difference_visualization import (  # noqa: E402
    OUTPUT_NAMES,
    create_lung_masked_difference_figures,
    render_lung_masked_turbo_rgb,
)
from lung_masked_rgb_secondary_capture_export import (  # noqa: E402
    MANIFEST_NAME,
    export_lung_masked_rgb_secondary_capture,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path, array, *, spacing=(1.2, 1.5, 2.0), origin=(10.0, 20.0, 30.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(array))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    sitk.WriteImage(image, str(path))
    return path


def _fixtures(tmp_path: Path):
    case_dirs = {}
    masks = {}
    base = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
    for number, before, after in ((1, 4.0, 2.0), (2, 8.0, 3.0)):
        case = f"case_{number}"
        case_dir = tmp_path / case
        case_dirs[case] = case_dir
        full = case_dir / "full"
        (full / "registration_manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (full / "registration_manifest.json").write_text(json.dumps({
            "case_id": case, "experiment": "full", "completion_status": "complete"
        }))
        fixed = _write_image(full / "preprocessed/fixed.nii.gz", base)
        moving_array = base + before
        moving_array[:, 0, 0] += 20
        moving = _write_image(full / "preprocessed/moving.nii.gz", moving_array)
        registered = _write_image(full / "deformable/moving_registered.nii.gz", base + after)
        mask_array = np.zeros(base.shape, dtype=np.uint8)
        mask_array[:, 1:3, 2:5] = 1
        mask_root = tmp_path / f"mask_{number}"
        mask = _write_image(mask_root / "lungs/combined_lungs.nii.gz", mask_array)
        masks[case] = mask
        (mask_root / "manifest.json").write_text(json.dumps({
            "case_id": f"{case}_fixed",
            "input_path": "[external]",
            "combined_mask_filename": "combined_lungs.nii.gz",
            "completion_status": "success",
        }))
        # Tests use external absolute paths while production manifests use project paths.
        manifest = json.loads((mask_root / "manifest.json").read_text())
        manifest["input_path"] = str(fixed)
        (mask_root / "manifest.json").write_text(json.dumps(manifest))

    inputs = {
        case: {
            "fixed": case_dirs[case] / "full/preprocessed/fixed.nii.gz",
            "moving": case_dirs[case] / "full/preprocessed/moving.nii.gz",
            "registered": case_dirs[case] / "full/deformable/moving_registered.nii.gz",
        }
        for case in ("case_1", "case_2")
    }
    limits = {"vmin": 0.0, "vmax": 10.0}
    report = {
        "analysis_type": "Case 1-versus-Case 2 equal-window registration difference visualization",
        "completion_status": "complete",
        "inputs": {
            case: {"analytical_images": {
                role: {"path": "[external path]", "sha256": _sha(path)}
                for role, path in inputs[case].items()
            }} for case in ("case_1", "case_2")
        },
        "shared_difference_window": {
            "percentile": 99.0,
            "pooled_volume_order": [
                "case_1_before", "case_1_after", "case_2_before", "case_2_after"
            ],
            **limits,
            "identical_for_every_difference_panel": True,
            "panel_limits": {
                figure: {
                    key: limits.copy() for key in (
                        "case_1_before", "case_1_after", "case_2_before", "case_2_after"
                    )
                } for figure in ("grayscale", "turbo")
            },
        },
    }
    report_path = tmp_path / "shared.json"
    report_path.write_text(json.dumps(report))
    return case_dirs, masks, report_path


def _kwargs(tmp_path):
    cases, masks, report = _fixtures(tmp_path)
    return {
        "case_1_dir": cases["case_1"],
        "case_2_dir": cases["case_2"],
        "case_1_lung_mask": masks["case_1"],
        "case_2_lung_mask": masks["case_2"],
        "shared_window_report": report,
    }


def test_render_keeps_whole_difference_gray_outside_and_turbo_only_inside():
    difference = np.array([[0, 5], [10, 2]], dtype=np.float32)
    mask = np.array([[0, 1], [0, 0]], dtype=bool)
    output = render_lung_masked_turbo_rgb(
        difference, mask, vmin=0, vmax=10, opacity=0.45
    )
    normalized = difference / 10
    assert np.allclose(output[~mask], np.repeat(normalized[..., None], 3, axis=2)[~mask])
    assert not np.allclose(output[mask][0], [0.5, 0.5, 0.5])
    turbo = np.asarray(matplotlib.colormaps["turbo"](0.5)[:3])
    assert np.allclose(output[mask][0], 0.55 * 0.5 + 0.45 * turbo)
    assert output[1, 1, 0] == pytest.approx(0.2)  # nonzero outside-mask change retained


def test_figures_record_exact_report_limits_and_preserve_sources(tmp_path):
    kwargs = _kwargs(tmp_path)
    sources = list(tmp_path.rglob("*.nii.gz")) + list(tmp_path.rglob("*.json"))
    before = {path: _sha(path) for path in sources}
    output = tmp_path / "figures"
    result = create_lung_masked_difference_figures(**kwargs, output_dir=output)
    assert result["shared_difference_window"]["vmin"] == 0.0
    assert result["shared_difference_window"]["vmax"] == 10.0
    for limits in result["shared_difference_window"]["panel_limits"]["axial"].values():
        assert limits == {"vmin": 0.0, "vmax": 10.0}
    for view in ("axial", "coronal", "sagittal"):
        assert all(
            limits == {"vmin": 0.0, "vmax": 10.0}
            for limits in result["shared_difference_window"]["panel_limits"]["multiplanar"][view].values()
        )
    assert result["display"]["fixed_anatomy_used_as_background"] is False
    assert result["display"]["lung_mask_restricts_analysis"] is False
    assert result["mask_and_clipping_statistics"]["by_volume"]["case_1_before"]["outside_mask_above_vmax"] > 0
    assert all(_sha(path) == digest for path, digest in before.items())
    assert result["source_immutability"]["unchanged"] is True
    assert all((output / name).is_file() for name in OUTPUT_NAMES.values())


def test_case_order_does_not_change_loaded_normalization(tmp_path):
    kwargs = _kwargs(tmp_path)
    forward = create_lung_masked_difference_figures(**kwargs, output_dir=tmp_path / "a", dry_run=True)
    swapped = kwargs.copy()
    swapped["case_1_dir"], swapped["case_2_dir"] = kwargs["case_2_dir"], kwargs["case_1_dir"]
    swapped["case_1_lung_mask"], swapped["case_2_lung_mask"] = kwargs["case_2_lung_mask"], kwargs["case_1_lung_mask"]
    # Swapping report identities must be explicit too; normalization remains report-owned.
    report = json.loads(Path(kwargs["shared_window_report"]).read_text())
    report["inputs"]["case_1"], report["inputs"]["case_2"] = report["inputs"]["case_2"], report["inputs"]["case_1"]
    swapped_report = tmp_path / "swapped.json"
    swapped_report.write_text(json.dumps(report))
    swapped["shared_window_report"] = swapped_report
    reverse = create_lung_masked_difference_figures(**swapped, output_dir=tmp_path / "b", dry_run=True)
    assert forward["shared_difference_window"]["vmax"] == reverse["shared_difference_window"]["vmax"] == 10.0


def test_geometry_binary_mask_and_bad_report_fail_safely(tmp_path):
    kwargs = _kwargs(tmp_path)
    mask_path = kwargs["case_2_lung_mask"]
    image = sitk.ReadImage(str(mask_path))
    image.SetSpacing((1.2, 1.5, 2.1))
    sitk.WriteImage(image, str(mask_path))
    with pytest.raises(ValueError, match="fixed/lung_mask exact geometry mismatch"):
        create_lung_masked_difference_figures(**kwargs, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()

    kwargs = _kwargs(tmp_path / "second")
    report = json.loads(Path(kwargs["shared_window_report"]).read_text())
    report["completion_status"] = "pending"
    Path(kwargs["shared_window_report"]).write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not complete"):
        create_lung_masked_difference_figures(**kwargs, output_dir=tmp_path / "bad-report")

    kwargs = _kwargs(tmp_path / "third")
    bad_mask = sitk.ReadImage(str(kwargs["case_1_lung_mask"]))
    values = sitk.GetArrayFromImage(bad_mask)
    values[0, 0, 0] = 2
    replacement = sitk.GetImageFromArray(values)
    replacement.CopyInformation(bad_mask)
    sitk.WriteImage(replacement, str(kwargs["case_1_lung_mask"]))
    with pytest.raises(ValueError, match="must be binary"):
        export_lung_masked_rgb_secondary_capture(**kwargs, output_dir=tmp_path / "bad-mask")
    assert not (tmp_path / "bad-mask").exists()


def test_dicom_is_valid_rgb_spatial_anonymous_and_same_scaling(tmp_path):
    kwargs = _kwargs(tmp_path)
    sources = list(tmp_path.rglob("*.nii.gz")) + list(tmp_path.rglob("*.json"))
    before = {path: _sha(path) for path in sources}
    output = tmp_path / "dicom"
    result = export_lung_masked_rgb_secondary_capture(**kwargs, output_dir=output)
    series = {}
    all_sops = set()
    decoded = {}
    for case in ("case_1", "case_2"):
        datasets = [pydicom.dcmread(str(path)) for path in sorted((output / case).glob("*.dcm"))]
        assert len(datasets) == 3 == result["slice_count"][case]
        assert [int(ds.InstanceNumber) for ds in datasets] == [1, 2, 3]
        assert all(str(ds.SOPClassUID) == str(SecondaryCaptureImageStorage) for ds in datasets)
        assert all(str(ds.file_meta.TransferSyntaxUID) == str(ExplicitVRLittleEndian) for ds in datasets)
        assert all(ds.pixel_array.shape == (4, 6, 3) for ds in datasets)
        assert all(len(ds.PixelData) == 4 * 6 * 3 for ds in datasets)
        assert all(np.allclose(ds.PixelSpacing, [1.5, 1.2]) for ds in datasets)
        assert [list(np.asarray(ds.ImagePositionPatient, float)) for ds in datasets] == [
            [10, 20, 30], [10, 20, 32], [10, 20, 34]
        ]
        case_sops = {str(ds.SOPInstanceUID) for ds in datasets}
        assert len(case_sops) == 3 and not all_sops.intersection(case_sops)
        all_sops.update(case_sops)
        case_series = {str(ds.SeriesInstanceUID) for ds in datasets}
        assert len(case_series) == 1
        series[case] = next(iter(case_series))
        assert all(str(ds.PatientName) == "ANONYMIZED^RESEARCH" for ds in datasets)
        assert all(ds.PatientIdentityRemoved == "YES" and ds.BurnedInAnnotation == "NO" for ds in datasets)
        assert all(not hasattr(ds, "SourceImageSequence") for ds in datasets)
        decoded[case] = np.stack([ds.pixel_array for ds in datasets])
    assert series["case_1"] != series["case_2"]
    assert result["validation"]["identical_shared_scaling"] == {
        "passed": True, "vmin": 0.0, "vmax": 10.0
    }
    assert result["display"]["fixed_anatomy_used_as_background"] is False
    assert result["source_immutability"]["unchanged"] is True
    assert all(_sha(path) == digest for path, digest in before.items())
    assert (output / MANIFEST_NAME).is_file()


def test_dry_run_clis_and_overwrite_protection(tmp_path, capsys):
    kwargs = _kwargs(tmp_path)
    common = [
        "--case-1-dir", str(kwargs["case_1_dir"]),
        "--case-2-dir", str(kwargs["case_2_dir"]),
        "--case-1-lung-mask", str(kwargs["case_1_lung_mask"]),
        "--case-2-lung-mask", str(kwargs["case_2_lung_mask"]),
        "--shared-window-report", str(kwargs["shared_window_report"]),
    ]
    assert figure_cli_main([*common, "--output-dir", str(tmp_path / "dry-fig"), "--dry-run"]) == 0
    assert dicom_cli_main([*common, "--output-dir", str(tmp_path / "dry-dicom"), "--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out
    assert not (tmp_path / "dry-fig").exists() and not (tmp_path / "dry-dicom").exists()
    output = tmp_path / "fig"
    create_lung_masked_difference_figures(**kwargs, output_dir=output)
    with pytest.raises(FileExistsError, match="--overwrite"):
        create_lung_masked_difference_figures(**kwargs, output_dir=output)
