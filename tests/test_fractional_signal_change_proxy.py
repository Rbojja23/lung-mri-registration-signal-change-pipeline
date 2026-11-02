from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fractional_signal_change_proxy import (  # noqa: E402
    ANALYSIS_TYPE,
    compute_case_proxies,
    compute_shared_symmetric_limit,
    generate_fractional_signal_change_comparison,
    gaussian_kernel_radius_xyz,
    masked_normalized_box_mean,
    masked_normalized_gaussian_mean,
    sigma_mm_to_voxels,
)
from generate_fractional_signal_change_proxy import build_parser  # noqa: E402


DIRECTION = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _image(array, *, origin=(2.0, 3.0, 4.0)):
    image = sitk.GetImageFromArray(np.asarray(array))
    image.SetSpacing((1.2, 1.5, 2.0))
    image.SetOrigin(origin)
    image.SetDirection(DIRECTION)
    return image


def _write(path, array, *, origin=(2.0, 3.0, 4.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(_image(array, origin=origin), str(path))
    return path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _case(root, number, fixed, registered, mask, *, registered_origin=(2.0, 3.0, 4.0)):
    case_id = f"registration_case_{number}"
    directory = root / case_id
    fixed_path = _write(directory / "full/preprocessed/fixed.nii.gz", fixed)
    moving_path = _write(directory / "full/preprocessed/moving.nii.gz", registered + number)
    registered_path = _write(
        directory / "full/deformable/moving_registered.nii.gz",
        registered,
        origin=registered_origin,
    )
    role = {
        "fixed": f"case_{number}/bimg1.nii",
        "moving": f"case_{number}/bimg{number + 1}.nii",
        "source": "inferred-from-repeated-bimg1",
        "confirmed": False,
    }
    (directory / "full/registration_manifest.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "experiment": "full",
                "completion_status": "complete",
                "role_assignment": role,
            }
        )
    )
    (directory / "full/preprocessed/preprocessing_manifest.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "experiment": "full",
                "completion_status": "complete",
                "role_assignment": role,
            }
        )
    )
    mask_root = root / "totalsegmentator" / f"{case_id}_fixed"
    mask_path = _write(mask_root / "lungs/combined_lungs.nii.gz", mask.astype(np.uint8))
    (mask_root / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": f"{case_id}_fixed",
                "modality": "MR",
                "task": "total_mr",
                "totalsegmentator_version": "2.15.0",
                "input_path": str(fixed_path),
                "selected_lung_labels": ["lung_left", "lung_right"],
                "completion_status": "success",
            }
        )
    )
    return directory, mask_path, [
        fixed_path,
        moving_path,
        registered_path,
        mask_path,
        directory / "full/registration_manifest.json",
        directory / "full/preprocessed/preprocessing_manifest.json",
        mask_root / "manifest.json",
    ]


def _fixtures(tmp_path, *, bad_geometry=False):
    shape = (9, 10, 11)
    fixed = np.full(shape, 8.0, np.float32)
    mask = np.zeros(shape, bool)
    mask[1:8, 1:9, 1:10] = True
    registered_1 = np.full(shape, 10.0, np.float32)
    registered_2 = np.full(shape, 12.0, np.float32)
    case_1 = _case(tmp_path, 1, fixed, registered_1, mask)
    case_2 = _case(
        tmp_path,
        2,
        fixed,
        registered_2,
        mask,
        registered_origin=(9.0, 3.0, 4.0) if bad_geometry else (2.0, 3.0, 4.0),
    )
    return case_1, case_2


def _prior_proxy(tmp_path, case_1, case_2):
    directory = tmp_path / "prior_proxy"
    outputs = {}
    for case, fixture in (("case_1", case_1), ("case_2", case_2)):
        fixed = sitk.GetArrayFromImage(sitk.ReadImage(str(fixture[2][0]))).astype(np.float32)
        validity = np.ones_like(fixed, np.uint8)
        case_dir = directory / case
        outputs[case] = {}
        for key, filename, array in (
            ("native", "native_fractional_proxy.nii.gz", fixed),
            ("native_validity", "native_validity.nii.gz", validity),
            ("mean_filtered", "mean_filtered_fractional_proxy.nii.gz", fixed),
            ("mean_filtered_validity", "mean_filtered_validity.nii.gz", validity),
        ):
            path = _write(case_dir / filename, array)
            outputs[case][key] = str(path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fractional_proxy_report.json").write_text(
        json.dumps(
            {
                "completion_status": "complete",
                "outputs": {"cases": outputs},
                "display": {
                    "shared_symmetric_window": {
                        "percentile": 99.0,
                        "pooled_valid_voxel_count": 4,
                        "vmin": -1.0,
                        "vcenter": 0.0,
                        "vmax": 1.0,
                    }
                },
            }
        )
    )
    return directory


def _run(tmp_path, **kwargs):
    case_1, case_2 = _fixtures(tmp_path, bad_geometry=kwargs.pop("bad_geometry", False))
    prior_proxy_dir = _prior_proxy(tmp_path, case_1, case_2)
    result = generate_fractional_signal_change_comparison(
        case_1_dir=case_1[0],
        case_2_dir=case_2[0],
        case_1_lung_mask=case_1[1],
        case_2_lung_mask=case_2[1],
        output_dir=tmp_path / "proxy",
        denominator_threshold=kwargs.pop("denominator_threshold", 1.0),
        denominator_threshold_fraction=kwargs.pop("denominator_threshold_fraction", None),
        prior_proxy_dir=prior_proxy_dir,
        **kwargs,
    )
    return result, case_1, case_2


def test_native_formula_mask_confinement_and_invalid_denominator():
    fixed = np.full((3, 3, 3), 4.0, np.float32)
    registered = np.full_like(fixed, 8.0)
    registered[1, 1, 1] = 0.25
    mask = np.zeros_like(fixed, bool)
    mask[1:, 1:, 1:] = True
    result = compute_case_proxies(fixed, registered, mask, threshold=1.0)
    valid = result["native_valid"]
    assert result["native"][1, 1, 2] == pytest.approx(0.5)
    assert not valid[1, 1, 1] and np.isnan(result["native"][1, 1, 1])
    assert np.isnan(result["native"][~mask]).all()


def test_normalized_masked_filter_excludes_invalid_and_boundary_values():
    values = np.full((5, 5, 5), 10.0, np.float32)
    values[2, 2, 2] = 1000.0
    valid = np.zeros_like(values, bool)
    valid[1:4, 1:4, 1:4] = True
    valid[2, 2, 2] = False
    filtered, support = masked_normalized_box_mean(
        values, valid, kernel_radius_xyz=(1, 1, 1)
    )
    assert filtered[2, 2, 2] == pytest.approx(10.0)
    assert support[2, 2, 2] == 26
    assert filtered[0, 0, 0] == pytest.approx(10.0)
    assert support[0, 0, 0] == 1


def test_seven_voxel_kernel_behavior_and_filtered_ratio():
    shape = (9, 9, 9)
    valid = np.ones(shape, bool)
    impulse = np.zeros(shape, np.float32)
    impulse[4, 4, 4] = 343.0
    filtered, support = masked_normalized_box_mean(impulse, valid)
    assert filtered[4, 4, 4] == pytest.approx(1.0)
    assert support[4, 4, 4] == 343
    assert filtered[0, 0, 0] == pytest.approx(0.0)
    fixed = np.full(shape, 8.0, np.float32)
    registered = np.full(shape, 10.0, np.float32)
    result = compute_case_proxies(fixed, registered, valid, threshold=1.0)
    assert result["mean_filtered"][4, 4, 4] == pytest.approx(0.2)
    assert result["gaussian_filtered"][4, 4, 4] == pytest.approx(0.2)


def test_gaussian_weights_are_centered_mask_safe_and_finite_seven_voxels():
    shape = (9, 9, 9)
    valid = np.ones(shape, bool)
    impulse = np.zeros(shape, np.float32)
    impulse[4, 4, 4] = 1.0
    filtered, weight = masked_normalized_gaussian_mean(
        impulse, valid, sigma_voxels_xyz=(1.0, 1.0, 1.0)
    )
    assert filtered[4, 4, 4] > filtered[4, 4, 3] > filtered[4, 4, 2]
    assert filtered[4, 4, 1] > 0
    assert filtered[4, 4, 0] == pytest.approx(0.0)
    assert weight[4, 4, 4] == pytest.approx(1.0)

    constant = np.full(shape, 7.0, np.float32)
    lung = np.zeros(shape, bool)
    lung[2:7, 2:7, 2:7] = True
    constant_filtered, support_weight = masked_normalized_gaussian_mean(
        constant, lung, sigma_voxels_xyz=(1.0, 1.0, 1.0)
    )
    assert np.allclose(constant_filtered[support_weight > 0], 7.0)


def test_shared_symmetric_window_is_order_invariant_and_excludes_invalid():
    sets = [np.array([1.0, -2.0]), np.array([3.0]), np.array([-4.0]), np.array([5.0])]
    forward = compute_shared_symmetric_limit(sets, percentile=80)
    reverse = compute_shared_symmetric_limit(list(reversed(sets)), percentile=80)
    assert forward == reverse == pytest.approx(np.percentile([1, 2, 3, 4, 5], 80))
    with pytest.raises(ValueError, match="finite"):
        compute_shared_symmetric_limit([np.array([1.0, np.nan])])


def test_geometry_mismatch_rejected_without_outputs(tmp_path):
    with pytest.raises(ValueError, match="fixed/registered exact geometry mismatch"):
        _run(tmp_path, bad_geometry=True)
    assert not (tmp_path / "proxy").exists()


def test_report_outputs_geometry_validity_and_source_immutability(tmp_path):
    case_1, case_2 = _fixtures(tmp_path)
    prior_proxy_dir = _prior_proxy(tmp_path, case_1, case_2)
    prior_sources = list(prior_proxy_dir.rglob("*.nii.gz"))
    sources = case_1[2] + case_2[2]
    before = {path: _sha(path) for path in sources}
    prior_before = {path: _sha(path) for path in prior_sources}
    result = generate_fractional_signal_change_comparison(
        case_1_dir=case_1[0],
        case_2_dir=case_2[0],
        case_1_lung_mask=case_1[1],
        case_2_lung_mask=case_2[1],
        output_dir=tmp_path / "proxy",
        denominator_threshold=1.0,
        denominator_threshold_fraction=None,
        prior_proxy_dir=prior_proxy_dir,
    )
    assert {path: _sha(path) for path in sources} == before
    assert {path: _sha(path) for path in prior_sources} == prior_before
    assert result["analysis_type"] == ANALYSIS_TYPE
    assert result["formula"]["expression"] == "(registered_moving - fixed) / registered_moving"
    assert result["formula"]["machine_epsilon_used"] is False
    assert result["filtering"]["post_ratio_filtering_performed"] is False
    assert result["display"]["colormap"] == "RdBu_r"
    assert result["display"]["shared_symmetric_window"]["vmin"] == -result["display"]["shared_symmetric_window"]["vmax"]
    assert result["input_immutability"]["unchanged"] is True
    assert result["completion_status"] == "complete"
    for case in ("case_1", "case_2"):
        for filename in (
            "native_fractional_proxy.nii.gz",
            "native_validity.nii.gz",
            "mean_filtered_fractional_proxy.nii.gz",
            "mean_filtered_validity.nii.gz",
            "gaussian_filtered_fractional_proxy.nii.gz",
            "gaussian_filtered_validity.nii.gz",
        ):
            assert (tmp_path / "proxy" / case / filename).is_file()
        analytical = np.asanyarray(
            nib.load(str(tmp_path / "proxy" / case / "native_fractional_proxy.nii.gz")).dataobj
        )
        analytical_zyx = np.transpose(analytical, (2, 1, 0))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(case_1[1] if case == "case_1" else case_2[1]))).astype(bool)
        assert np.isnan(analytical_zyx[~mask]).all()
    assert (tmp_path / "proxy/fractional_proxy_axial_comparison.png").is_file()
    assert (tmp_path / "proxy/fractional_proxy_multiplanar_comparison.png").is_file()
    saved = json.loads((tmp_path / "proxy/fractional_proxy_report.json").read_text())
    assert saved["completion_status"] == "complete"
    assert saved["input_immutability"]["unchanged"] is True
    assert saved["display"]["previous_four_map_window_for_reference"]["vmax"] == 1.0
    assert saved["outputs"] and saved["output_checksums_sha256"]
    assert saved["geometry_validation"]["cross_case_fixed_grid"]["exact_agreement"] is True
    assert saved["gaussian_filtering"]["kernel"]["sigma_mm"] == pytest.approx(2.5)
    assert saved["gaussian_filtering"]["kernel"]["sigma_voxels_xyz"] == pytest.approx(
        [2.5 / 1.2, 2.5 / 1.5, 2.5 / 2.0]
    )
    assert saved["display"]["shared_symmetric_window"]["pooled_order"] == [
        "case_1_native",
        "case_1_mean_filtered",
        "case_1_gaussian_filtered",
        "case_2_native",
        "case_2_mean_filtered",
        "case_2_gaussian_filtered",
    ]


def test_relative_threshold_dry_run_and_cli_defaults_write_nothing(tmp_path):
    result, _, _ = _run(
        tmp_path,
        dry_run=True,
        denominator_threshold=None,
        denominator_threshold_fraction=0.1,
    )
    assert result["denominator_policy"]["threshold_mode"].startswith("fraction_of")
    assert result["cases"]["case_1"]["threshold"]["value"] == pytest.approx(1.0)
    assert result["writes_performed"] is False
    assert not (tmp_path / "proxy").exists()
    args = build_parser().parse_args([])
    assert args.denominator_threshold_fraction == pytest.approx(0.01)
    assert args.gaussian_sigma_mm == pytest.approx(2.5)
    assert args.gaussian_truncate == pytest.approx(1.5)
    assert args.dry_run is False and args.overwrite is False


def test_overwrite_is_explicit(tmp_path):
    result, case_1, case_2 = _run(tmp_path)
    assert result["completion_status"] == "complete"
    kwargs = dict(
        case_1_dir=case_1[0], case_2_dir=case_2[0],
        case_1_lung_mask=case_1[1], case_2_lung_mask=case_2[1],
        output_dir=tmp_path / "proxy", denominator_threshold=1.0,
        denominator_threshold_fraction=None,
        prior_proxy_dir=_prior_proxy(tmp_path, case_1, case_2),
    )
    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_fractional_signal_change_comparison(**kwargs)
    overwritten = generate_fractional_signal_change_comparison(**kwargs, overwrite=True)
    assert overwritten["completion_status"] == "complete"


def test_physical_sigma_conversion_anisotropy_and_support_policy():
    sigma = sigma_mm_to_voxels(3.0, (1.0, 1.5, 3.0))
    assert sigma == pytest.approx((3.0, 2.0, 1.0))
    assert gaussian_kernel_radius_xyz(
        sigma,
        truncate=1.5,
        support_radius_mm=None,
        spacing_xyz_mm=(1.0, 1.5, 3.0),
    ) == (5, 3, 2)
    assert gaussian_kernel_radius_xyz(
        sigma,
        truncate=99.0,
        support_radius_mm=3.1,
        spacing_xyz_mm=(1.0, 1.5, 3.0),
    ) == (4, 3, 2)


def test_gaussian_formula_uses_filtered_inputs_and_retains_native_centers():
    fixed = np.full((7, 7, 7), 2.0, np.float32)
    registered = np.full_like(fixed, 4.0)
    registered[3, 3, 3] = 8.0
    mask = np.ones_like(fixed, bool)
    result = compute_case_proxies(
        fixed,
        registered,
        mask,
        threshold=1.0,
        spacing_xyz_mm=(1.0, 2.0, 3.0),
        gaussian_sigma_mm=2.0,
        gaussian_truncate=1.0,
    )
    gf, support = masked_normalized_gaussian_mean(
        fixed,
        result["native_valid"],
        sigma_voxels_xyz=(2.0, 1.0, 2.0 / 3.0),
        kernel_radius_xyz=(2, 1, 1),
    )
    gm, _ = masked_normalized_gaussian_mean(
        registered,
        result["native_valid"],
        sigma_voxels_xyz=(2.0, 1.0, 2.0 / 3.0),
        kernel_radius_xyz=(2, 1, 1),
    )
    expected = (gm[3, 3, 3] - gf[3, 3, 3]) / gm[3, 3, 3]
    assert support[3, 3, 3] > 0
    assert result["gaussian_filtered"][3, 3, 3] == pytest.approx(expected)
    assert np.isnan(result["gaussian_filtered"][~mask]).all()


def test_invalid_voxels_have_zero_gaussian_contribution_and_nan_output():
    values = np.full((5, 5, 5), 6.0, np.float32)
    values[2, 2, 2] = 1_000_000.0
    valid = np.ones_like(values, bool)
    valid[2, 2, 2] = False
    filtered, support = masked_normalized_gaussian_mean(
        values,
        valid,
        sigma_voxels_xyz=(1.0, 1.0, 1.0),
        kernel_radius_xyz=(1, 1, 1),
    )
    assert filtered[2, 2, 2] == pytest.approx(6.0)
    assert support[2, 2, 2] > 0
