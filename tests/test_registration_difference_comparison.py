from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from compare_registration_differences import main as cli_main  # noqa: E402
from registration_difference_comparison import (  # noqa: E402
    OUTPUT_NAMES,
    PANEL_KEYS,
    compare_registration_differences,
    compute_shared_difference_percentile,
)


def _write_image(path: Path, array: np.ndarray, *, spacing=(1.0, 1.0, 1.0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin((2.0, 3.0, 4.0))
    sitk.WriteImage(image, str(path))
    return path


def _case(root: Path, case: str, before: float, after: float) -> Path:
    case_dir = root / case
    manifest = case_dir / "full/registration_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {"case_id": case, "experiment": "full", "completion_status": "complete"}
        )
    )
    fixed = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    _write_image(case_dir / "full/preprocessed/fixed.nii.gz", fixed)
    _write_image(case_dir / "full/preprocessed/moving.nii.gz", fixed + before)
    registered = fixed + after
    registered[0, 0, 0] += before * 10
    _write_image(case_dir / "full/deformable/moving_registered.nii.gz", registered)
    return case_dir


def _fixtures(tmp_path: Path):
    return _case(tmp_path, "case_1", 10, 2), _case(tmp_path, "case_2", 20, 3)


def _hashes(case_dirs):
    paths = []
    for case_dir in case_dirs:
        paths.extend(
            (
                case_dir / "full/preprocessed/fixed.nii.gz",
                case_dir / "full/preprocessed/moving.nii.gz",
                case_dir / "full/deformable/moving_registered.nii.gz",
            )
        )
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_shared_limit_uses_every_voxel_from_all_four_volumes():
    fixed = np.zeros((2, 2, 2), dtype=np.float32)
    candidates = [np.ones_like(fixed), np.ones_like(fixed) * 2, np.ones_like(fixed) * 3]
    fourth = np.ones_like(fixed) * 4
    fourth[0, 0, 0] = 100
    candidates.append(fourth)
    actual = compute_shared_difference_percentile(
        [(fixed, candidate) for candidate in candidates], percentile=99
    )
    expected = np.percentile(
        np.concatenate([np.abs(fixed - candidate).reshape(-1) for candidate in candidates]),
        99,
        method="linear",
    )
    assert actual == pytest.approx(expected)
    assert actual > 4  # The high value exists only in the fourth volume.


def test_shared_limit_is_invariant_to_case_and_volume_order():
    fixed = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    pairs = [(fixed, fixed + value) for value in (1, 7, 3, 20)]
    forward = compute_shared_difference_percentile(pairs)
    reversed_order = compute_shared_difference_percentile(list(reversed(pairs)))
    swapped_cases = compute_shared_difference_percentile(pairs[2:] + pairs[:2])
    assert forward == reversed_order == swapped_cases


def test_every_figure_panel_receives_and_records_the_exact_same_limits(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    case_1, case_2 = _fixtures(tmp_path)
    original_imshow = Axes.imshow
    received_limits = []

    def recording_imshow(axis, *args, **kwargs):
        norm = kwargs.get("norm")
        if norm is not None:
            received_limits.append((norm.vmin, norm.vmax))
        return original_imshow(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", recording_imshow)
    result = compare_registration_differences(
        case_1_dir=case_1,
        case_2_dir=case_2,
        output_dir=tmp_path / "out",
    )
    window = result["shared_difference_window"]
    expected = {"vmin": window["vmin"], "vmax": window["vmax"]}
    limits = window["panel_limits"]
    for figure in ("grayscale", "turbo", "overlay"):
        assert set(limits[figure]) == set(PANEL_KEYS)
        assert all(value == expected for value in limits[figure].values())
    for view in ("axial", "coronal", "sagittal"):
        assert all(value == expected for value in limits["multiplanar"][view].values())
    assert len(received_limits) == 24
    assert set(received_limits) == {(window["vmin"], window["vmax"])}


def test_geometry_mismatch_fails_without_creating_outputs(tmp_path):
    case_1, case_2 = _fixtures(tmp_path)
    registered_path = case_2 / "full/deformable/moving_registered.nii.gz"
    image = sitk.ReadImage(str(registered_path))
    image.SetSpacing((1.0, 1.0, 1.1))
    sitk.WriteImage(image, str(registered_path))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="fixed/registered exact geometry mismatch"):
        compare_registration_differences(
            case_1_dir=case_1, case_2_dir=case_2, output_dir=output
        )
    assert not output.exists()


def test_inputs_are_unchanged_and_report_has_exact_shared_window(tmp_path):
    case_dirs = _fixtures(tmp_path)
    before = _hashes(case_dirs)
    output = tmp_path / "out"
    result = compare_registration_differences(
        case_1_dir=case_dirs[0], case_2_dir=case_dirs[1], output_dir=output
    )
    after = _hashes(case_dirs)
    saved = json.loads((output / OUTPUT_NAMES["report"]).read_text())
    assert before == after
    assert result["input_immutability"]["unchanged"] is True
    assert saved["shared_difference_window"] == result["shared_difference_window"]
    assert saved["shared_difference_window"]["vmin"] == 0.0
    assert saved["shared_difference_window"]["vmax"] == result["shared_difference_window"]["vmax"]
    assert saved["completion_status"] == "complete"
    for name, filename in OUTPUT_NAMES.items():
        assert (output / filename).is_file(), name


def test_dry_run_and_cli_do_not_write(tmp_path, capsys):
    case_1, case_2 = _fixtures(tmp_path)
    output = tmp_path / "out"
    assert cli_main(
        [
            "--case-1-dir", str(case_1),
            "--case-2-dir", str(case_2),
            "--output-dir", str(output),
            "--dry-run",
        ]
    ) == 0
    assert "dry-run" in capsys.readouterr().out
    assert not output.exists()


def test_overwrite_is_explicit(tmp_path):
    case_1, case_2 = _fixtures(tmp_path)
    output = tmp_path / "out"
    compare_registration_differences(case_1_dir=case_1, case_2_dir=case_2, output_dir=output)
    with pytest.raises(FileExistsError, match="--overwrite"):
        compare_registration_differences(case_1_dir=case_1, case_2_dir=case_2, output_dir=output)
    result = compare_registration_differences(
        case_1_dir=case_1, case_2_dir=case_2, output_dir=output, overwrite=True
    )
    assert result["completion_status"] == "complete"
