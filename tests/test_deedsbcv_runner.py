from pathlib import Path
import json
import subprocess
import sys

import pytest
import SimpleITK as sitk


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import deedsbcv_runner  # noqa: E402
from deedsbcv_runner import (  # noqa: E402
    RegistrationRunError,
    build_deformable_command,
    build_linear_command,
    run_deedsbcv_registration,
    validate_prepared_inputs,
    verify_affine_matrix,
)
from registration_inputs import RegistrationInputPair, prepare_registration_inputs  # noqa: E402


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake binary")
    path.chmod(0o755)
    return path


def make_installation(tmp_path: Path) -> Path:
    directory = tmp_path / "binary tools with spaces"
    executable(directory / "linearBCV")
    executable(directory / "deedsBCV")
    return directory


def write_image(path: Path, *, size=(6, 5, 4), spacing=(1.0, 1.0, 1.0), value=1.0):
    image = sitk.Image(size, sitk.sitkFloat32)
    image += value
    image.SetSpacing(spacing)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def make_prepared(tmp_path: Path, *, case_id="case", spacing=(1.0, 1.0, 1.0)) -> Path:
    fixed = write_image(tmp_path / "source images/fixed.nii", spacing=spacing, value=2)
    moving = write_image(tmp_path / "source images/moving.nii", spacing=spacing, value=3)
    output_root = tmp_path / "registration outputs"
    prepare_registration_inputs(
        RegistrationInputPair(
            fixed,
            moving,
            case_id,
            "full",
            role_assignment_source="manual-test",
            roles_confirmed=True,
        ),
        output_root=output_root,
    )
    return output_root / case_id / "full/preprocessed"


class FakeProcess:
    def __init__(self, command, behavior, calls):
        self.command = command
        self.behavior = behavior
        self.calls = calls
        self.returncode = behavior.get("returncode", 0)
        self.terminated = False
        self.killed = False

    def communicate(self, timeout):
        self.calls.append((self.command, timeout))
        stage = "affine" if Path(self.command[0]).name == "linearBCV" else "deformable"
        if self.behavior.get("timeout") == stage:
            raise subprocess.TimeoutExpired(self.command, timeout, output="partial out", stderr="partial err")
        if self.behavior.get("interrupt") == stage:
            raise KeyboardInterrupt
        returncode = self.behavior.get(f"{stage}_returncode", self.returncode)
        self.returncode = returncode
        if returncode == 0:
            prefix = Path(self.command[self.command.index("-O") + 1])
            prefix.parent.mkdir(parents=True, exist_ok=True)
            if stage == "affine" and not self.behavior.get("missing_matrix"):
                matrix = prefix.parent / f"{prefix.name}_matrix.txt"
                matrix.write_text(
                    "bad matrix\n"
                    if self.behavior.get("malformed_matrix")
                    else "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
                )
            if stage == "deformable":
                if not self.behavior.get("missing_registered"):
                    fixed = Path(self.command[self.command.index("-F") + 1])
                    image = sitk.ReadImage(str(fixed))
                    if self.behavior.get("dimension_mismatch"):
                        image = sitk.Image([7, 5, 4], sitk.sitkFloat32)
                    if self.behavior.get("geometry_mismatch"):
                        image.SetOrigin((4.0, 0.0, 0.0))
                    sitk.WriteImage(image, str(prefix.parent / f"{prefix.name}_deformed.nii.gz"))
                if not self.behavior.get("missing_displacement"):
                    (prefix.parent / f"{prefix.name}_displacements.dat").write_bytes(b"field-data")
        return f"{stage} stdout", f"{stage} stderr"

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout):
        self.returncode = -15
        return self.returncode


def fake_factory(behavior=None):
    behavior = behavior or {}
    calls = []
    processes = []

    def factory(command, **kwargs):
        assert kwargs["shell"] is False
        assert kwargs["text"] is True
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        process = FakeProcess(command, behavior, calls)
        processes.append(process)
        return process

    factory.calls = calls
    factory.processes = processes
    return factory


def run(tmp_path, *, behavior=None, prepared=None, **kwargs):
    prepared = prepared or make_prepared(tmp_path)
    binaries = make_installation(tmp_path)
    factory = fake_factory(behavior)
    result = run_deedsbcv_registration(
        prepared_dir=prepared,
        case_id="case",
        bin_dir=binaries,
        timeout_seconds=10,
        process_factory=factory,
        **kwargs,
    )
    return result, factory, prepared


def test_exact_default_commands_and_paths_with_spaces(tmp_path):
    linear = build_linear_command("bin tools/linearBCV", "fixed image.nii.gz", "moving image.nii.gz", "affine out/affine")
    deformable = build_deformable_command(
        "bin tools/deedsBCV",
        "fixed image.nii.gz",
        "moving image.nii.gz",
        "deform out/deformable",
        "affine out/affine_matrix.txt",
    )
    assert linear == ["bin tools/linearBCV", "-F", "fixed image.nii.gz", "-M", "moving image.nii.gz", "-O", "affine out/affine"]
    assert deformable == ["bin tools/deedsBCV", "-F", "fixed image.nii.gz", "-M", "moving image.nii.gz", "-O", "deform out/deformable", "-A", "affine out/affine_matrix.txt"]
    assert not {"-G", "-L", "-Q", "-a", "-S"}.intersection(linear + deformable)


def test_prepared_manifest_validation_and_binary_discovery(tmp_path):
    prepared = make_prepared(tmp_path)
    validated = validate_prepared_inputs(prepared, case_id="case", experiment="full")
    assert validated["comparison"]["classification"] == "same-grid"
    result, factory, _ = run(tmp_path, prepared=prepared, dry_run=True)
    assert result["deedsbcv"]["installation_source"] == "explicit"
    assert factory.calls == []


def test_explicit_prepared_file_api(tmp_path):
    prepared = make_prepared(tmp_path)
    validated = validate_prepared_inputs(
        case_id="case",
        experiment="full",
        fixed_path=prepared / "fixed.nii.gz",
        moving_path=prepared / "moving.nii.gz",
        manifest_path=prepared / "preprocessing_manifest.json",
    )
    assert validated["directory"] == prepared.resolve()


@pytest.mark.parametrize("missing", ["fixed.nii.gz", "moving.nii.gz"])
def test_missing_prepared_input(tmp_path, missing):
    prepared = make_prepared(tmp_path)
    (prepared / missing).unlink()
    with pytest.raises(FileNotFoundError, match="working copy is missing"):
        validate_prepared_inputs(prepared, case_id="case", experiment="full")


def test_incomplete_or_mismatched_preprocessing_manifest(tmp_path):
    prepared = make_prepared(tmp_path)
    path = prepared / "preprocessing_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["completion_status"] = "pending"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not complete"):
        validate_prepared_inputs(prepared, case_id="case", experiment="full")


def test_anisotropic_and_dimension_mismatch_rejected(tmp_path):
    anisotropic = make_prepared(tmp_path / "anisotropic", spacing=(1.0, 1.0, 1.0))
    manifest_path = anisotropic / "preprocessing_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["geometry_comparison"]["classification"] = "anisotropic"
    manifest["geometry_comparison"]["resampling_required"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same-grid"):
        validate_prepared_inputs(anisotropic, case_id="case", experiment="full")

    prepared = make_prepared(tmp_path / "dimensions")
    write_image(prepared / "moving.nii.gz", size=(7, 5, 4))
    manifest = json.loads((prepared / "preprocessing_manifest.json").read_text())
    manifest["checksums_sha256"]["working_copies"]["moving"] = deedsbcv_runner.sha256_file(prepared / "moving.nii.gz")
    (prepared / "preprocessing_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different-dimensions"):
        validate_prepared_inputs(prepared, case_id="case", experiment="full")


def test_success_maps_exact_upstream_outputs_and_writes_logs(tmp_path):
    result, factory, prepared = run(tmp_path)
    experiment = prepared.parent
    assert result["completion_status"] == "complete"
    assert (experiment / "affine/affine_matrix.txt").is_file()
    assert (experiment / "deformable/moving_registered.nii.gz").is_file()
    assert (experiment / "deformable/displacements.dat").is_file()
    assert not (experiment / "deformable/deformable_deformed.nii.gz").exists()
    assert (experiment / "affine/linear_stdout.log").read_text() == "affine stdout"
    assert (experiment / "deformable/deeds_stderr.log").read_text() == "deformable stderr"
    assert len(factory.calls) == 2


@pytest.mark.parametrize("stage", ["affine", "deformable"])
def test_nonzero_exit_records_incomplete_manifest_and_logs(tmp_path, stage):
    prepared = make_prepared(tmp_path)
    binaries = make_installation(tmp_path)
    factory = fake_factory({f"{stage}_returncode": 7})
    with pytest.raises(RegistrationRunError, match="status 7"):
        run_deedsbcv_registration(
            prepared_dir=prepared,
            case_id="case",
            bin_dir=binaries,
            timeout_seconds=10,
            process_factory=factory,
        )
    manifest = json.loads((prepared.parent / "registration_manifest.json").read_text())
    assert manifest["completion_status"] == "incomplete"
    assert manifest["stages"][stage]["status"] == "failed"
    log = prepared.parent / ("affine/linear_stdout.log" if stage == "affine" else "deformable/deeds_stdout.log")
    assert log.is_file()


@pytest.mark.parametrize("stage", ["affine", "deformable"])
def test_timeout_terminates_process_and_records_failure(tmp_path, stage):
    prepared = make_prepared(tmp_path)
    binaries = make_installation(tmp_path)
    factory = fake_factory({"timeout": stage})
    with pytest.raises(RegistrationRunError, match="timed out"):
        run_deedsbcv_registration(
            prepared_dir=prepared,
            case_id="case",
            bin_dir=binaries,
            timeout_seconds=0.25,
            process_factory=factory,
        )
    assert factory.processes[-1].terminated is True


def test_keyboard_interrupt_terminates_process_and_marks_incomplete(tmp_path):
    prepared = make_prepared(tmp_path)
    factory = fake_factory({"interrupt": "affine"})
    with pytest.raises(KeyboardInterrupt):
        run_deedsbcv_registration(
            prepared_dir=prepared,
            case_id="case",
            bin_dir=make_installation(tmp_path),
            process_factory=factory,
        )
    assert factory.processes[0].terminated is True
    manifest = json.loads((prepared.parent / "registration_manifest.json").read_text())
    assert manifest["completion_status"] == "incomplete"
    assert manifest["stages"]["affine"]["status"] == "failed"


@pytest.mark.parametrize(
    ("behavior", "message"),
    [
        ({"missing_matrix": True}, "matrix is missing"),
        ({"malformed_matrix": True}, "16 finite"),
        ({"missing_registered": True}, "registered moving image is missing"),
        ({"missing_displacement": True}, "displacement output is missing"),
        ({"dimension_mismatch": True}, "dimensions do not match"),
    ],
)
def test_missing_or_malformed_stage_outputs(tmp_path, behavior, message):
    with pytest.raises(RegistrationRunError, match=message):
        run(tmp_path, behavior=behavior)


def test_affine_matrix_rejects_nan_and_wrong_value_count(tmp_path):
    matrix = tmp_path / "matrix.txt"
    matrix.write_text("1 0 0 nan\n" * 4)
    with pytest.raises(ValueError, match="NaN"):
        verify_affine_matrix(matrix)
    matrix.write_text("1 2 3")
    with pytest.raises(ValueError, match="16 finite"):
        verify_affine_matrix(matrix)


def test_registered_nonfinite_validation_failure_is_not_success(tmp_path, monkeypatch):
    original = deedsbcv_runner.validate_registration_image

    def validating(path, **kwargs):
        if kwargs.get("role") == "registered moving":
            raise ValueError("Registered moving image contains NaN values.")
        return original(path, **kwargs)

    monkeypatch.setattr(deedsbcv_runner, "validate_registration_image", validating)
    with pytest.raises(RegistrationRunError, match="contains NaN"):
        run(tmp_path)


def test_registered_geometry_mismatch_is_detected_without_correction(tmp_path):
    result, _, prepared = run(tmp_path, behavior={"geometry_mismatch": True})
    verification = result["stages"]["deformable"]["verification"]
    assert verification["dimensions_match_fixed"] is True
    assert verification["full_geometry_matches_fixed"] is False
    assert "no metadata correction" in result["warnings"][-1]
    image = sitk.ReadImage(str(prepared.parent / "deformable/moving_registered.nii.gz"))
    assert image.GetOrigin() == (4.0, 0.0, 0.0)


def test_dry_run_executes_nothing_and_writes_nothing(tmp_path):
    prepared = make_prepared(tmp_path)
    before = sorted(path.relative_to(prepared.parent) for path in prepared.parent.rglob("*"))
    result, factory, _ = run(tmp_path, prepared=prepared, dry_run=True)
    after = sorted(path.relative_to(prepared.parent) for path in prepared.parent.rglob("*"))
    assert result["completion_status"] == "dry-run"
    assert factory.calls == []
    assert before == after
    assert not (prepared.parent / "registration_manifest.json").exists()


def test_overwrite_protection_and_scoped_overwrite(tmp_path):
    result, _, prepared = run(tmp_path)
    source_hash = deedsbcv_runner.sha256_file(prepared / "fixed.nii.gz")
    unrelated = prepared.parent / "unrelated.txt"
    unrelated.write_text("keep")
    binaries = make_installation(tmp_path)
    with pytest.raises(FileExistsError):
        run_deedsbcv_registration(
            prepared_dir=prepared, case_id="case", bin_dir=binaries, process_factory=fake_factory()
        )
    run_deedsbcv_registration(
        prepared_dir=prepared,
        case_id="case",
        bin_dir=binaries,
        overwrite=True,
        process_factory=fake_factory(),
    )
    assert unrelated.read_text() == "keep"
    assert deedsbcv_runner.sha256_file(prepared / "fixed.nii.gz") == source_hash


def test_affine_and_full_resume(tmp_path):
    _, _, prepared = run(tmp_path)
    binaries = make_installation(tmp_path)
    full_factory = fake_factory()
    full = run_deedsbcv_registration(
        prepared_dir=prepared,
        case_id="case",
        bin_dir=binaries,
        resume=True,
        process_factory=full_factory,
    )
    assert full_factory.calls == []
    assert full["stages"]["affine"]["status"] == "reused"
    assert full["stages"]["deformable"]["status"] == "reused"

    (prepared.parent / "deformable/moving_registered.nii.gz").unlink()
    affine_factory = fake_factory()
    resumed = run_deedsbcv_registration(
        prepared_dir=prepared,
        case_id="case",
        bin_dir=binaries,
        resume=True,
        process_factory=affine_factory,
    )
    assert len(affine_factory.calls) == 1
    assert Path(affine_factory.calls[0][0][0]).name == "deedsBCV"
    assert resumed["stages"]["affine"]["status"] == "reused"


def test_stale_prepared_input_is_rejected_before_execution(tmp_path):
    prepared = make_prepared(tmp_path)
    image = sitk.ReadImage(str(prepared / "fixed.nii.gz"))
    image += 1
    sitk.WriteImage(image, str(prepared / "fixed.nii.gz"))
    factory = fake_factory()
    with pytest.raises(ValueError, match="stale"):
        run_deedsbcv_registration(
            prepared_dir=prepared,
            case_id="case",
            bin_dir=make_installation(tmp_path),
            resume=True,
            process_factory=factory,
        )
    assert factory.calls == []


def test_manifest_is_privacy_safe_and_contains_no_raw_logs(tmp_path):
    result, _, _ = run(tmp_path)
    encoded = json.dumps(result)
    assert str(tmp_path) not in encoded
    assert "affine stdout" not in encoded
    assert "deformable stderr" not in encoded
    assert result["intended_use"] == "Research-only and non-diagnostic."
    assert result["commands"]["default_parameters_by_omission"] is True
