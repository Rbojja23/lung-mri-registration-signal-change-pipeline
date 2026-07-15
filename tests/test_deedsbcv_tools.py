from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import check_deedsbcv  # noqa: E402
from deedsbcv_tools import (  # noqa: E402
    DeedsBCVInstallation,
    check_deedsbcv_health,
    discover_deedsbcv,
    installation_summary,
    validate_project_local_build,
)


def executable(directory: Path, name: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path.resolve()


def install(directory: Path, names=("linearBCV", "deedsBCV", "applyBCV", "applyBCVfloat")):
    return {name: executable(directory, name) for name in names}


def no_config(tmp_path: Path) -> Path:
    return tmp_path / "missing-config.yaml"


def local_patched_installation(tmp_path: Path, *, timestamp_epoch: float = 0):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "-C", str(source), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    (source / "src").mkdir()
    target = source / "src" / "writer.h"
    target.write_text("delete values;\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-m", "base"], check=True, capture_output=True)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    target.write_text("delete []values;\n")
    patch = tmp_path / "0001-fix-nifti-writer-and-array-deallocation.patch"
    patch.write_bytes(
        subprocess.run(
            ["git", "-C", str(source), "diff", "--binary", "--no-ext-diff", "--", "src"],
            check=True,
            capture_output=True,
        ).stdout
    )
    directory = tmp_path / "build"
    install(directory, ("linearBCV", "deedsBCV"))
    marker = directory / ".deedsbcv-build-state.json"
    marker.write_text(
        json.dumps(
            {
                "pinned_commit": commit,
                "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                "timestamp_epoch": timestamp_epoch,
            }
        )
    )
    installation = discover_deedsbcv(
        directory,
        config_path=no_config(tmp_path),
        project_build_dir=directory,
        project_source_dir=source,
        patch_path=patch,
        expected_commit=commit,
    )
    return installation, source, patch, marker, commit


def test_explicit_directory_has_highest_precedence(tmp_path):
    explicit = tmp_path / "explicit tools"
    configured = tmp_path / "configured"
    install(explicit)
    install(configured)

    result = discover_deedsbcv(
        explicit,
        config_bin_dir=configured,
        environ={"DEEDSBCV_BIN_DIR": str(tmp_path / "env")},
        project_build_dir=tmp_path / "local",
    )

    assert result.status == "available"
    assert result.source == "explicit"
    assert result.binaries["linearBCV"] == explicit.resolve() / "linearBCV"


def test_config_then_environment_then_project_precedence(tmp_path):
    configured = tmp_path / "config tools"
    environment = tmp_path / "env tools"
    local = tmp_path / "local tools"
    install(configured)
    install(environment)
    install(local)

    result = discover_deedsbcv(
        config_bin_dir=configured,
        environ={"DEEDSBCV_BIN_DIR": str(environment)},
        project_build_dir=local,
    )
    assert result.source == "config"

    result = discover_deedsbcv(
        config_path=no_config(tmp_path),
        environ={"DEEDSBCV_BIN_DIR": str(environment)},
        project_build_dir=local,
    )
    assert result.source == "environment"

    result = discover_deedsbcv(
        config_path=no_config(tmp_path), environ={}, project_build_dir=local
    )
    assert result.source == "project-local"


def test_default_config_file_value_is_read(tmp_path):
    tools = tmp_path / "configured"
    install(tools)
    config = tmp_path / "config.yaml"
    config.write_text(f"registration:\n  deeds_bin_dir: {tools}\n")
    result = discover_deedsbcv(
        config_path=config, environ={}, project_build_dir=tmp_path / "missing"
    )
    assert result.source == "config"
    assert result.status == "available"


def test_path_fallback_can_resolve_individual_binaries(tmp_path):
    path_dir = tmp_path / "path bin"
    found = install(path_dir)

    def fake_which(name, *, path):
        assert path == "controlled-path"
        return str(found[name])

    result = discover_deedsbcv(
        config_path=no_config(tmp_path),
        environ={},
        project_build_dir=tmp_path / "missing",
        path="controlled-path",
        which=fake_which,
    )
    assert result.source == "PATH"
    assert result.status == "available"


@pytest.mark.parametrize(
    ("names", "status"),
    [
        (("linearBCV", "deedsBCV"), "available"),
        (("linearBCV",), "partial"),
        (("applyBCV",), "partial"),
        ((), "not_found"),
    ],
)
def test_installation_states(tmp_path, names, status):
    directory = tmp_path / "bin"
    install(directory, names)
    result = discover_deedsbcv(
        directory, config_path=no_config(tmp_path), environ={}
    )
    assert result.status == status


def test_missing_optional_apply_binaries_is_available(tmp_path):
    directory = tmp_path / "bin"
    install(directory, ("linearBCV", "deedsBCV"))
    result = discover_deedsbcv(directory, config_path=no_config(tmp_path))
    assert result.status == "available"
    assert result.binaries["applyBCV"] is None


def test_health_accepts_recognizable_help_at_zero_or_nonzero_exit(tmp_path):
    directory = tmp_path / "bin with spaces"
    install(directory, ("linearBCV", "deedsBCV"))
    installation = discover_deedsbcv(directory, config_path=no_config(tmp_path))
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        returncode = 0 if command[0].endswith("linearBCV") else 1
        return subprocess.CompletedProcess(command, returncode, stdout="Usage: fixed.nii moving.nii", stderr="")

    result = check_deedsbcv_health(installation, runner=runner)
    assert result.status == "available"
    assert result.health["linearBCV"].status == "healthy"
    assert result.health["deedsBCV"].status == "healthy"
    assert result.health["applyBCV"].status == "missing"
    assert all(call[0][1] == "-h" for call in calls)
    assert all(call[1]["shell"] is False for call in calls)
    assert all(call[1]["capture_output"] is True for call in calls)
    assert all(len(call[0]) == 2 for call in calls)


def test_health_unrecognized_output_and_timeout_are_unhealthy(tmp_path):
    directory = tmp_path / "bin"
    install(directory, ("linearBCV", "deedsBCV"))
    installation = discover_deedsbcv(directory, config_path=no_config(tmp_path))

    def runner(command, **kwargs):
        if command[0].endswith("linearBCV"):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout="unexpected", stderr="")

    result = check_deedsbcv_health(installation, runner=runner, timeout=0.25)
    assert result.status == "unhealthy"
    assert "timed out" in result.health["linearBCV"].message
    assert "not recognized" in result.health["deedsBCV"].message


def test_health_does_not_probe_partial_required_installation_with_images(tmp_path):
    directory = tmp_path / "bin"
    install(directory, ("linearBCV",))
    installation = discover_deedsbcv(directory, config_path=no_config(tmp_path))
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="linearBCV Usage", stderr="")

    result = check_deedsbcv_health(installation, runner=runner)
    assert result.status == "unhealthy"
    assert calls == [[str(directory.resolve() / "linearBCV"), "-h"]]
    assert not any(".nii" in argument for command in calls for argument in command)


def test_json_summary_hides_external_absolute_paths(tmp_path):
    directory = tmp_path / "private external tools"
    install(directory, ("linearBCV", "deedsBCV"))
    installation = discover_deedsbcv(directory, config_path=no_config(tmp_path))
    encoded = json.dumps(installation_summary(installation))
    assert str(tmp_path) not in encoded
    assert "[external path]" in encoded
    assert ".nii" not in encoded
    assert installation_summary(installation)["patch"]["status"] == "not_required"


def test_project_local_patch_sha_mismatch_is_unhealthy(tmp_path):
    installation, _, _, marker, _ = local_patched_installation(tmp_path)
    state = json.loads(marker.read_text())
    state["patch_sha256"] = "0" * 64
    marker.write_text(json.dumps(state))
    result = validate_project_local_build(
        installation,
        source_dir=tmp_path / "source",
        patch_path=tmp_path / "0001-fix-nifti-writer-and-array-deallocation.patch",
        expected_commit=installation.patch.pinned_commit_found,
    )
    assert result.status == "unhealthy"
    assert result.patch.status == "unhealthy"
    assert "SHA-256 differs" in result.patch.message


def test_project_local_stale_binary_is_unhealthy(tmp_path):
    future = 4_102_444_800
    installation, _, _, _, _ = local_patched_installation(
        tmp_path, timestamp_epoch=future
    )
    assert installation.status == "unhealthy"
    assert "required binary is stale" in installation.patch.message


def test_project_local_missing_marker_is_unhealthy(tmp_path):
    installation, source, patch, marker, commit = local_patched_installation(tmp_path)
    marker.unlink()
    result = validate_project_local_build(
        installation,
        source_dir=source,
        patch_path=patch,
        expected_commit=commit,
    )
    assert result.status == "unhealthy"
    assert "marker is missing" in result.patch.message


def test_json_health_report_includes_safe_patch_status(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "external tools"
    install(directory, ("linearBCV", "deedsBCV"))
    monkeypatch.setattr(
        check_deedsbcv,
        "check_deedsbcv_health",
        lambda installation: installation,
    )
    code = check_deedsbcv.main(["--bin-dir", str(directory), "--health-check", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["patch"]["status"] == "not_required"
    assert str(tmp_path) not in json.dumps(report)


def test_cli_json_dry_run_executes_no_health_subprocess(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "bin"
    install(directory, ("linearBCV", "deedsBCV"))
    monkeypatch.setattr(
        check_deedsbcv,
        "check_deedsbcv_health",
        lambda installation: pytest.fail("health probe must not run in dry-run"),
    )
    code = check_deedsbcv.main(
        ["--bin-dir", str(directory), "--health-check", "--dry-run", "--json"]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["dry_run"] is True
    assert report["health"] == {}
