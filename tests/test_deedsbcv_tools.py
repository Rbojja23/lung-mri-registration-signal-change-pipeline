from pathlib import Path
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
