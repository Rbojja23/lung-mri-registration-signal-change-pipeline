from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup_deedsbcv_macos.sh"
TRACKED_PATCH = (
    ROOT
    / "third_party"
    / "deedsBCV"
    / "patches"
    / "0001-fix-nifti-writer-and-array-deallocation.patch"
)


def git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def fake_setup(tmp_path: Path):
    project = tmp_path / "project with spaces"
    deeds_root = project / "third_party" / "deedsBCV"
    patch_dir = deeds_root / "patches"
    patch_dir.mkdir(parents=True)

    upstream = tmp_path / "fake upstream"
    upstream.mkdir()
    git(upstream, "init")
    git(upstream, "config", "user.email", "test@example.invalid")
    git(upstream, "config", "user.name", "Test")
    (upstream / "src").mkdir()
    (upstream / "src" / "writer.h").write_text("short dimensions[4];\ndelete values;\n")
    (upstream / "LICENSE").write_text("MIT\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-m", "fixture upstream")
    commit = git(upstream, "rev-parse", "HEAD")
    (upstream / "src" / "writer.h").write_text("short dimensions[5];\ndelete []values;\n")
    patch_text = git(upstream, "diff", "--binary", "--", "src") + "\n"
    (patch_dir / TRACKED_PATCH.name).write_text(patch_text)
    git(upstream, "restore", "src/writer.h")

    tools = tmp_path / "fake tools"
    tools.mkdir()
    write_executable(
        tools / "uname",
        "#!/bin/sh\n[ \"$1\" = -s ] && echo Darwin || echo arm64\n",
    )
    write_executable(
        tools / "clang++",
        """#!/bin/sh
output=
previous=
for argument in "$@"; do
  if [ "$previous" = -o ]; then output=$argument; fi
  previous=$argument
done
name=$(basename "$output")
case "$name" in
  linearBCV) [ "${FAIL_REQUIRED:-}" = linearBCV ] && exit 9 ;;
  deedsBCV) [ "${FAIL_REQUIRED:-}" = deedsBCV ] && exit 9 ;;
  applyBCV|applyBCVfloat)
    { [ "${FAIL_OPTIONAL:-}" = all ] || [ "${FAIL_OPTIONAL:-}" = "$name" ]; } && exit 8
    ;;
esac
printf '#!/bin/sh\necho Usage: fixed.nii moving.nii\n' > "$output"
chmod +x "$output"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tools}:{env['PATH']}",
            "DEEDSBCV_PROJECT_ROOT": str(project),
            "DEEDSBCV_UPSTREAM_URL": str(upstream),
            "DEEDSBCV_PINNED_COMMIT": commit,
        }
    )

    def run(*args: str, **updates: str) -> subprocess.CompletedProcess[str]:
        run_env = env | updates
        return subprocess.run(
            [str(SETUP), "--without-openmp", *args],
            env=run_env,
            capture_output=True,
            text=True,
        )

    return project, commit, run


def test_tracked_patch_exists_and_contains_no_binary_changes():
    text = TRACKED_PATCH.read_text()
    assert text.startswith("diff --git a/src/")
    assert "GIT binary patch" not in text
    assert "Binary files" not in text
    assert text.count("new short[5]") == 3


def test_setup_requires_patch(fake_setup):
    project, _, run = fake_setup
    (project / "third_party" / "deedsBCV" / "patches" / TRACKED_PATCH.name).unlink()
    result = run()
    assert result.returncode != 0
    assert "patch is missing" in result.stderr


def test_patch_is_applied_once_and_paths_with_spaces_work(fake_setup):
    project, _, run = fake_setup
    first = run()
    second = run()
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "Patch status: newly applied" in first.stdout
    assert "Patch status: already applied" in second.stdout
    source = project / "third_party" / "deedsBCV" / "source"
    assert (source / "src" / "writer.h").read_text() == "short dimensions[5];\ndelete []values;\n"


def test_unrelated_dirty_source_is_rejected(fake_setup):
    project, _, run = fake_setup
    assert run().returncode == 0
    source = project / "third_party" / "deedsBCV" / "source"
    (source / "unrelated.txt").write_text("do not discard\n")
    result = run()
    assert result.returncode != 0
    assert "modifications other than the tracked patch" in result.stderr
    assert (source / "unrelated.txt").exists()


def test_wrong_pinned_commit_is_rejected(fake_setup):
    project, _, run = fake_setup
    assert run().returncode == 0
    source = project / "third_party" / "deedsBCV" / "source"
    git(source, "config", "user.email", "test@example.invalid")
    git(source, "config", "user.name", "Test")
    git(source, "add", "src/writer.h")
    git(source, "commit", "-m", "locally committed patch")
    result = run()
    assert result.returncode != 0
    assert "not pinned commit" in result.stderr


@pytest.mark.parametrize(
    ("failure", "optional_built", "optional_unavailable"),
    [
        ("all", [], ["applyBCV", "applyBCVfloat"]),
        ("applyBCVfloat", ["applyBCV"], ["applyBCVfloat"]),
        ("", ["applyBCV", "applyBCVfloat"], []),
    ],
)
def test_optional_build_combinations_are_nounset_safe(
    fake_setup, failure, optional_built, optional_unavailable
):
    project, commit, run = fake_setup
    result = run(FAIL_OPTIONAL=failure)
    assert result.returncode == 0, result.stderr
    assert "unbound variable" not in result.stderr
    assert "Required binaries built: linearBCV,deedsBCV" in result.stdout
    assert f"Optional binaries built: {','.join(optional_built)}" in result.stdout
    assert f"Optional binaries unavailable: {','.join(optional_unavailable)}" in result.stdout
    marker = project / "third_party" / "deedsBCV" / "build-arm64" / ".deedsbcv-build-state.json"
    state = json.loads(marker.read_text())
    assert state["pinned_commit"] == commit
    assert len(state["patch_sha256"]) == 64
    assert state["build_mode"] == "serial"
    assert state["compiler"].endswith("clang++")
    assert state["timestamp"].endswith("Z")
    assert state["required_binaries"] == ["linearBCV", "deedsBCV"]
    assert state["optional_binaries"] == optional_built
    assert state["optional_unavailable"] == optional_unavailable
    if failure:
        assert "Warning: optional binary unavailable" in result.stderr


def test_required_failure_is_nonzero_and_runs_no_registration(fake_setup):
    project, _, run = fake_setup
    result = run(FAIL_REQUIRED="deedsBCV")
    assert result.returncode != 0
    assert "Required binary failed to build: deedsBCV" in result.stderr
    assert not (project / "third_party" / "deedsBCV" / "build-arm64" / ".deedsbcv-build-state.json").exists()
    combined = result.stdout + result.stderr
    assert "fixed.nii" not in combined
    assert "moving.nii" not in combined
