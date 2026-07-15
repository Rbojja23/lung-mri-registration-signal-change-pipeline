from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Mapping, Sequence

import yaml

from project_paths import PROJECT_ROOT, display_path


REQUIRED_BINARIES = ("linearBCV", "deedsBCV")
OPTIONAL_BINARIES = ("applyBCV", "applyBCVfloat")
EXPECTED_BINARIES = REQUIRED_BINARIES + OPTIONAL_BINARIES
DEFAULT_BUILD_DIR = PROJECT_ROOT / "third_party" / "deedsBCV" / "build-arm64"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "third_party" / "deedsBCV" / "source"
DEFAULT_PATCH_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "deedsBCV"
    / "patches"
    / "0001-fix-nifti-writer-and-array-deallocation.patch"
)
EXPECTED_PINNED_COMMIT = "00b46539aa03d23988a241173d8f1f6c0c68dc6a"
BUILD_STATE_FILENAME = ".deedsbcv-build-state.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class BinaryHealth:
    name: str
    status: str
    message: str
    returncode: int | None = None


@dataclass(frozen=True)
class PatchHealth:
    project_local: bool
    status: str
    message: str
    pinned_commit_expected: str = EXPECTED_PINNED_COMMIT
    pinned_commit_found: str | None = None
    patch_expected: str = DEFAULT_PATCH_PATH.name
    patch_sha256: str | None = None
    source_patched: bool | None = None
    build_state_found: bool | None = None


@dataclass(frozen=True)
class DeedsBCVInstallation:
    status: str
    binaries: dict[str, Path | None]
    source: str | None = None
    bin_dir: Path | None = None
    message: str | None = None
    health: dict[str, BinaryHealth] = field(default_factory=dict)
    patch: PatchHealth | None = None

    @property
    def required_available(self) -> bool:
        return all(self.binaries.get(name) is not None for name in REQUIRED_BINARIES)


def resolve_binary_path(directory: str | Path, name: str) -> Path | None:
    """Resolve one executable in a known directory without searching elsewhere."""
    if name not in EXPECTED_BINARIES:
        raise ValueError(f"Unknown deedsBCV binary: {name}")
    candidate = Path(directory).expanduser().resolve(strict=False) / name
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def discover_deedsbcv(
    bin_dir: str | Path | None = None,
    *,
    config_bin_dir: str | Path | None = None,
    config_path: str | Path | None = DEFAULT_CONFIG_PATH,
    environ: Mapping[str, str] | None = None,
    project_build_dir: str | Path = DEFAULT_BUILD_DIR,
    path: str | None = None,
    which: Callable[..., str | None] = shutil.which,
    project_source_dir: str | Path = DEFAULT_SOURCE_DIR,
    patch_path: str | Path = DEFAULT_PATCH_PATH,
    expected_commit: str = EXPECTED_PINNED_COMMIT,
) -> DeedsBCVInstallation:
    """Discover deedsBCV using bounded, deterministic precedence."""
    env = os.environ if environ is None else environ
    configured = config_bin_dir
    if configured is None and config_path is not None:
        configured = _read_config_bin_dir(config_path)

    directory_candidates: list[tuple[str, str | Path | None, bool]] = [
        ("explicit", bin_dir, True),
        ("config", configured, True),
        ("environment", env.get("DEEDSBCV_BIN_DIR"), True),
        ("project-local", project_build_dir, False),
    ]
    for source, value, authoritative in directory_candidates:
        if value is None or str(value).strip() == "":
            continue
        directory = Path(value).expanduser().resolve(strict=False)
        binaries = {name: resolve_binary_path(directory, name) for name in EXPECTED_BINARIES}
        if authoritative or any(binaries.values()):
            installation = _installation_from_paths(binaries, source=source, bin_dir=directory)
            if directory == Path(project_build_dir).expanduser().resolve(strict=False):
                return validate_project_local_build(
                    installation,
                    source_dir=project_source_dir,
                    patch_path=patch_path,
                    expected_commit=expected_commit,
                )
            return installation

    binaries = {
        name: _which_path(name, path=path, which=which)
        for name in EXPECTED_BINARIES
    }
    return _installation_from_paths(binaries, source="PATH", bin_dir=None)


def check_deedsbcv_health(
    installation: DeedsBCVInstallation,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout: float = 5.0,
) -> DeedsBCVInstallation:
    """Probe help output only; no image paths or registration arguments are used."""
    if installation.status == "not_found":
        return installation

    results: dict[str, BinaryHealth] = {}
    for name in EXPECTED_BINARIES:
        executable = installation.binaries.get(name)
        if executable is None:
            requirement = "required" if name in REQUIRED_BINARIES else "optional"
            results[name] = BinaryHealth(name, "missing", f"{requirement} binary was not found")
            continue
        command = [str(executable), "-h"]
        try:
            completed = runner(
                command,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            results[name] = BinaryHealth(name, "unhealthy", f"help probe timed out after {timeout:g}s")
        except (OSError, subprocess.SubprocessError) as exc:
            results[name] = BinaryHealth(name, "unhealthy", f"help probe failed: {exc}")
        else:
            output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
            if _recognizable_help(name, output):
                results[name] = BinaryHealth(
                    name, "healthy", "recognizable help output", completed.returncode
                )
            else:
                results[name] = BinaryHealth(
                    name,
                    "unhealthy",
                    f"help output was not recognized (exit {completed.returncode})",
                    completed.returncode,
                )

    required_healthy = all(results[name].status == "healthy" for name in REQUIRED_BINARIES)
    status = installation.status if required_healthy else "unhealthy"
    message = (
        "Required deedsBCV binaries passed non-registration help probes."
        if required_healthy
        else "One or more required deedsBCV binaries failed the health check."
    )
    return replace(installation, status=status, message=message, health=results)


def installation_summary(installation: DeedsBCVInstallation) -> dict[str, object]:
    """Return a privacy-safe, JSON-serializable installation summary."""
    return {
        "status": installation.status,
        "source": installation.source,
        "bin_dir": display_path(installation.bin_dir) if installation.bin_dir else None,
        "binaries": {
            name: {
                "found": path is not None,
                "path": display_path(path) if path is not None else None,
                "required": name in REQUIRED_BINARIES,
            }
            for name, path in installation.binaries.items()
        },
        "health": {
            name: {
                "status": result.status,
                "message": result.message,
                "returncode": result.returncode,
            }
            for name, result in installation.health.items()
        },
        "message": installation.message,
        "patch": _patch_summary(installation.patch),
    }


def validate_project_local_build(
    installation: DeedsBCVInstallation,
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    patch_path: str | Path = DEFAULT_PATCH_PATH,
    expected_commit: str = EXPECTED_PINNED_COMMIT,
) -> DeedsBCVInstallation:
    """Validate provenance metadata only for the managed project-local build."""
    source = Path(source_dir).expanduser().resolve(strict=False)
    patch = Path(patch_path).expanduser().resolve(strict=False)
    bin_dir = installation.bin_dir
    if bin_dir is None:
        return installation
    marker = bin_dir / BUILD_STATE_FILENAME
    patch_hash = _sha256_file(patch) if patch.is_file() else None
    commit = _git_output(source, "rev-parse", "HEAD")
    source_patched = False
    if patch.is_file() and commit == expected_commit:
        try:
            source_diff = subprocess.run(
                ["git", "-C", str(source), "diff", "--binary", "--no-ext-diff", "--", "src"],
                shell=False,
                capture_output=True,
                check=True,
            ).stdout
            source_patched = source_diff == patch.read_bytes()
        except (OSError, subprocess.SubprocessError):
            source_patched = False

    problem: str | None = None
    state: dict[str, object] = {}
    if not patch.is_file():
        problem = "tracked patch is missing"
    elif commit != expected_commit:
        problem = "project-local source is not at the pinned commit"
    elif not source_patched:
        problem = "project-local source does not exactly match the tracked patch"
    elif not marker.is_file():
        problem = "build-state marker is missing"
    else:
        try:
            state = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            problem = "build-state marker is unreadable"
    if problem is None and state.get("pinned_commit") != expected_commit:
        problem = "build-state pinned commit differs"
    if problem is None and state.get("patch_sha256") != patch_hash:
        problem = "build-state patch SHA-256 differs"
    build_epoch = state.get("timestamp_epoch")
    if problem is None and not isinstance(build_epoch, (int, float)):
        problem = "build-state timestamp is missing"
    if problem is None:
        for name in REQUIRED_BINARIES:
            executable = installation.binaries.get(name)
            if executable is None or executable.stat().st_mtime < float(build_epoch):
                problem = f"required binary is stale: {name}"
                break

    patch_health = PatchHealth(
        project_local=True,
        status="healthy" if problem is None else "unhealthy",
        message=(
            "Project-local source and required binaries match the tracked patch."
            if problem is None
            else problem
        ),
        pinned_commit_expected=expected_commit,
        pinned_commit_found=commit,
        patch_sha256=patch_hash,
        source_patched=source_patched,
        build_state_found=marker.is_file(),
    )
    if problem is None:
        return replace(installation, patch=patch_health)
    return replace(installation, status="unhealthy", message=problem, patch=patch_health)


def _patch_summary(patch: PatchHealth | None) -> dict[str, object]:
    if patch is None:
        return {
            "project_local": False,
            "status": "not_required",
            "message": "External installations do not require project-local patch metadata.",
            "pinned_commit_expected": EXPECTED_PINNED_COMMIT,
            "patch_expected": DEFAULT_PATCH_PATH.name,
            "patch_sha256": _sha256_file(DEFAULT_PATCH_PATH) if DEFAULT_PATCH_PATH.is_file() else None,
            "source_patched": None,
            "build_state_found": None,
        }
    return {
        "project_local": patch.project_local,
        "status": patch.status,
        "message": patch.message,
        "pinned_commit_expected": patch.pinned_commit_expected,
        "pinned_commit_found": patch.pinned_commit_found,
        "patch_expected": patch.patch_expected,
        "patch_sha256": patch.patch_sha256,
        "source_patched": patch.source_patched,
        "build_state_found": patch.build_state_found,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(directory: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), *arguments],
            shell=False,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _installation_from_paths(
    binaries: dict[str, Path | None], *, source: str, bin_dir: Path | None
) -> DeedsBCVInstallation:
    required_count = sum(binaries[name] is not None for name in REQUIRED_BINARIES)
    found_count = sum(value is not None for value in binaries.values())
    if required_count == len(REQUIRED_BINARIES):
        status = "available"
        message = "Required deedsBCV binaries were found."
    elif found_count:
        status = "partial"
        message = "Only part of the deedsBCV installation was found."
    else:
        status = "not_found"
        message = "deedsBCV binaries were not found in configured locations or PATH."
    return DeedsBCVInstallation(status, binaries, source, bin_dir, message)


def _which_path(
    name: str, *, path: str | None, which: Callable[..., str | None]
) -> Path | None:
    result = which(name, path=path)
    if result is None:
        return None
    candidate = Path(result).expanduser().resolve(strict=False)
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def _read_config_bin_dir(config_path: str | Path) -> str | Path | None:
    path = Path(config_path).expanduser()
    if not path.is_file():
        return None
    with path.open() as stream:
        config = yaml.safe_load(stream) or {}
    registration = config.get("registration", {})
    return registration.get("deeds_bin_dir") if isinstance(registration, dict) else None


def _recognizable_help(name: str, output: str) -> bool:
    normalized = output.lower()
    tokens: Sequence[str] = ("usage", name.lower(), "fixed.nii", "moving.nii")
    return any(token in normalized for token in tokens)
