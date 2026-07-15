from __future__ import annotations

from dataclasses import dataclass, field, replace
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
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class BinaryHealth:
    name: str
    status: str
    message: str
    returncode: int | None = None


@dataclass(frozen=True)
class DeedsBCVInstallation:
    status: str
    binaries: dict[str, Path | None]
    source: str | None = None
    bin_dir: Path | None = None
    message: str | None = None
    health: dict[str, BinaryHealth] = field(default_factory=dict)

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
            return _installation_from_paths(binaries, source=source, bin_dir=directory)

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
    }


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
