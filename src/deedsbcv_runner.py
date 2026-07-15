from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from deedsbcv_tools import DeedsBCVInstallation, discover_deedsbcv
from project_paths import DEFAULT_RAW_DATA_DIR, PROJECT_ROOT, display_path
from registration_inputs import (
    DEFAULT_ISOTROPY_TOLERANCE,
    compare_registration_geometry,
    sha256_file,
    validate_registration_image,
)
from version import PIPELINE_VERSION


PINNED_UPSTREAM_COMMIT = "00b46539aa03d23988a241173d8f1f6c0c68dc6a"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
FLOAT_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z])"
)


class RegistrationRunError(RuntimeError):
    """Raised after a failed stage has been recorded in the local manifest."""

    def __init__(self, message: str, *, stage_details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.stage_details = dict(stage_details or {})


def build_linear_command(
    executable: str | Path,
    fixed: str | Path,
    moving: str | Path,
    output_prefix: str | Path,
) -> list[str]:
    return [str(executable), "-F", str(fixed), "-M", str(moving), "-O", str(output_prefix)]


def build_deformable_command(
    executable: str | Path,
    fixed: str | Path,
    moving: str | Path,
    output_prefix: str | Path,
    affine_matrix: str | Path,
) -> list[str]:
    return [
        str(executable),
        "-F",
        str(fixed),
        "-M",
        str(moving),
        "-O",
        str(output_prefix),
        "-A",
        str(affine_matrix),
    ]


def validate_prepared_inputs(
    prepared_dir: str | Path | None = None,
    *,
    case_id: str,
    experiment: str,
    fixed_path: str | Path | None = None,
    moving_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    isotropy_tolerance: float = DEFAULT_ISOTROPY_TOLERANCE,
) -> dict[str, Any]:
    explicit_values = (fixed_path, moving_path, manifest_path)
    if prepared_dir is not None and any(value is not None for value in explicit_values):
        raise ValueError("Prepared-directory mode cannot be combined with explicit prepared files.")
    if prepared_dir is not None:
        directory = Path(prepared_dir).expanduser().resolve(strict=False)
        manifest_file = directory / "preprocessing_manifest.json"
        fixed_file = directory / "fixed.nii.gz"
        moving_file = directory / "moving.nii.gz"
    elif all(value is not None for value in explicit_values):
        manifest_file = Path(manifest_path).expanduser().resolve(strict=False)  # type: ignore[arg-type]
        fixed_file = Path(fixed_path).expanduser().resolve(strict=False)  # type: ignore[arg-type]
        moving_file = Path(moving_path).expanduser().resolve(strict=False)  # type: ignore[arg-type]
        directory = manifest_file.parent
    else:
        raise ValueError(
            "Provide a prepared directory or all three explicit fixed, moving, and manifest paths."
        )
    if not manifest_file.is_file():
        raise FileNotFoundError("Prepared-input manifest is missing.")
    try:
        manifest = json.loads(manifest_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Prepared-input manifest is unreadable or malformed.") from exc
    if manifest.get("completion_status") != "complete":
        raise ValueError("Prepared-input manifest is not complete.")
    if manifest.get("case_id") != case_id or manifest.get("experiment") != experiment:
        raise ValueError("Prepared-input manifest case ID or experiment does not match the request.")
    roles = manifest.get("role_assignment")
    if not isinstance(roles, dict) or not roles.get("fixed") or not roles.get("moving"):
        raise ValueError("Prepared-input manifest does not contain explicit fixed/moving roles.")
    comparison_manifest = manifest.get("geometry_comparison", {})
    if (
        comparison_manifest.get("classification") != "same-grid"
        or comparison_manifest.get("resampling_required") is not False
    ):
        raise ValueError("Prepared-input manifest does not describe an unresampled same-grid pair.")
    for role, path in (("fixed", fixed_file), ("moving", moving_file)):
        if not path.is_file():
            raise FileNotFoundError(f"Prepared {role} working copy is missing.")
        if not path.name.lower().endswith(".nii.gz"):
            raise ValueError(f"Prepared {role} working copy must end with .nii.gz.")

    fixed = validate_registration_image(
        fixed_file, role="prepared fixed", isotropy_tolerance=isotropy_tolerance
    )
    moving = validate_registration_image(
        moving_file, role="prepared moving", isotropy_tolerance=isotropy_tolerance
    )
    comparison = compare_registration_geometry(fixed, moving)
    if comparison["classification"] != "same-grid":
        raise ValueError(
            f"Prepared working copies are not registration-ready: {comparison['classification']}."
        )
    if fixed.summary["anisotropic_spacing"] or moving.summary["anisotropic_spacing"]:
        raise ValueError("Prepared working copies are anisotropic.")

    expected_hashes = manifest.get("checksums_sha256", {}).get("working_copies", {})
    if not expected_hashes.get("fixed") or not expected_hashes.get("moving"):
        raise ValueError("Prepared-input manifest lacks working-copy checksums.")
    actual_hashes = {"fixed": sha256_file(fixed_file), "moving": sha256_file(moving_file)}
    if actual_hashes != {"fixed": expected_hashes["fixed"], "moving": expected_hashes["moving"]}:
        raise ValueError("Prepared working copies are stale or differ from their manifest checksums.")
    return {
        "directory": directory,
        "manifest_path": manifest_file,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_file),
        "fixed_path": fixed_file,
        "moving_path": moving_file,
        "fixed": fixed,
        "moving": moving,
        "comparison": comparison,
        "input_hashes": actual_hashes,
    }


def verify_affine_matrix(path: str | Path) -> dict[str, Any]:
    matrix_path = Path(path)
    if not matrix_path.is_file() or matrix_path.stat().st_size == 0:
        raise ValueError("Expected affine matrix is missing or empty.")
    try:
        text = matrix_path.read_text()
    except (OSError, UnicodeError) as exc:
        raise ValueError("Affine matrix is not readable text.") from exc
    lowered = text.lower()
    if "nan" in lowered or "inf" in lowered:
        raise ValueError("Affine matrix contains NaN or infinity.")
    values = [float(match.group(0)) for match in FLOAT_PATTERN.finditer(text)]
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise ValueError("Affine matrix must contain exactly 16 finite numeric values.")
    return {
        "passed": True,
        "numeric_value_count": 16,
        "rows": 4,
        "columns": 4,
        "file_size_bytes": matrix_path.stat().st_size,
        "value_min": min(values),
        "value_max": max(values),
    }


def verify_deformable_outputs(
    registered_path: str | Path,
    displacement_path: str | Path,
    *,
    fixed: Any,
    isotropy_tolerance: float = DEFAULT_ISOTROPY_TOLERANCE,
) -> dict[str, Any]:
    registered = Path(registered_path)
    displacement = Path(displacement_path)
    if not registered.is_file() or registered.stat().st_size == 0:
        raise ValueError("Expected registered moving image is missing or empty.")
    if not displacement.is_file() or displacement.stat().st_size == 0:
        raise ValueError("Expected displacement output is missing or empty.")
    output = validate_registration_image(
        registered, role="registered moving", isotropy_tolerance=isotropy_tolerance
    )
    dimensions_match = output.image.GetSize() == fixed.image.GetSize()
    if not dimensions_match:
        raise ValueError("Registered moving image dimensions do not match the fixed input.")
    spacing_matches = bool(
        np.allclose(output.image.GetSpacing(), fixed.image.GetSpacing(), atol=1e-5, rtol=0)
    )
    origin_matches = bool(
        np.allclose(output.image.GetOrigin(), fixed.image.GetOrigin(), atol=1e-5, rtol=0)
    )
    direction_matches = bool(
        np.allclose(output.image.GetDirection(), fixed.image.GetDirection(), atol=1e-5, rtol=0)
    )
    geometry_matches = spacing_matches and origin_matches and direction_matches
    return {
        "passed": True,
        "registered_image_size_bytes": registered.stat().st_size,
        "displacement_size_bytes": displacement.stat().st_size,
        "dimensions_match_fixed": True,
        "spacing_matches_fixed": spacing_matches,
        "origin_matches_fixed": origin_matches,
        "direction_matches_fixed": direction_matches,
        "full_geometry_matches_fixed": geometry_matches,
        "registered_geometry": _public_geometry(output.summary),
    }


def run_deedsbcv_registration(
    *,
    prepared_dir: str | Path | None = None,
    fixed_path: str | Path | None = None,
    moving_path: str | Path | None = None,
    preprocessing_manifest: str | Path | None = None,
    case_id: str,
    experiment: str = "full",
    bin_dir: str | Path | None = None,
    timeout_seconds: float | None = None,
    resume: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    process_factory: Callable[..., Any] = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be combined.")
    timeout = _resolve_timeout(timeout_seconds)
    prepared = validate_prepared_inputs(
        prepared_dir,
        case_id=case_id,
        experiment=experiment,
        fixed_path=fixed_path,
        moving_path=moving_path,
        manifest_path=preprocessing_manifest,
    )
    installation = discover_deedsbcv(bin_dir)
    if installation.status != "available" or not installation.required_available:
        raise FileNotFoundError(installation.message or "Required deedsBCV binaries are unavailable.")
    linear = installation.binaries["linearBCV"]
    deeds = installation.binaries["deedsBCV"]
    if linear is None or deeds is None:  # defensive narrowing
        raise FileNotFoundError("Required deedsBCV binaries are unavailable.")

    experiment_dir = prepared["directory"].parent
    _reject_raw_output(experiment_dir)
    affine_dir = experiment_dir / "affine"
    deformable_dir = experiment_dir / "deformable"
    manifest_path = experiment_dir / "registration_manifest.json"
    paths = {
        "affine_prefix": affine_dir / "affine",
        "affine_matrix": affine_dir / "affine_matrix.txt",
        "linear_stdout": affine_dir / "linear_stdout.log",
        "linear_stderr": affine_dir / "linear_stderr.log",
        "deformable_prefix": deformable_dir / "deformable",
        "upstream_registered": deformable_dir / "deformable_deformed.nii.gz",
        "upstream_displacement": deformable_dir / "deformable_displacements.dat",
        "registered": deformable_dir / "moving_registered.nii.gz",
        "displacement": deformable_dir / "displacements.dat",
        "deeds_stdout": deformable_dir / "deeds_stdout.log",
        "deeds_stderr": deformable_dir / "deeds_stderr.log",
        "manifest": manifest_path,
    }
    linear_command = build_linear_command(
        linear, prepared["fixed_path"], prepared["moving_path"], paths["affine_prefix"]
    )
    deformable_command = build_deformable_command(
        deeds,
        prepared["fixed_path"],
        prepared["moving_path"],
        paths["deformable_prefix"],
        paths["affine_matrix"],
    )
    manifest = _initial_manifest(
        case_id,
        experiment,
        prepared,
        installation,
        linear_command,
        deformable_command,
        paths,
        timeout,
        dry_run=dry_run,
    )
    if dry_run:
        return manifest

    existing_manifest = _read_registration_manifest(manifest_path)
    known_outputs = [path for key, path in paths.items() if key not in {"affine_prefix", "deformable_prefix"}]
    if overwrite:
        _unlink_known_outputs(known_outputs)
        existing_manifest = None
    elif not resume and any(path.exists() for path in known_outputs):
        raise FileExistsError(
            "Registration outputs already exist for this case/experiment; use --resume or --overwrite."
        )

    affine_reused = False
    deformable_reused = False
    if resume and _manifest_matches_current(existing_manifest, manifest):
        affine_state = (existing_manifest or {}).get("stages", {}).get("affine", {})
        if affine_state.get("verification", {}).get("passed"):
            try:
                affine_verification = verify_affine_matrix(paths["affine_matrix"])
                affine_reused = True
            except ValueError:
                affine_reused = False
        deformable_state = (existing_manifest or {}).get("stages", {}).get("deformable", {})
        if affine_reused and deformable_state.get("verification", {}).get("passed"):
            try:
                deformable_verification = verify_deformable_outputs(
                    paths["registered"], paths["displacement"], fixed=prepared["fixed"]
                )
                deformable_reused = True
            except ValueError:
                deformable_reused = False

    manifest["completion_status"] = "in-progress"
    affine_dir.mkdir(parents=True, exist_ok=True)
    deformable_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, manifest)
    try:
        if affine_reused:
            manifest["stages"]["affine"] = {
                **(existing_manifest or {})["stages"]["affine"],
                "status": "reused",
                "verification": affine_verification,
            }
        else:
            _unlink_known_outputs(
                [paths["affine_matrix"], paths["linear_stdout"], paths["linear_stderr"]]
            )
            result = _execute_stage(
                "affine",
                linear_command,
                cwd=experiment_dir,
                stdout_path=paths["linear_stdout"],
                stderr_path=paths["linear_stderr"],
                timeout=timeout,
                process_factory=process_factory,
                clock=clock,
            )
            manifest["stages"]["affine"] = {**result, "status": "verifying"}
            affine_verification = verify_affine_matrix(paths["affine_matrix"])
            manifest["stages"]["affine"] = {
                **result,
                "status": "complete",
                "verification": affine_verification,
            }
        _write_manifest(manifest_path, manifest)

        if deformable_reused:
            manifest["stages"]["deformable"] = {
                **(existing_manifest or {})["stages"]["deformable"],
                "status": "reused",
                "verification": deformable_verification,
            }
        else:
            _unlink_known_outputs(
                [
                    paths["upstream_registered"],
                    paths["upstream_displacement"],
                    paths["registered"],
                    paths["displacement"],
                    paths["deeds_stdout"],
                    paths["deeds_stderr"],
                ]
            )
            result = _execute_stage(
                "deformable",
                deformable_command,
                cwd=experiment_dir,
                stdout_path=paths["deeds_stdout"],
                stderr_path=paths["deeds_stderr"],
                timeout=timeout,
                process_factory=process_factory,
                clock=clock,
            )
            manifest["stages"]["deformable"] = {**result, "status": "verifying"}
            deformable_verification = verify_deformable_outputs(
                paths["upstream_registered"],
                paths["upstream_displacement"],
                fixed=prepared["fixed"],
            )
            paths["upstream_registered"].replace(paths["registered"])
            paths["upstream_displacement"].replace(paths["displacement"])
            deformable_verification["registered_image_size_bytes"] = paths["registered"].stat().st_size
            deformable_verification["displacement_size_bytes"] = paths["displacement"].stat().st_size
            manifest["stages"]["deformable"] = {
                **result,
                "status": "complete",
                "verification": deformable_verification,
            }
        if not deformable_verification["full_geometry_matches_fixed"]:
            manifest["warnings"].append(
                "Registered image dimensions match the fixed image, but physical geometry metadata differ; no metadata correction was applied."
            )
        manifest["completion_status"] = "complete"
        manifest["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        return manifest
    except (Exception, KeyboardInterrupt) as exc:
        manifest["completion_status"] = "incomplete"
        failed_stage = (
            "affine"
            if manifest["stages"]["affine"].get("status") not in {"complete", "reused"}
            else "deformable"
        )
        manifest["stages"][failed_stage] = {
            **manifest["stages"][failed_stage],
            **(exc.stage_details if isinstance(exc, RegistrationRunError) else {}),
            "status": "failed",
        }
        if failed_stage == "affine":
            manifest["stages"][failed_stage].setdefault(
                "stdout_log", display_path(paths["linear_stdout"])
            )
            manifest["stages"][failed_stage].setdefault(
                "stderr_log", display_path(paths["linear_stderr"])
            )
        else:
            manifest["stages"][failed_stage].setdefault(
                "stdout_log", display_path(paths["deeds_stdout"])
            )
            manifest["stages"][failed_stage].setdefault(
                "stderr_log", display_path(paths["deeds_stderr"])
            )
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": _safe_failure_message(exc),
        }
        manifest["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, RegistrationRunError):
            raise
        raise RegistrationRunError(str(manifest["failure"]["message"])) from exc


def _execute_stage(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
    process_factory: Callable[..., Any],
    clock: Callable[[], float],
) -> dict[str, Any]:
    started = clock()
    process = None
    stdout = ""
    stderr = ""
    try:
        process = process_factory(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout = _as_text(exc.output)
        stderr = _as_text(exc.stderr)
        if process is not None:
            _terminate_process(process)
        _write_logs(stdout_path, stderr_path, stdout, stderr)
        raise RegistrationRunError(
            f"{name} stage timed out after {timeout:g} seconds.",
            stage_details={
                "runtime_seconds": clock() - started,
                "return_code": None,
                "stdout_log": display_path(stdout_path),
                "stderr_log": display_path(stderr_path),
            },
        ) from exc
    except KeyboardInterrupt:
        if process is not None:
            _terminate_process(process)
        _write_logs(stdout_path, stderr_path, stdout, stderr)
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        if process is not None:
            _terminate_process(process)
        _write_logs(stdout_path, stderr_path, stdout, stderr)
        raise RegistrationRunError(
            f"{name} stage could not be executed; inspect local logs.",
            stage_details={
                "runtime_seconds": clock() - started,
                "return_code": None,
                "stdout_log": display_path(stdout_path),
                "stderr_log": display_path(stderr_path),
            },
        ) from exc
    runtime = clock() - started
    stdout = stdout or ""
    stderr = stderr or ""
    _write_logs(stdout_path, stderr_path, stdout, stderr)
    returncode = process.returncode
    if returncode != 0:
        raise RegistrationRunError(
            f"{name} stage exited with status {returncode}.",
            stage_details={
                "runtime_seconds": runtime,
                "return_code": returncode,
                "stdout_log": display_path(stdout_path),
                "stderr_log": display_path(stderr_path),
            },
        )
    return {
        "runtime_seconds": runtime,
        "return_code": returncode,
        "stdout_log": display_path(stdout_path),
        "stderr_log": display_path(stderr_path),
    }


def _terminate_process(process: Any) -> None:
    try:
        process.terminate()
        process.wait(timeout=2)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=2)
        except Exception:
            pass


def _initial_manifest(
    case_id: str,
    experiment: str,
    prepared: Mapping[str, Any],
    installation: DeedsBCVInstallation,
    linear_command: list[str],
    deformable_command: list[str],
    paths: Mapping[str, Path],
    timeout: float,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    roles = prepared["manifest"]["role_assignment"]
    warnings = list(prepared["manifest"].get("warnings", []))
    return {
        "pipeline_version": PIPELINE_VERSION,
        "case_id": case_id,
        "experiment": experiment,
        "role_assignment": roles,
        "prepared_inputs": {
            "directory": display_path(prepared["directory"]),
            "manifest": display_path(prepared["manifest_path"]),
            "fixed": display_path(prepared["fixed_path"]),
            "moving": display_path(prepared["moving_path"]),
            "manifest_sha256": prepared["manifest_sha256"],
            "working_copy_sha256": prepared["input_hashes"],
        },
        "fixed_geometry": _public_geometry(prepared["fixed"].summary),
        "moving_geometry": _public_geometry(prepared["moving"].summary),
        "deedsbcv": {
            "installation_source": installation.source,
            "pinned_upstream_commit": PINNED_UPSTREAM_COMMIT,
            "binaries": {
                name: {
                    "filename": path.name if path else None,
                    "path": display_path(path) if path else None,
                    "sha256": sha256_file(path) if path else None,
                }
                for name, path in installation.binaries.items()
                if name in {"linearBCV", "deedsBCV"}
            },
        },
        "commands": {
            "affine": _safe_command(linear_command),
            "deformable": _safe_command(deformable_command),
            "default_parameters_by_omission": True,
        },
        "timeout_seconds_per_stage": timeout,
        "output_name_mapping": {
            "affine_matrix": {
                "upstream": display_path(paths["affine_matrix"]),
                "stable": display_path(paths["affine_matrix"]),
            },
            "registered_moving": {
                "upstream": display_path(paths["upstream_registered"]),
                "stable": display_path(paths["registered"]),
            },
            "displacements": {
                "upstream": display_path(paths["upstream_displacement"]),
                "stable": display_path(paths["displacement"]),
            },
        },
        "outputs": {
            "affine_matrix": display_path(paths["affine_matrix"]),
            "registered_moving": display_path(paths["registered"]),
            "displacements": display_path(paths["displacement"]),
            "manifest": display_path(paths["manifest"]),
        },
        "stages": {
            "affine": {"status": "planned"},
            "deformable": {"status": "planned"},
        },
        "warnings": warnings,
        "completion_status": "dry-run" if dry_run else "planned",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Research-only and non-diagnostic.",
    }


def _manifest_matches_current(existing: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    if not existing:
        return False
    return (
        existing.get("case_id") == current.get("case_id")
        and existing.get("experiment") == current.get("experiment")
        and existing.get("prepared_inputs") == current.get("prepared_inputs")
        and existing.get("commands") == current.get("commands")
        and existing.get("deedsbcv") == current.get("deedsbcv")
    )


def _read_registration_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def _write_logs(stdout_path: Path, stderr_path: Path, stdout: str, stderr: str) -> None:
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)


def _unlink_known_outputs(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _safe_command(command: Sequence[str]) -> list[str]:
    safe = []
    for index, item in enumerate(command):
        if item.startswith("-"):
            safe.append(item)
        elif index == 0 or index % 2 == 0:
            safe.append(display_path(item))
        else:
            safe.append(item)
    return safe


def _public_geometry(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "pixel_id"}


def _resolve_timeout(value: float | None) -> float:
    if value is None:
        with DEFAULT_CONFIG_PATH.open() as stream:
            config = yaml.safe_load(stream) or {}
        value = config.get("registration", {}).get("timeout_seconds", 3600)
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive.")
    return timeout


def _reject_raw_output(path: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(DEFAULT_RAW_DATA_DIR.resolve(strict=False))
    except ValueError:
        return
    raise ValueError("Refusing registration outputs under data/raw/.")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _safe_failure_message(exc: BaseException) -> str:
    if isinstance(exc, (RegistrationRunError, ValueError)):
        return str(exc)
    return f"{type(exc).__name__} during registration stage; inspect local logs."
