from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lung_ct_lung_masked_difference_mpl"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import SimpleITK as sitk

from project_paths import DEFAULT_OUTPUTS_DIR, PROJECT_ROOT, display_path
from registration_difference_comparison import (
    COLORBAR_LABEL,
    FORMULAS,
    PANEL_KEYS,
    PANEL_LABELS,
    _aspect,
    _exact_geometry_mismatches,
    _extract_slice,
    _geometry_dict,
    _load_complete_registration,
    _matched_slice_indices,
)
from registration_inputs import sha256_file, validate_registration_image
from version import PIPELINE_VERSION


ANALYSIS_TYPE = "Case 1-versus-Case 2 lung-masked turbo difference visualization"
EXPECTED_SHARED_WINDOW_ANALYSIS = (
    "Case 1-versus-Case 2 equal-window registration difference visualization"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OUTPUTS_DIR
    / "registration"
    / "case_comparison"
    / "equal_window_registration_differences"
    / "lung_masked_turbo_difference"
)
OUTPUT_NAMES = {
    "axial": "lung_masked_turbo_axial_comparison.png",
    "multiplanar": "lung_masked_turbo_multiplanar_comparison.png",
    "report": "lung_masked_turbo_difference_report.json",
}
CASE_KEYS = ("case_1", "case_2")


def render_lung_masked_turbo_rgb(
    difference: np.ndarray,
    lung_mask: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    opacity: float = 0.45,
) -> np.ndarray:
    """Render grayscale difference everywhere and masked turbo inside lungs."""
    values = np.asarray(difference, dtype=np.float32)
    mask = np.asarray(lung_mask, dtype=bool)
    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError("Difference and lung-mask slices must have identical 2D shapes.")
    _validate_limits_and_opacity(vmin, vmax, opacity)
    if not np.isfinite(values).all():
        raise ValueError("Difference slice must contain only finite values.")
    normalized = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    gray_rgb = np.repeat(normalized[..., None], 3, axis=2)
    turbo_rgb = matplotlib.colormaps["turbo"](normalized)[..., :3]
    result = gray_rgb.copy()
    result[mask] = (1.0 - opacity) * gray_rgb[mask] + opacity * turbo_rgb[mask]
    return np.ascontiguousarray(np.clip(result, 0, 1))


def create_lung_masked_difference_figures(
    *,
    case_1_dir: str | Path,
    case_2_dir: str | Path,
    case_1_lung_mask: str | Path,
    case_2_lung_mask: str | Path,
    shared_window_report: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    opacity: float = 0.45,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create the professor-requested masked-color, whole-difference figures."""
    _validate_limits_and_opacity(0.0, 1.0, opacity)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    outputs = {key: destination / name for key, name in OUTPUT_NAMES.items()}
    if any(path.exists() for path in outputs.values()) and not overwrite and not dry_run:
        raise FileExistsError("Lung-masked difference outputs already exist; use --overwrite.")

    context = prepare_lung_masked_inputs(
        case_1_dir=case_1_dir,
        case_2_dir=case_2_dir,
        case_1_lung_mask=case_1_lung_mask,
        case_2_lung_mask=case_2_lung_mask,
        shared_window_report=shared_window_report,
    )
    vmin, vmax = context["limits"]
    arrays = context["arrays"]
    masks = context["masks"]
    slices, physical = _matched_slice_indices(context["cases"], context["cross_case_mismatches"])
    clipping = _mask_and_clipping_statistics(arrays, masks, vmax)
    panel_limits = {
        "axial": {key: {"vmin": vmin, "vmax": vmax} for key in PANEL_KEYS},
        "multiplanar": {
            view: {key: {"vmin": vmin, "vmax": vmax} for key in PANEL_KEYS}
            for view in ("axial", "coronal", "sagittal")
        },
    }
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "analysis_type": ANALYSIS_TYPE,
        "intended_use": "Research-only, descriptive, visualization-only, and non-diagnostic.",
        "inputs": context["input_report"],
        "geometry_validation": context["geometry_report"],
        "formulas": FORMULAS,
        "slice_selection": {
            "indices_numpy_zyx": slices,
            "physical_points_xyz_mm": physical,
        },
        "shared_difference_window": {
            "source_report": display_path(context["report_path"]),
            "percentile": context["shared_report"]["shared_difference_window"]["percentile"],
            "vmin": vmin,
            "vmax": vmax,
            "identical_for_every_panel_and_component": True,
            "panel_limits": panel_limits,
        },
        "display": {
            "background_everywhere": "grayscale abs(fixed - candidate)",
            "inside_lung_mask": "(1-opacity)*grayscale difference + opacity*turbo difference",
            "outside_lung_mask": "grayscale abs(fixed - candidate), retained and not zeroed",
            "fixed_anatomy_used_as_background": False,
            "lung_mask_restricts_analysis": False,
            "grayscale_colormap": "gray",
            "inside_mask_colormap": "turbo",
            "opacity": float(opacity),
            "colorbar_label": COLORBAR_LABEL,
            "physical_aspect_ratios_preserved": True,
        },
        "mask_and_clipping_statistics": clipping,
        "outputs": {key: display_path(path) for key, path in outputs.items()},
        "output_checksums_sha256": None,
        "source_immutability": {"verified_before_and_after": None, "unchanged": None},
        "warnings": [
            "Research-only and non-diagnostic; absolute intensity difference is not a pure registration-error measurement.",
            "The lung mask controls color placement only; the full-image difference remains visible and analytical inputs are unchanged.",
            "Values above the stored shared vmax are visually clipped.",
        ],
        "completion_status": "dry-run" if dry_run else "pending",
        "writes_performed": False if dry_run else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        return report

    destination.mkdir(parents=True, exist_ok=True)
    _render_axial(context, slices, outputs["axial"], opacity)
    _render_multiplanar(context, slices, outputs["multiplanar"], opacity)
    hashes_after = _source_hashes(context["source_paths"])
    if hashes_after != context["hashes_before"]:
        raise RuntimeError("A source NIfTI, mask, manifest, or shared report changed during rendering.")
    report["source_immutability"] = {
        "verified_before_and_after": True,
        "unchanged": True,
        "sha256_after": hashes_after,
    }
    report["output_checksums_sha256"] = {
        key: sha256_file(path) for key, path in outputs.items() if key != "report"
    }
    report["completion_status"] = "complete"
    report["writes_performed"] = True
    outputs["report"].write_text(json.dumps(report, indent=2) + "\n")
    return report


def prepare_lung_masked_inputs(
    *, case_1_dir, case_2_dir, case_1_lung_mask, case_2_lung_mask, shared_window_report
) -> dict[str, Any]:
    case_dirs = {
        "case_1": Path(case_1_dir).expanduser().resolve(strict=False),
        "case_2": Path(case_2_dir).expanduser().resolve(strict=False),
    }
    mask_paths = {
        "case_1": Path(case_1_lung_mask).expanduser().resolve(strict=False),
        "case_2": Path(case_2_lung_mask).expanduser().resolve(strict=False),
    }
    report_path = Path(shared_window_report).expanduser().resolve(strict=False)
    shared_report = _load_shared_report(report_path)
    registrations = {case: _load_complete_registration(case, path) for case, path in case_dirs.items()}
    cases = {}
    paths = {}
    masks = {}
    mask_manifests = {}
    for case in CASE_KEYS:
        paths[case] = {
            "fixed": case_dirs[case] / "full/preprocessed/fixed.nii.gz",
            "moving": case_dirs[case] / "full/preprocessed/moving.nii.gz",
            "registered": case_dirs[case] / "full/deformable/moving_registered.nii.gz",
            "lung_mask": mask_paths[case],
        }
        loaded = {
            role: validate_registration_image(path, role=f"{case} {role}")
            for role, path in paths[case].items()
        }
        for role in ("moving", "registered", "lung_mask"):
            mismatches = _exact_geometry_mismatches(loaded["fixed"].image, loaded[role].image)
            if mismatches:
                raise ValueError(
                    f"{case.replace('_', ' ').title()} fixed/{role} exact geometry mismatch: "
                    + "; ".join(mismatches)
                    + ". Inputs were not resampled or altered."
                )
        mask_array = sitk.GetArrayViewFromImage(loaded["lung_mask"].image)
        unique = np.unique(mask_array)
        if unique.size > 2 or not set(float(value) for value in unique).issubset({0.0, 1.0}):
            raise ValueError(f"{case.replace('_', ' ').title()} lung mask must be binary with values 0 and 1.")
        if not np.any(mask_array > 0):
            raise ValueError(f"{case.replace('_', ' ').title()} lung mask is empty.")
        masks[case] = mask_array > 0
        cases[case] = loaded
        mask_manifests[case] = _validate_mask_manifest(mask_paths[case], paths[case]["fixed"], case)

    hashes = {
        case: {role: sha256_file(path) for role, path in paths[case].items()}
        for case in CASE_KEYS
    }
    _validate_shared_report_inputs(shared_report, paths, hashes)
    cross = _exact_geometry_mismatches(cases["case_1"]["fixed"].image, cases["case_2"]["fixed"].image)
    source_paths = {
        str(path): path
        for path in [
            report_path,
            *(registrations[c]["path"] for c in CASE_KEYS),
            *(mask_manifests[c]["path"] for c in CASE_KEYS),
            *(paths[c][role] for c in CASE_KEYS for role in paths[c]),
        ]
    }
    hashes_before = _source_hashes(source_paths)
    arrays = {
        case: {
            role: sitk.GetArrayViewFromImage(cases[case][role].image)
            for role in ("fixed", "moving", "registered")
        }
        for case in CASE_KEYS
    }
    window = shared_report["shared_difference_window"]
    input_report = {
        "shared_window_report": {"path": display_path(report_path), "sha256": sha256_file(report_path)},
        "cases": {
            case: {
                "registration_manifest": {
                    "path": display_path(registrations[case]["path"]),
                    "sha256": sha256_file(registrations[case]["path"]),
                },
                "lung_mask_manifest": {
                    "path": display_path(mask_manifests[case]["path"]),
                    "sha256": sha256_file(mask_manifests[case]["path"]),
                },
                **{
                    role: {"path": display_path(path), "sha256": hashes[case][role]}
                    for role, path in paths[case].items()
                },
            }
            for case in CASE_KEYS
        },
    }
    geometry_report = {
        case: {
            "exact_grid_validated": True,
            "fixed_vs_moving_mismatches": [],
            "fixed_vs_registered_mismatches": [],
            "fixed_vs_lung_mask_mismatches": [],
            "fixed": _geometry_dict(cases[case]["fixed"].image),
            "lung_mask": _geometry_dict(cases[case]["lung_mask"].image),
        }
        for case in CASE_KEYS
    }
    geometry_report["cross_case_fixed_grids"] = {
        "agree": not cross,
        "mismatches": cross,
        "slice_policy": "identical voxel indices" if not cross else "matched physical center",
    }
    return {
        "case_dirs": case_dirs,
        "paths": paths,
        "cases": cases,
        "arrays": arrays,
        "masks": masks,
        "limits": (float(window["vmin"]), float(window["vmax"])),
        "shared_report": shared_report,
        "report_path": report_path,
        "cross_case_mismatches": cross,
        "input_report": input_report,
        "geometry_report": geometry_report,
        "source_paths": source_paths,
        "hashes_before": hashes_before,
    }


def _load_shared_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError("Shared-window comparison report does not exist.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Shared-window comparison report is unreadable or malformed.") from exc
    if report.get("completion_status") != "complete":
        raise ValueError("Shared-window comparison report is not complete.")
    if report.get("analysis_type") != EXPECTED_SHARED_WINDOW_ANALYSIS:
        raise ValueError("Shared-window report has an incompatible analysis type.")
    if set(report.get("inputs", {})) != set(CASE_KEYS):
        raise ValueError("Shared-window report must include exactly Case 1 and Case 2.")
    window = report.get("shared_difference_window", {})
    if set(window.get("pooled_volume_order", [])) != set(PANEL_KEYS):
        raise ValueError("Shared-window report does not include all four difference volumes.")
    if window.get("identical_for_every_difference_panel") is not True:
        raise ValueError("Shared-window report does not certify identical normalization.")
    try:
        vmin, vmax = float(window["vmin"]), float(window["vmax"])
        percentile = float(window["percentile"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Shared-window report contains invalid limits or percentile.") from exc
    _validate_limits_and_opacity(vmin, vmax, 0.45)
    if not 0 < percentile <= 100:
        raise ValueError("Shared-window report contains an invalid percentile.")
    for figure in ("grayscale", "turbo"):
        panels = window.get("panel_limits", {}).get(figure, {})
        for key in PANEL_KEYS:
            if panels.get(key) != {"vmin": vmin, "vmax": vmax}:
                raise ValueError("Shared-window report has incompatible panel limits.")
    return report


def _validate_shared_report_inputs(report, paths, hashes):
    for case in CASE_KEYS:
        recorded = report["inputs"][case].get("analytical_images", {})
        for role in ("fixed", "moving", "registered"):
            item = recorded.get(role, {})
            if item.get("sha256") != hashes[case][role]:
                raise ValueError(f"Shared-window report {case} {role} checksum is incompatible.")
            recorded_path = item.get("path")
            if not isinstance(recorded_path, str):
                raise ValueError(f"Shared-window report {case} {role} path is incompatible.")
            if recorded_path != "[external path]":
                if (PROJECT_ROOT / recorded_path).resolve(strict=False) != paths[case][role]:
                    raise ValueError(f"Shared-window report {case} {role} path is incompatible.")


def _validate_mask_manifest(mask_path: Path, fixed_path: Path, case: str) -> dict[str, Any]:
    manifest_path = mask_path.parent.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{case} lung-mask manifest does not exist.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{case} lung-mask manifest is unreadable or malformed.") from exc
    if manifest.get("completion_status") != "success":
        raise ValueError(f"{case} lung-mask manifest is not successful.")
    if manifest.get("combined_mask_filename") != mask_path.name:
        raise ValueError(f"{case} lung-mask path is incompatible with its manifest.")
    recorded_input = manifest.get("input_path")
    if not isinstance(recorded_input, str) or (PROJECT_ROOT / recorded_input).resolve(strict=False) != fixed_path:
        raise ValueError(f"{case} lung-mask manifest does not reference the fixed NIfTI.")
    return {"path": manifest_path, "manifest": manifest}


def _source_hashes(source_paths):
    return {key: sha256_file(path) for key, path in source_paths.items()}


def _difference_slices(context, indices, view):
    result = {}
    mask_slices = {}
    for key in PANEL_KEYS:
        case, state = key.rsplit("_", 1)
        candidate = "moving" if state == "before" else "registered"
        fixed = _extract_slice(context["arrays"][case]["fixed"], indices[case], view)
        other = _extract_slice(context["arrays"][case][candidate], indices[case], view)
        result[key] = np.abs(fixed.astype(np.float32) - other.astype(np.float32))
        mask_slices[key] = _extract_slice(context["masks"][case], indices[case], view)
    return result, mask_slices


def _render_axial(context, indices, path, opacity):
    differences, masks = _difference_slices(context, indices, "axial")
    _render_grid(context, differences, masks, path, opacity, (2, 2), ("axial",) * 4)


def _render_multiplanar(context, indices, path, opacity):
    vmin, vmax = context["limits"]
    figure, axes = plt.subplots(3, 4, figsize=(14, 11), layout="constrained")
    for row, view in enumerate(("axial", "coronal", "sagittal")):
        differences, masks = _difference_slices(context, indices, view)
        for column, key in enumerate(PANEL_KEYS):
            case = key.rsplit("_", 1)[0]
            rgb = render_lung_masked_turbo_rgb(
                differences[key], masks[key], vmin=vmin, vmax=vmax, opacity=opacity
            )
            axes[row, column].imshow(
                rgb, origin="lower", aspect=_aspect(context["cases"][case]["fixed"].image, view)
            )
            axes[row, column].set_title(
                f"{view.title()} — {PANEL_LABELS[key]}\n{FORMULAS[key.rsplit('_', 1)[1]]}", fontsize=9
            )
            axes[row, column].axis("off")
    figure.suptitle(
        "Case 1 vs Case 2 lung-masked turbo differences — multiplanar\n"
        f"gray difference outside lungs; turbo difference inside mask, α={opacity:g}; "
        f"shared window [{vmin:.6g}, {vmax:.6g}] — RESEARCH ONLY",
        fontsize=11,
    )
    _add_colorbar(figure, axes.ravel().tolist(), vmin, vmax)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_grid(context, differences, masks, path, opacity, shape, views):
    vmin, vmax = context["limits"]
    figure, axes = plt.subplots(*shape, figsize=(10, 10), layout="constrained")
    for axis, key, view in zip(axes.flat, PANEL_KEYS, views):
        case = key.rsplit("_", 1)[0]
        rgb = render_lung_masked_turbo_rgb(
            differences[key], masks[key], vmin=vmin, vmax=vmax, opacity=opacity
        )
        axis.imshow(rgb, origin="lower", aspect=_aspect(context["cases"][case]["fixed"].image, view))
        axis.set_title(
            f"{PANEL_LABELS[key]}\n{FORMULAS[key.rsplit('_', 1)[1]]}; lung turbo α={opacity:g}"
        )
        axis.axis("off")
    figure.suptitle(
        "Case 1 vs Case 2 lung-masked turbo differences — axial\n"
        f"gray difference outside lungs; turbo difference inside mask; shared window "
        f"[{vmin:.6g}, {vmax:.6g}] — RESEARCH ONLY",
        fontsize=11,
    )
    _add_colorbar(figure, axes.ravel().tolist(), vmin, vmax)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _add_colorbar(figure, axes, vmin, vmax):
    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="turbo")
    mappable.set_array([])
    figure.colorbar(
        mappable, ax=axes, label=COLORBAR_LABEL,
        orientation="horizontal", shrink=0.82, pad=0.04, aspect=42,
    )


def _mask_and_clipping_statistics(arrays, masks, vmax):
    cases = {}
    volumes = {}
    for case in CASE_KEYS:
        mask = masks[case]
        cases[case] = {
            "mask_voxels": int(np.count_nonzero(mask)),
            "outside_mask_voxels": int(mask.size - np.count_nonzero(mask)),
            "total_voxels": int(mask.size),
            "mask_fraction": float(np.count_nonzero(mask) / mask.size),
            "binary_values_validated": True,
        }
        for state, candidate in (("before", "moving"), ("after", "registered")):
            difference = np.abs(
                arrays[case]["fixed"].astype(np.float32) - arrays[case][candidate].astype(np.float32)
            )
            inside = difference[mask]
            outside = difference[~mask]
            volumes[f"{case}_{state}"] = {
                "inside_mask_above_vmax": int(np.count_nonzero(inside > vmax)),
                "outside_mask_above_vmax": int(np.count_nonzero(outside > vmax)),
                "inside_mask_voxels": int(inside.size),
                "outside_mask_voxels": int(outside.size),
            }
    return {
        "clipping_definition": "absolute-difference voxels strictly greater than shared vmax",
        "cases": cases,
        "by_volume": volumes,
    }


def _validate_limits_and_opacity(vmin, vmax, opacity):
    if not math.isfinite(float(vmin)) or not math.isfinite(float(vmax)) or float(vmin) != 0 or float(vmax) <= float(vmin):
        raise ValueError("Shared difference limits must be finite with vmin=0 and vmax>0.")
    if not 0 <= float(opacity) <= 1:
        raise ValueError("Opacity must be between 0 and 1.")
