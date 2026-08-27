from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lung_ct_case_comparison_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import display_path
from version import PIPELINE_VERSION


ANALYSIS_TYPE = "Report-driven MRI registration case comparison"
OUTPUT_NAMES = {
    "report": "case_1_vs_case_2.json",
    "runtime": "registration_runtime_comparison.png",
    "quality": "registration_quality_comparison.png",
    "intensity": "intensity_change_comparison.png",
    "functional": "functional_lung_comparison.png",
    "normalized": "normalized_intensity_comparison.png",
}
REPORT_PATHS = {
    "full_registration": "full/registration_manifest.json",
    "cropped_registration": "cropped/registration_manifest.json",
    "full_qc": "full/qc/registration_qc.json",
    "cropped_qc": "cropped/qc/registration_qc.json",
    "within_case_comparison": "comparison/registration_comparison.json",
    "metric_analysis": "full/metric_analysis/registration_metric_report.json",
    "intensity_change": "full/intensity_change/intensity_change_manifest.json",
    "functional_lung": "full/functional_lung/functional_lung_report.json",
    "normalized_intensity": (
        "full/normalized_intensity_epsilon_machine/normalized_intensity_report.json"
    ),
}
CORE_REPORTS = {
    "full_registration",
    "cropped_registration",
    "full_qc",
    "cropped_qc",
    "within_case_comparison",
}
CASE_KEYS = ("case_1", "case_2")
CASE_LABELS = {"case_1": "Case 1", "case_2": "Case 2"}
METRIC_NAMES = ("mse", "mae", "rmse", "ncc", "mutual_information")


def compare_registration_cases(
    *,
    case_1_dir: str | Path,
    case_2_dir: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compare two completed cases using JSON reports only."""
    case_dirs = {
        "case_1": Path(case_1_dir).expanduser().resolve(strict=False),
        "case_2": Path(case_2_dir).expanduser().resolve(strict=False),
    }
    reports, report_paths, warnings = _load_reports(case_dirs)
    compatibility = _validate_compatibility(case_dirs, reports, warnings)
    if not compatibility["core_method_compatible"]:
        failures = [
            name for name, value in compatibility["checks"].items()
            if value.get("required") and not value.get("compatible")
        ]
        raise ValueError("Incompatible case methods: " + ", ".join(failures))

    registration = {
        case: _registration_summary(reports[case]) for case in CASE_KEYS
    }
    lung_metrics = _optional_pair(
        reports,
        "metric_analysis",
        lambda report: {
            "voxel_count": report["metrics"]["lung_only"]["voxel_count"],
            "metrics": {
                name: report["metrics"]["lung_only"]["metrics"][name]
                for name in METRIC_NAMES
            },
        },
    )
    intensity = _optional_pair(reports, "intensity_change", _intensity_summary)
    functional = _optional_pair(reports, "functional_lung", _functional_summary)
    normalized = _normalized_summary(reports, compatibility)
    conclusions = _conclusions(registration, lung_metrics, intensity, normalized)

    destination = Path(output_dir).expanduser().resolve(strict=False)
    paths = {name: destination / filename for name, filename in OUTPUT_NAMES.items()}
    planned = [paths[name] for name in ("report", "runtime", "quality", "intensity", "functional")]
    if normalized["available"]:
        planned.append(paths["normalized"])
    if not dry_run and any(path.exists() for path in planned) and not overwrite:
        raise FileExistsError("Case-comparison outputs already exist; use --overwrite.")

    result = {
        "pipeline_version": PIPELINE_VERSION,
        "analysis_type": ANALYSIS_TYPE,
        "comparison_scope": "registration_case_1 versus registration_case_2",
        "inputs": {
            case: {
                name: display_path(path) if path is not None else None
                for name, path in report_paths[case].items()
            }
            for case in CASE_KEYS
        },
        "compatibility_validation": compatibility,
        "registration": registration,
        "lung_only_metric_comparison": lung_metrics,
        "intensity_change_comparison": intensity,
        "functional_lung_comparison": functional,
        "epsilon_normalized_comparison": normalized,
        "interpretation": conclusions,
        "overall_case_winner": None,
        "limitations": [
            "No biological, clinical, disease-severity, treatment-response, or ventilation conclusion is supported.",
            "Full and cropped metrics use different spatial domains; their absolute MSE and MAE values are not direct better/worse comparisons.",
            "Voxel locations are not compared across cases because cross-case anatomical correspondence is not established.",
            "Registration and trilinear interpolation contribute to residual intensity differences.",
            "Respiratory state and MRI intensity calibration remain unresolved.",
        ],
        "warnings": warnings,
        "outputs": {
            name: display_path(path) if name != "normalized" or normalized["available"] else None
            for name, path in paths.items()
        },
        "completion_status": "dry-run" if dry_run else "pending",
        "writes_performed": False if dry_run else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Research-only, descriptive, and non-diagnostic.",
    }
    if dry_run:
        return result

    destination.mkdir(parents=True, exist_ok=True)
    _plot_runtime(registration, paths["runtime"])
    _plot_quality(registration, paths["quality"])
    _plot_intensity(intensity, paths["intensity"])
    _plot_functional(functional, paths["functional"])
    if normalized["available"]:
        _plot_normalized(normalized, paths["normalized"])
    result["completion_status"] = "complete"
    result["writes_performed"] = True
    paths["report"].write_text(json.dumps(result, indent=2) + "\n")
    return result


def _load_reports(case_dirs):
    reports: dict[str, dict[str, Any | None]] = {}
    paths: dict[str, dict[str, Path | None]] = {}
    warnings: list[str] = []
    for case, directory in case_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{CASE_LABELS[case]} directory does not exist.")
        reports[case] = {}
        paths[case] = {}
        for name, relative in REPORT_PATHS.items():
            path = directory / relative
            if not path.is_file():
                if name in CORE_REPORTS:
                    raise FileNotFoundError(f"{CASE_LABELS[case]} required report is missing: {relative}")
                reports[case][name] = None
                paths[case][name] = None
                warnings.append(f"{CASE_LABELS[case]} optional {name} report is missing.")
                continue
            try:
                report = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"{CASE_LABELS[case]} {name} report is unreadable or malformed.") from exc
            if report.get("completion_status") != "complete":
                if name in CORE_REPORTS:
                    raise ValueError(f"{CASE_LABELS[case]} {name} report is not complete.")
                reports[case][name] = None
                paths[case][name] = None
                warnings.append(f"{CASE_LABELS[case]} optional {name} report is not complete.")
                continue
            reports[case][name] = report
            paths[case][name] = path
        _validate_core_identity(case, reports[case])
    return reports, paths, warnings


def _validate_core_identity(case, reports):
    expected = reports["full_registration"].get("case_id")
    if not expected:
        raise ValueError(f"{CASE_LABELS[case]} registration manifest lacks case_id.")
    expected_experiments = {
        "full_registration": "full", "cropped_registration": "cropped",
        "full_qc": "full", "cropped_qc": "cropped",
    }
    for name, experiment in expected_experiments.items():
        report = reports[name]
        if report.get("case_id") != expected or report.get("experiment") != experiment:
            raise ValueError(f"{CASE_LABELS[case]} {name} identity does not match the case request.")
    if reports["within_case_comparison"].get("case_id") != expected:
        raise ValueError(f"{CASE_LABELS[case]} within-case comparison identity does not match.")


def _check(name, left, right, *, required=True, details=None):
    return name, {
        "compatible": left == right,
        "required": required,
        "case_1": left,
        "case_2": right,
        **({"details": details} if details else {}),
    }


def _validate_compatibility(case_dirs, reports, warnings):
    checks = {}
    supporting_reports = {}
    reg1 = reports["case_1"]["full_registration"]
    reg2 = reports["case_2"]["full_registration"]
    methods = {}
    for case in CASE_KEYS:
        methods[case] = {
            experiment: {
                "pinned_upstream_commit": reports[case][f"{experiment}_registration"]["deedsbcv"]["pinned_upstream_commit"],
                "binary_sha256": {k: v["sha256"] for k, v in reports[case][f"{experiment}_registration"]["deedsbcv"]["binaries"].items()},
                "default_parameters_by_omission": reports[case][f"{experiment}_registration"]["commands"]["default_parameters_by_omission"],
            }
            for experiment in ("full", "cropped")
        }
    method_compatible = methods["case_1"] == methods["case_2"] and all(
        methods[case][experiment]["default_parameters_by_omission"] is True
        for case in CASE_KEYS for experiment in ("full", "cropped")
    )
    checks["deedsbcv_method_and_build"] = {
        "compatible": method_compatible, "required": True,
        "case_1": methods["case_1"], "case_2": methods["case_2"],
    }

    approach = {}
    for case in CASE_KEYS:
        approach[case] = {
            experiment: {
                "affine_flags": _flags(reports[case][f"{experiment}_registration"]["commands"]["affine"]),
                "deformable_flags": _flags(reports[case][f"{experiment}_registration"]["commands"]["deformable"]),
                "defaults_omitted": reports[case][f"{experiment}_registration"]["commands"]["default_parameters_by_omission"],
            }
            for experiment in ("full", "cropped")
        }
    expected_approach = {
        experiment: {"affine_flags": ["-F", "-M", "-O"], "deformable_flags": ["-F", "-M", "-O", "-A"], "defaults_omitted": True}
        for experiment in ("full", "cropped")
    }
    checks["full_and_cropped_registration_approach"] = {
        "compatible": approach["case_1"] == approach["case_2"] == expected_approach,
        "required": True, "case_1": approach["case_1"], "case_2": approach["case_2"],
    }

    crop_values, padding_values = {}, {}
    for case, directory in case_dirs.items():
        crop_path = directory / "cropped/preprocessed/shared_crop_manifest.json"
        crop = _optional_json(crop_path)
        crop_values[case] = None if crop is None else {
            "completion_status": crop.get("completion_status"),
            "source_experiment": crop.get("source_experiment"),
            "output_experiment": crop.get("output_experiment"),
            "same_bbox_applied": crop.get("geometry_verification", {}).get("same_bbox_applied"),
            "no_resampling": crop.get("geometry_verification", {}).get("no_resampling"),
        }
        bbox_path = directory.parent.parent / "bbox" / f"{reg1['case_id'] if case == 'case_1' else reg2['case_id']}_fixed" / "bbox.json"
        bbox = _optional_json(bbox_path)
        padding_values[case] = None if bbox is None else bbox.get("padding", {}).get("requested_mm_zyx")
        supporting_reports[case] = {
            "shared_crop_manifest": display_path(crop_path) if crop is not None else None,
            "bbox_report": display_path(bbox_path) if bbox is not None else None,
        }
    crop_required = all(value is not None for value in crop_values.values())
    padding_required = all(value is not None for value in padding_values.values())
    if not crop_required:
        warnings.append("Fixed-derived crop compatibility could not be fully verified from support manifests.")
    if not padding_required:
        warnings.append("BBox padding compatibility could not be fully verified from support manifests.")
    valid_crop = {"completion_status": "complete", "source_experiment": "full", "output_experiment": "cropped", "same_bbox_applied": True, "no_resampling": True}
    checks["fixed_derived_crop_strategy"] = {
        "compatible": crop_values["case_1"] == crop_values["case_2"] == valid_crop,
        "required": crop_required, "case_1": crop_values["case_1"], "case_2": crop_values["case_2"],
    }
    name, value = _check("bbox_padding_mm_zyx", padding_values["case_1"], padding_values["case_2"], required=padding_required)
    checks[name] = value

    qc_bins = {
        case: {
            "full_qc": reports[case]["full_qc"].get("histogram_bins"),
            "cropped_qc": reports[case]["cropped_qc"].get("histogram_bins"),
        }
        for case in CASE_KEYS
    }
    name, value = _check("registration_qc_mutual_information_bins", qc_bins["case_1"], qc_bins["case_2"])
    checks[name] = value
    metric_bins = {
        case: _get(reports[case]["metric_analysis"], "histogram_configuration", "mutual_information_bins")
        for case in CASE_KEYS
    }
    required_metric_bins = all(value is not None for value in metric_bins.values())
    name, value = _check("metric_analysis_mutual_information_bins", metric_bins["case_1"], metric_bins["case_2"], required=required_metric_bins)
    checks[name] = value
    definitions = {
        case: _get(reports[case]["metric_analysis"], "metric_definitions") for case in CASE_KEYS
    }
    required = all(value is not None for value in definitions.values())
    name, value = _check("metric_definitions", definitions["case_1"], definitions["case_2"], required=required)
    checks[name] = value

    displays = {case: _display_percentiles(reports[case]) for case in CASE_KEYS}
    display_required = all(all(value is not None for value in displays[case].values()) for case in CASE_KEYS)
    name, value = _check("display_percentiles", displays["case_1"], displays["case_2"], required=display_required)
    checks[name] = value
    masks = {case: _lung_mask_method(reports[case]) for case in CASE_KEYS}
    mask_required = all(all(value for value in masks[case].values()) for case in CASE_KEYS)
    name, value = _check("fixed_space_lung_masking", masks["case_1"], masks["case_2"], required=mask_required)
    checks[name] = value

    norm = {case: _normalized_method(reports[case]["normalized_intensity"]) for case in CASE_KEYS}
    normalized_available = all(value is not None for value in norm.values())
    formula_equal = normalized_available and norm["case_1"]["formula"] == norm["case_2"]["formula"] == "(fixed - registered) / (fixed + epsilon)"
    policy_equal = normalized_available and norm["case_1"]["policy"] == norm["case_2"]["policy"] == "epsilon"
    source_equal = normalized_available and norm["case_1"]["epsilon_source"] == norm["case_2"]["epsilon_source"] == "machine"
    epsilon_equal = normalized_available and norm["case_1"]["epsilon"] == norm["case_2"]["epsilon"] == float(np.finfo(np.float64).eps)
    checks["normalized_formula"] = {"compatible": bool(formula_equal), "required": False, "case_1": None if norm["case_1"] is None else norm["case_1"]["formula"], "case_2": None if norm["case_2"] is None else norm["case_2"]["formula"]}
    checks["epsilon_policy_and_source"] = {"compatible": bool(policy_equal and source_equal), "required": False, "case_1": None if norm["case_1"] is None else {"policy": norm["case_1"]["policy"], "source": norm["case_1"]["epsilon_source"]}, "case_2": None if norm["case_2"] is None else {"policy": norm["case_2"]["policy"], "source": norm["case_2"]["epsilon_source"]}}
    checks["exact_epsilon_value"] = {"compatible": bool(epsilon_equal), "required": False, "case_1": None if norm["case_1"] is None else norm["case_1"]["epsilon"], "case_2": None if norm["case_2"] is None else norm["case_2"]["epsilon"]}
    normalized_compatible = bool(formula_equal and policy_equal and source_equal and epsilon_equal)
    if normalized_available and not normalized_compatible:
        warnings.append("Normalized reports are incompatible; normalized comparison and figure were omitted.")
    required_checks = [value for value in checks.values() if value["required"]]
    return {
        "core_method_compatible": all(value["compatible"] for value in required_checks),
        "normalized_comparison_compatible": normalized_compatible,
        "supporting_reports": supporting_reports,
        "checks": checks,
    }


def _optional_json(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _flags(command):
    return [value for value in command if isinstance(value, str) and value.startswith("-")]


def _get(value, *keys):
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _display_percentiles(reports):
    return {
        "metric_analysis": _get(reports["metric_analysis"], "display_scaling", "percentile"),
        "intensity_change": _get(reports["intensity_change"], "display_scaling", "signed", "clipping_percentile"),
        "functional_lung": _get(reports["functional_lung"], "display_scaling", "lung_only_absolute_percentile"),
        "normalized_intensity": _get(reports["normalized_intensity"], "display_scaling", "valid_lung_absolute_percentile"),
    }


def _lung_mask_method(reports):
    return {
        "metric_lung_domain": _get(reports["metric_analysis"], "metrics", "lung_only", "voxel_count") is not None,
        "metric_no_resampling": _get(reports["metric_analysis"], "geometry_validation", "resampling_performed") is False,
        "intensity_exact_geometry": _get(reports["intensity_change"], "geometry_validation", "exact_agreement_required") is True,
        "functional_exact_geometry": _get(reports["functional_lung"], "geometry_validation", "exact_agreement_required") is True,
        "normalized_exact_geometry": _get(reports["normalized_intensity"], "geometry_validation", "exact_agreement_required") is True,
        "semantic_left_right": _get(reports["normalized_intensity"], "left_right_partition_source") == "supplied semantic masks",
    }


def _normalized_method(report):
    if report is None:
        return None
    return {
        "formula": _get(report, "formula", "expression"),
        "policy": _get(report, "denominator_policy", "policy"),
        "epsilon_source": _get(report, "denominator_policy", "epsilon_source"),
        "epsilon": _get(report, "denominator_policy", "epsilon"),
    }


def _registration_summary(reports):
    experiments = {}
    for experiment in ("full", "cropped"):
        qc = reports[f"{experiment}_qc"]
        manifest = reports[f"{experiment}_registration"]
        metrics = {}
        before, after = qc["metrics"]["before"], qc["metrics"]["after"]
        mapping = {
            "mse": "mean_squared_error", "mae": "mean_absolute_error",
            "ncc": "normalized_cross_correlation", "mutual_information": "mutual_information",
        }
        for short, source in mapping.items():
            direction = "lower-is-better" if short in {"mse", "mae"} else "higher-is-better"
            metrics[short] = _metric_change(before[source], after[source], direction)
        metrics["rmse"] = _metric_change(math.sqrt(before["mean_squared_error"]), math.sqrt(after["mean_squared_error"]), "lower-is-better")
        ssd = qc.get("deedsbcv_ssd")
        if ssd:
            metrics["deedsbcv_ssd"] = _metric_change(ssd["before"], ssd["after"], "lower-is-better")
            metrics["deedsbcv_ssd"]["source"] = ssd.get("source")
        affine = float(manifest["stages"]["affine"]["runtime_seconds"])
        deformable = float(manifest["stages"]["deformable"]["runtime_seconds"])
        experiments[experiment] = {
            "voxel_count": int(qc["voxel_count"]),
            "runtime_seconds": {
                "affine": affine,
                "deformable": deformable,
                "total": affine + deformable,
            },
            "metrics": metrics,
        }
    full, cropped = experiments["full"], experiments["cropped"]
    return {
        **experiments,
        "derived": {
            "voxel_reduction_percent": _reduction(full["voxel_count"], cropped["voxel_count"]),
            "runtime_reduction_percent": _reduction(full["runtime_seconds"]["total"], cropped["runtime_seconds"]["total"]),
            "combined_registration_runtime_seconds": full["runtime_seconds"]["total"] + cropped["runtime_seconds"]["total"],
        },
    }


def _metric_change(before, after, direction):
    before, after = float(before), float(after)
    absolute = before - after if direction == "lower-is-better" else after - before
    percent = None if before == 0 else 100.0 * absolute / abs(before)
    return {"before": before, "after": after, "absolute_improvement": absolute, "percent_improvement": percent, "direction": direction}


def _reduction(original, reduced):
    return None if original == 0 else 100.0 * (1.0 - float(reduced) / float(original))


def _optional_pair(reports, name, transform):
    if any(reports[case][name] is None for case in CASE_KEYS):
        return {"available": False, "reason": f"Both completed {name} reports are required."}
    return {"available": True, "cases": {case: transform(reports[case][name]) for case in CASE_KEYS}}


def _intensity_summary(report):
    signed = report["statistics"]["signed"]
    absolute = report["statistics"]["absolute"]
    whole_signed, whole_absolute = signed["combined_lung"], absolute["combined_lung"]
    return {
        "signed_whole_lung": _select(whole_signed, ("mean", "median", "standard_deviation", "positive_fraction", "negative_fraction"), percentiles=(5, 95)),
        "absolute_whole_lung": _select(whole_absolute, ("mean", "median"), percentiles=(95, 99)),
        "left_right_signed": {
            "left_mean": signed["left_lung"]["mean"],
            "right_mean": signed["right_lung"]["mean"],
            "left_minus_right_mean": signed["left_lung"]["mean"] - signed["right_lung"]["mean"],
            "pooled_standardized_effect_size": _pooled_effect(signed["left_lung"], signed["right_lung"]),
        },
        "display_scaling": report["display_scaling"],
    }


def _select(values, keys, *, percentiles=()):
    result = {key: values.get(key) for key in keys}
    if percentiles:
        result["percentiles"] = {str(p): values["percentiles"].get(str(p)) for p in percentiles}
    return result


def _pooled_effect(left, right):
    n1, n2 = left["valid_voxel_count"], right["valid_voxel_count"]
    denominator = n1 + n2 - 2
    if denominator <= 0:
        return None
    pooled = math.sqrt(((n1 - 1) * left["standard_deviation"] ** 2 + (n2 - 1) * right["standard_deviation"] ** 2) / denominator)
    return None if pooled == 0 else (left["mean"] - right["mean"]) / pooled


def _functional_summary(report):
    populated = [row for row in report["slice_statistics"] if row.get("lung_voxel_count", 0) > 0]
    def value_range(key):
        values = [float(row[key]) for row in populated if row.get(key) is not None]
        return {"minimum": min(values), "maximum": max(values)} if values else None
    hotspots = report["hotspot_statistics"]
    return {
        "lung_volumes": report["lung_volumes"],
        "slice_wise_signed_mean_range": value_range("signed_mean"),
        "slice_wise_signed_standard_deviation_range": value_range("signed_standard_deviation"),
        "largest_positive_connected_region": _compact_hotspot(hotspots["largest_positive_change_region"]),
        "largest_negative_connected_region": _compact_hotspot(hotspots["largest_negative_change_region"]),
        "connected_region_definition": {"connectivity": hotspots.get("connectivity"), "threshold_rule": hotspots.get("threshold_rule")},
        "left_right_asymmetry": report["asymmetry"],
        "display_scaling": report["display_scaling"],
    }


def _compact_hotspot(region):
    keys = ("available", "voxel_count", "volume_mm3", "signed_change_mean", "signed_change_minimum", "signed_change_maximum")
    return {key: region.get(key) for key in keys}


def _normalized_summary(reports, compatibility):
    if not compatibility["normalized_comparison_compatible"]:
        return {"available": False, "reason": "Both reports must use the identical machine-epsilon formula and value."}
    cases = {}
    for case in CASE_KEYS:
        report = reports[case]["normalized_intensity"]
        cases[case] = {
            "formula": report["formula"]["expression"],
            "epsilon_source": report["denominator_policy"]["epsilon_source"],
            "epsilon": report["denominator_policy"]["epsilon"],
            "whole_lung": report["statistics"]["whole_lung"],
            "left_lung": report["statistics"]["left_lung"],
            "right_lung": report["statistics"]["right_lung"],
            "fixed_lung_intensity": report["fixed_lung_intensity"],
            "display_scaling": report["display_scaling"],
            "denominator_warnings": [
                warning for warning in report.get("warnings", [])
                if any(term in warning.lower() for term in ("epsilon", "denominator", "zero", "asymmetric", "small"))
            ],
        }
    return {"available": True, "cases": cases, "validated_epsilon": cases["case_1"]["epsilon"], "research_only": "Descriptive normalized values are not validated ventilation."}


def _conclusions(registration, lung_metrics, intensity, normalized):
    runtimes = {case: registration[case]["derived"]["combined_registration_runtime_seconds"] for case in CASE_KEYS}
    slower = max(runtimes, key=runtimes.get)
    reductions = {case: registration[case]["derived"]["runtime_reduction_percent"] for case in CASE_KEYS}
    greater_reduction = max(reductions, key=reductions.get)
    both_improved = all(
        registration[case][experiment]["metrics"][metric]["absolute_improvement"] > 0
        for case in CASE_KEYS for experiment in ("full", "cropped")
        for metric in ("mse", "mae", "ncc", "mutual_information")
    )
    result = {
        "registration_computation": f"{CASE_LABELS[slower]} required more combined full-plus-cropped registration runtime ({runtimes[slower]:.3f} s versus {runtimes['case_2' if slower == 'case_1' else 'case_1']:.3f} s).",
        "cropping_runtime": f"Cropping reduced runtime similarly in both cases ({CASE_LABELS['case_1']}: {reductions['case_1']:.3f}%; {CASE_LABELS['case_2']}: {reductions['case_2']:.3f}%); {CASE_LABELS[greater_reduction]} had the larger reduction by {abs(reductions['case_1'] - reductions['case_2']):.3f} percentage points.",
        "registration_improvement": "Both cases improved all reported full and cropped MSE, MAE, NCC, and MI metrics." if both_improved else "Not every reported registration metric improved in both cases.",
        "final_quality_comparability": "Final metrics are method-compatible for descriptive comparison, but no equivalence threshold or universal quality score is defined.",
        "cross_domain_warning": "Absolute full and cropped error values use different spatial domains and are not direct better/worse comparisons.",
    }
    if lung_metrics["available"]:
        result["lung_only_quality"] = "Lung-only metrics use the same fixed-space masking methodology and are preferred for descriptive lung-domain comparison."
    if intensity["available"]:
        means = {case: intensity["cases"][case]["absolute_whole_lung"]["mean"] for case in CASE_KEYS}
        larger = max(means, key=means.get)
        result["residual_intensity_change"] = f"{CASE_LABELS[larger]} retained the larger mean absolute fixed-space lung intensity difference."
        result["left_right_patterns"] = "Left/right descriptive means and standardized differences differ between cases; no biological interpretation is assigned."
    if normalized["available"]:
        means = {case: normalized["cases"][case]["whole_lung"]["mean"] for case in CASE_KEYS}
        result["normalized_distributions"] = f"Machine-epsilon normalized descriptive means differ ({CASE_LABELS['case_1']}: {means['case_1']:.6g}; {CASE_LABELS['case_2']}: {means['case_2']:.6g}); these are not ventilation measurements."
    return result


def _base_figure(nrows=1, ncols=1, figsize=(10, 6)):
    return plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)


def _save(fig, path):
    fig.savefig(path, dpi=160, metadata={"Software": "lung-ct-bounding-box-pipeline"})
    plt.close(fig)


def _grouped_bars(ax, categories, series, ylabel, title):
    x = np.arange(len(categories)); width = 0.36
    for index, (label, values) in enumerate(series.items()):
        ax.bar(x + (index - 0.5) * width, values, width, label=label)
    ax.set_xticks(x, categories); ax.set_ylabel(ylabel); ax.set_title(title); ax.axhline(0, color="black", linewidth=0.7); ax.legend()


def _plot_runtime(registration, path):
    fig, axes = _base_figure(1, 2, (12, 5))
    _grouped_bars(axes[0], ["Full", "Cropped"], {CASE_LABELS[c]: [registration[c][e]["runtime_seconds"]["total"] for e in ("full", "cropped")] for c in CASE_KEYS}, "Seconds", "Registration runtime")
    _grouped_bars(axes[1], ["Voxel reduction", "Runtime reduction"], {CASE_LABELS[c]: [registration[c]["derived"]["voxel_reduction_percent"], registration[c]["derived"]["runtime_reduction_percent"]] for c in CASE_KEYS}, "Reduction (%)", "Effect of fixed-derived cropping")
    fig.suptitle("Case 1 vs Case 2 registration computation (zero-based axes)")
    _save(fig, path)


def _plot_quality(registration, path):
    fig, axes = _base_figure(2, 2, (12, 9))
    configs = (("mse", "MSE (lower is better)"), ("mae", "MAE (lower is better)"), ("ncc", "NCC (higher is better)"), ("mutual_information", "MI (higher is better)"))
    for ax, (metric, title) in zip(axes.flat, configs):
        _grouped_bars(ax, ["Full", "Cropped"], {CASE_LABELS[c]: [registration[c][e]["metrics"][metric]["after"] for e in ("full", "cropped")] for c in CASE_KEYS}, "Post-registration value", title)
    fig.suptitle("Final registration metrics; full and cropped domains differ")
    _save(fig, path)


def _unavailable_figure(path, title, reason):
    fig, ax = _base_figure(figsize=(10, 5)); ax.axis("off"); ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True, fontsize=13); ax.set_title(title); _save(fig, path)


def _plot_intensity(section, path):
    if not section["available"]:
        return _unavailable_figure(path, "Intensity-change comparison", section["reason"])
    cases = section["cases"]; fig, axes = _base_figure(2, 2, (12, 9))
    _grouped_bars(axes[0, 0], ["Mean", "Median", "SD"], {CASE_LABELS[c]: [cases[c]["signed_whole_lung"][k] for k in ("mean", "median", "standard_deviation")] for c in CASE_KEYS}, "Intensity units", "Signed lung differences")
    _grouped_bars(axes[0, 1], ["Mean", "Median", "P95", "P99"], {CASE_LABELS[c]: [cases[c]["absolute_whole_lung"][k] if k in cases[c]["absolute_whole_lung"] else cases[c]["absolute_whole_lung"]["percentiles"][k[1:]] for k in ("mean", "median", "P95", "P99")] for c in CASE_KEYS}, "Intensity units", "Absolute lung differences")
    _grouped_bars(axes[1, 0], ["Positive", "Negative"], {CASE_LABELS[c]: [100 * cases[c]["signed_whole_lung"][f"{k}_fraction"] for k in ("positive", "negative")] for c in CASE_KEYS}, "Fraction (%)", "Signed fractions")
    _grouped_bars(axes[1, 1], ["Left mean", "Right mean", "Left-right"], {CASE_LABELS[c]: [cases[c]["left_right_signed"][k] for k in ("left_mean", "right_mean", "left_minus_right_mean")] for c in CASE_KEYS}, "Intensity units", "Left/right signed pattern")
    fig.suptitle("Residual fixed-space lung intensity change; no voxelwise cross-case correspondence")
    _save(fig, path)


def _plot_functional(section, path):
    if not section["available"]:
        return _unavailable_figure(path, "Functional-lung comparison", section["reason"])
    cases = section["cases"]; fig, axes = _base_figure(2, 2, (12, 9))
    _grouped_bars(axes[0, 0], ["Whole", "Left", "Right"], {CASE_LABELS[c]: [cases[c]["lung_volumes"][r]["volume_ml"] for r in ("whole_lung", "left_lung", "right_lung")] for c in CASE_KEYS}, "Volume (mL)", "Fixed-space lung-mask volumes")
    for c in CASE_KEYS:
        values = cases[c]["slice_wise_signed_mean_range"]
        axes[0, 1].plot([values["minimum"], values["maximum"]], [CASE_LABELS[c]] * 2, marker="o", linewidth=3, label=CASE_LABELS[c])
    axes[0, 1].axvline(0, color="black", linewidth=0.7); axes[0, 1].set_xlabel("Slice signed mean range"); axes[0, 1].set_title("Populated axial slices")
    _grouped_bars(axes[1, 0], ["Positive", "Negative"], {CASE_LABELS[c]: [cases[c]["largest_positive_connected_region"]["volume_mm3"] / 1000, cases[c]["largest_negative_connected_region"]["volume_mm3"] / 1000] for c in CASE_KEYS}, "Region volume (mL)", "Largest sign-connected regions")
    _grouped_bars(axes[1, 1], ["Left mean", "Right mean", "Left-right"], {CASE_LABELS[c]: [cases[c]["left_right_asymmetry"][k] for k in ("left_mean_signed_change", "right_mean_signed_change", "left_minus_right_mean_difference")] for c in CASE_KEYS}, "Intensity units", "Descriptive asymmetry")
    fig.suptitle("Exploratory functional-lung summaries; no abnormality inference")
    _save(fig, path)


def _plot_normalized(section, path):
    cases = section["cases"]; fig, axes = _base_figure(2, 2, (12, 9))
    _grouped_bars(axes[0, 0], ["Mean", "Median", "SD"], {CASE_LABELS[c]: [cases[c]["whole_lung"][k] for k in ("mean", "median", "standard_deviation")] for c in CASE_KEYS}, "Normalized value", "Whole-lung distribution")
    percentiles = ("5", "25", "50", "75", "95")
    for c in CASE_KEYS:
        axes[0, 1].plot([int(p) for p in percentiles], [cases[c]["whole_lung"]["percentiles"][p] for p in percentiles], marker="o", label=CASE_LABELS[c])
    axes[0, 1].axhline(0, color="black", linewidth=0.7); axes[0, 1].set_xlabel("Percentile"); axes[0, 1].set_ylabel("Normalized value"); axes[0, 1].set_title("Whole-lung percentiles"); axes[0, 1].legend()
    _grouped_bars(axes[1, 0], ["Positive", "Negative"], {CASE_LABELS[c]: [100 * cases[c]["whole_lung"][f"{k}_fraction"] for k in ("positive", "negative")] for c in CASE_KEYS}, "Fraction (%)", "Normalized sign fractions")
    _grouped_bars(axes[1, 1], ["Left mean", "Right mean"], {CASE_LABELS[c]: [cases[c][r]["mean"] for r in ("left_lung", "right_lung")] for c in CASE_KEYS}, "Normalized value", "Left/right normalized means")
    fig.suptitle("Machine-epsilon normalized descriptive comparison; not validated ventilation")
    _save(fig, path)
