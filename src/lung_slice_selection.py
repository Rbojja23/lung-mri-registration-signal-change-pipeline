"""Lung-mask-aware slice selection for visualization products."""

from __future__ import annotations

from typing import Any

import numpy as np


def _occupancy(mask: np.ndarray, orientation: str) -> np.ndarray:
    axes = {
        "axial": (1, 2),
        "coronal": (0, 2),
        "sagittal": (0, 1),
    }
    return np.count_nonzero(mask, axis=axes[orientation]).astype(np.int64)


def _quantile_panel_slices(
    occupancy: np.ndarray,
    *,
    requested_count: int,
    minimum_fraction_of_peak: float,
) -> tuple[list[int], int, int, int]:
    occupied = np.flatnonzero(occupancy > 0)
    if occupied.size == 0:
        raise ValueError("Cannot select visualization slices from an empty lung mask")

    first = int(occupied[0])
    last = int(occupied[-1])
    minimum_count = max(1, int(np.ceil(float(occupancy.max()) * minimum_fraction_of_peak)))
    eligible = occupied[occupancy[occupied] >= minimum_count]
    if eligible.size == 0:
        eligible = occupied

    count = min(requested_count, int(eligible.size))
    positions = np.linspace(0, int(eligible.size) - 1, count)
    selected = sorted({int(eligible[int(round(position))]) for position in positions})

    target_minimum = min(5, int(eligible.size), requested_count)
    if len(selected) < target_minimum:
        remaining = sorted(
            (int(index) for index in eligible if int(index) not in selected),
            key=lambda index: (-int(occupancy[index]), index),
        )
        selected.extend(remaining[: target_minimum - len(selected)])
        selected.sort()

    return selected, first, last, minimum_count


def select_lung_visualization_slices(
    mask: np.ndarray,
    *,
    axial_panel_count: int,
    minimum_fraction_of_peak: float = 0.10,
) -> dict[str, Any]:
    """Select high-occupancy single views and lung-only axial panel slices.

    Single-plane views use the maximum-occupancy slice in each orientation.
    Axial panel slices are quantiles of slices whose occupancy is at least the
    configured fraction of peak axial occupancy. All returned slices contain
    lung voxels and lie inside the mask's occupied range.
    """

    lung_mask = np.asarray(mask, dtype=bool)
    if lung_mask.ndim != 3:
        raise ValueError(f"Expected a 3D lung mask, got shape {lung_mask.shape}")
    if not np.any(lung_mask):
        raise ValueError("Cannot select visualization slices from an empty lung mask")
    if axial_panel_count < 1:
        raise ValueError("axial_panel_count must be at least 1")
    if not 0.0 <= minimum_fraction_of_peak <= 1.0:
        raise ValueError("minimum_fraction_of_peak must be between 0 and 1")

    occupancies = {
        orientation: _occupancy(lung_mask, orientation)
        for orientation in ("axial", "coronal", "sagittal")
    }
    representative_by_orientation = {
        orientation: int(np.argmax(values))
        for orientation, values in occupancies.items()
    }
    representative = {
        "axial_z": representative_by_orientation["axial"],
        "coronal_y": representative_by_orientation["coronal"],
        "sagittal_x": representative_by_orientation["sagittal"],
    }
    axial_panel, first_z, last_z, minimum_count = _quantile_panel_slices(
        occupancies["axial"],
        requested_count=axial_panel_count,
        minimum_fraction_of_peak=minimum_fraction_of_peak,
    )

    coordinates = np.argwhere(lung_mask)
    lower = coordinates.min(axis=0).astype(int)
    upper = coordinates.max(axis=0).astype(int)
    return {
        "central": representative,
        "axial_panel_z": axial_panel,
        "axial_contact_sheet_z": axial_panel,
        "lung_extent_zyx": {
            "minimum": [int(value) for value in lower],
            "maximum": [int(value) for value in upper],
        },
        "selection": {
            "single_slice_method": "maximum lung-mask voxel occupancy per orientation",
            "axial_panel_method": "evenly spaced quantiles of axial slices passing the minimum occupancy check",
            "axial_first_lung_slice": first_z,
            "axial_last_lung_slice": last_z,
            "axial_peak_occupancy_voxels": int(occupancies["axial"].max()),
            "axial_minimum_occupancy_voxels": minimum_count,
            "axial_minimum_fraction_of_peak": minimum_fraction_of_peak,
            "representative_occupancy_voxels": {
                orientation: int(occupancies[orientation][representative_by_orientation[orientation]])
                for orientation in occupancies
            },
            "axial_panel_occupancy_voxels": [
                int(occupancies["axial"][index]) for index in axial_panel
            ],
        },
    }
