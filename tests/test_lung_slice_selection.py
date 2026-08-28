from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from lung_slice_selection import select_lung_visualization_slices  # noqa: E402


def test_representative_and_panel_slices_are_lung_occupied_and_in_range():
    mask = np.zeros((20, 18, 16), dtype=bool)
    for z in range(3, 17):
        radius = min(z - 2, 17 - z, 5)
        mask[z, 4 : 4 + 2 * radius, 3 : 3 + 2 * radius] = True

    selected = select_lung_visualization_slices(mask, axial_panel_count=7)
    occupancy = {
        "axial": np.count_nonzero(mask, axis=(1, 2)),
        "coronal": np.count_nonzero(mask, axis=(0, 2)),
        "sagittal": np.count_nonzero(mask, axis=(0, 1)),
    }
    central = selected["central"]
    assert occupancy["axial"][central["axial_z"]] == occupancy["axial"].max()
    assert occupancy["coronal"][central["coronal_y"]] == occupancy["coronal"].max()
    assert occupancy["sagittal"][central["sagittal_x"]] == occupancy["sagittal"].max()

    first = selected["selection"]["axial_first_lung_slice"]
    last = selected["selection"]["axial_last_lung_slice"]
    minimum = selected["selection"]["axial_minimum_occupancy_voxels"]
    panel = selected["axial_panel_z"]
    assert 5 <= len(panel) <= 7
    assert all(first <= z <= last for z in panel)
    assert all(occupancy["axial"][z] >= minimum > 0 for z in panel)
