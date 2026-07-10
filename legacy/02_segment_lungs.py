"""Deprecated CT-only lungmask segmentation script retained for history."""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from lungmask import mask


ROOT_DIR = Path("data/raw/LCTSC_offsite")


def find_all_dicom_series(root_dir):
    series_list = []

    for folder in root_dir.rglob("*"):
        if not folder.is_dir():
            continue

        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
        except RuntimeError:
            continue

        if not series_ids:
            continue

        for series_id in series_ids:
            file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                str(folder),
                series_id
            )

            series_list.append({
                "folder": folder,
                "series_id": series_id,
                "num_files": len(file_names),
                "file_names": file_names,
            })

    return series_list


def load_largest_series(series_list):
    largest = max(series_list, key=lambda x: x["num_files"])

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(largest["file_names"])

    image = reader.Execute()
    ct_array = sitk.GetArrayFromImage(image)

    return image, ct_array


if __name__ == "__main__":
    Path("outputs/masks").mkdir(parents=True, exist_ok=True)

    series_list = find_all_dicom_series(ROOT_DIR)
    image, ct_array = load_largest_series(series_list)

    print("CT shape:", ct_array.shape)

    print("Running lungmask...")
    lung_mask = mask.apply(image)

    print("Lung mask shape:", lung_mask.shape)
    print("Unique labels:", np.unique(lung_mask))

    np.save("outputs/masks/first_case_lung_mask.npy", lung_mask)

    print("Saved lung mask.")
