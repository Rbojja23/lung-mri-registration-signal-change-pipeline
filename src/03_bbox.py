from pathlib import Path
import json
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


def get_lung_bbox(lung_mask, padding=10):
    lung_voxels = np.where(lung_mask > 0)

    if len(lung_voxels[0]) == 0:
        raise ValueError("No lung voxels found in mask.")

    z_min, z_max = lung_voxels[0].min(), lung_voxels[0].max()
    y_min, y_max = lung_voxels[1].min(), lung_voxels[1].max()
    x_min, x_max = lung_voxels[2].min(), lung_voxels[2].max()

    z_min = max(z_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_min = max(x_min - padding, 0)

    z_max = min(z_max + padding, lung_mask.shape[0] - 1)
    y_max = min(y_max + padding, lung_mask.shape[1] - 1)
    x_max = min(x_max + padding, lung_mask.shape[2] - 1)

    return {
        "z_min": int(z_min),
        "z_max": int(z_max),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "x_min": int(x_min),
        "x_max": int(x_max),
    }


def crop_volume(volume, bbox):
    return volume[
        bbox["z_min"]:bbox["z_max"] + 1,
        bbox["y_min"]:bbox["y_max"] + 1,
        bbox["x_min"]:bbox["x_max"] + 1,
    ]


if __name__ == "__main__":
    Path("outputs/crops").mkdir(parents=True, exist_ok=True)

    series_list = find_all_dicom_series(ROOT_DIR)
    image, ct_array = load_largest_series(series_list)

    print("CT shape:", ct_array.shape)

    lung_mask = mask.apply(image)

    bbox = get_lung_bbox(lung_mask, padding=10)
    print("Bounding box:", bbox)

    cropped_ct = crop_volume(ct_array, bbox)
    cropped_mask = crop_volume(lung_mask, bbox)

    print("Cropped CT shape:", cropped_ct.shape)

    np.save("outputs/crops/first_case_ct_crop.npy", cropped_ct)
    np.save("outputs/crops/first_case_lung_mask_crop.npy", cropped_mask)

    with open("outputs/crops/first_case_bbox.json", "w") as f:
        json.dump(bbox, f, indent=4)

    print("Saved cropped CT and bbox.")