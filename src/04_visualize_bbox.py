from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def show_bbox_views(ct, bbox, save_path):
    z_mid = (bbox["z_min"] + bbox["z_max"]) // 2
    y_mid = (bbox["y_min"] + bbox["y_max"]) // 2
    x_mid = (bbox["x_min"] + bbox["x_max"]) // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial view: y-x plane at one z slice
    axes[0].imshow(ct[z_mid, :, :], cmap="gray", vmin=-1000, vmax=400)
    rect = patches.Rectangle(
        (bbox["x_min"], bbox["y_min"]),
        bbox["x_max"] - bbox["x_min"],
        bbox["y_max"] - bbox["y_min"],
        fill=False,
        linewidth=2
    )
    axes[0].add_patch(rect)
    axes[0].set_title("Axial")

    # Coronal view: z-x plane at one y position
    axes[1].imshow(ct[:, y_mid, :], cmap="gray", vmin=-1000, vmax=400)
    rect = patches.Rectangle(
        (bbox["x_min"], bbox["z_min"]),
        bbox["x_max"] - bbox["x_min"],
        bbox["z_max"] - bbox["z_min"],
        fill=False,
        linewidth=2
    )
    axes[1].add_patch(rect)
    axes[1].set_title("Coronal")

    # Sagittal view: z-y plane at one x position
    axes[2].imshow(ct[:, :, x_mid], cmap="gray", vmin=-1000, vmax=400)
    rect = patches.Rectangle(
        (bbox["y_min"], bbox["z_min"]),
        bbox["y_max"] - bbox["y_min"],
        bbox["z_max"] - bbox["z_min"],
        fill=False,
        linewidth=2
    )
    axes[2].add_patch(rect)
    axes[2].set_title("Sagittal")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    series_list = find_all_dicom_series(ROOT_DIR)
    image, ct_array = load_largest_series(series_list)

    lung_mask = mask.apply(image)
    bbox = get_lung_bbox(lung_mask, padding=10)

    print("Bounding box:", bbox)

    show_bbox_views(
        ct_array,
        bbox,
        save_path="outputs/figures/first_case_bbox_views.png"
    )

    print("Saved figure to outputs/figures/first_case_bbox_views.png")