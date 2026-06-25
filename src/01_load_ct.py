from pathlib import Path
import SimpleITK as sitk


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
    if len(series_list) == 0:
        raise ValueError("No DICOM series found. Check your data folder path.")

    # CT series usually has many files/slices.
    # RTSTRUCT usually has 1 file.
    largest = max(series_list, key=lambda x: x["num_files"])

    print("Selected DICOM series:")
    print("Folder:", largest["folder"])
    print("Series ID:", largest["series_id"])
    print("Number of files:", largest["num_files"])

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(largest["file_names"])

    image = reader.Execute()
    ct_array = sitk.GetArrayFromImage(image)

    return image, ct_array


if __name__ == "__main__":
    print("Searching in:", ROOT_DIR)

    series_list = find_all_dicom_series(ROOT_DIR)

    print("\nAll DICOM series found:")
    for s in series_list:
        print(f"{s['num_files']} files | {s['folder']}")

    image, ct_array = load_largest_series(series_list)

    print("\nLoaded CT successfully.")
    print("CT shape:", ct_array.shape)
    print("Shape meaning: z, y, x")
    print("Spacing x/y/z:", image.GetSpacing())
    print("Intensity min:", ct_array.min())
    print("Intensity max:", ct_array.max())