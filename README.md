# Lung CT Bounding Box Pipeline

## Project Goal

This project develops a Python pipeline to automatically identify the lungs in a public thoracic CT dataset and generate a 3D bounding box around the whole lung region.

## Dataset

The first test case uses the LCTSC dataset from The Cancer Imaging Archive. The data are downloaded locally using NBIA Data Retriever and are not included in this repository.

## Pipeline

1. Load a DICOM CT series using SimpleITK.
2. Convert the CT scan into a 3D NumPy array in z, y, x order.
3. Segment the lungs using lungmask.
4. Compute the minimum and maximum z, y, and x coordinates where the lung mask is greater than zero.
5. Add optional padding around the bounding box.
6. Crop the CT volume.
7. Visualize the bounding box on axial, coronal, and sagittal slices.

## Outputs

The pipeline generates:

- Lung mask
- Bounding box coordinates
- Cropped CT volume
- Validation figure

## Notes

Raw medical image data are excluded from GitHub using `.gitignore`.

## Local Data Privacy

All medical imaging data must remain local. The repository ignores `data/raw/`,
`outputs/`, DICOM files, NIfTI files, and generated NumPy arrays so private raw
data, segmentation masks, crops, figures, and temporary processing outputs are
not committed.

Do not commit DICOM or NIfTI files. `requirements.txt` is intentionally tracked
because it contains only the minimal Python dependencies needed by the current
tracked CT pipeline.

To check the configured local paths without inspecting patient metadata, run:

```bash
python src/check_paths.py
```

Optional overrides are supported:

```bash
python src/check_paths.py --data-root data --output-root outputs
```

The diagnostic only reports whether expected directories exist. It does not list
medical files, read DICOM tags, or inspect patient metadata.
