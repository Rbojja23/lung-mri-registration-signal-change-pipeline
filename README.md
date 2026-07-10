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

## Image Loading

The modality-neutral loader is `src/01_load_image.py`. It supports DICOM
directories and NIfTI files ending in `.nii` or `.nii.gz`. The older
`src/01_load_ct.py` script is retained temporarily as a legacy CT-only entry
point while the replacement is validated.

NIfTI inputs require an explicit modality because modality is not reliably
encoded in the NIfTI header:

```bash
python src/01_load_image.py --input path/to/image.nii.gz --input-type nifti --modality mr --dry-run
```

DICOM inputs may use `--modality auto`, which reads only the DICOM Modality tag
and does not inspect patient-identifying metadata:

```bash
python src/01_load_image.py --input path/to/dicom_directory --input-type dicom --modality auto --dry-run
```

The local MRI DICOM directory and `input_mri.nii` are both retained because they
represent the same scan in different formats: DICOM preserves the original local
source files, while NIfTI provides a direct image input for later processing.

Coordinate ordering is explicit throughout the loader:

- SimpleITK image indices, size, spacing, and origin use `x, y, z`.
- NumPy arrays from `sitk.GetArrayFromImage` use `z, y, x`.
- Physical points are reported in millimeters in the image coordinate frame.

Optional safe metadata summaries and NIfTI conversion are read-only by default
and require explicit output paths:

```bash
python src/01_load_image.py \
  --input path/to/dicom_directory \
  --input-type dicom \
  --modality auto \
  --summary-json outputs/nifti/example/summary.json \
  --convert-to-nifti outputs/nifti/example/image.nii.gz
```

The loader refuses to write outputs under `data/raw/`, does not normalize or
resample image intensities, and does not save patient names, patient IDs, birth
dates, accession numbers, or study descriptions.

## Lung Segmentation With TotalSegmentator

The new lung segmentation entry point is `src/02_segment_lungs_totalseg.py`.
It supersedes the older `src/02_segment_lungs.py`, which is now a legacy
lungmask-based script kept temporarily while the TotalSegmentator workflow is
validated. `lungmask` remains in `requirements.txt` only for that legacy script.

TotalSegmentator replaces lungmask because it supports both CT and MRI anatomy
tasks and produces named class masks. Install dependencies from
`requirements.txt`; the current workflow expects `TotalSegmentator>=2.15,<2.16`.
The first real TotalSegmentator run may download model weights. Inference runs
locally and the pipeline does not upload medical image data.

Tasks are selected from modality:

- CT uses TotalSegmentator task `total`.
- MRI uses TotalSegmentator task `total_mr`.

For MRI, prefer the DICOM-aligned working NIfTI:

```text
outputs/nifti/mri_first_case/image_from_dicom.nii.gz
```

This preserves geometry that can later be mapped back to the source DICOM series.
The researcher-provided `data/raw/MRI_cases/first_case/input_mri.nii` is retained
but is padded/reoriented relative to the original DICOM series.

Outputs are written under:

```text
outputs/totalsegmentator/<case_id>/
  raw/
  lungs/
  manifest.json
```

Raw TotalSegmentator class masks go in `raw/`. Curated lung masks and the binary
whole-lung union `combined_lungs.nii.gz` go in `lungs/`. The combined mask has
voxel values `0` and `1` and preserves the input image size, spacing, origin, and
direction. Existing case output directories are not overwritten unless
`--overwrite` is supplied.

Device selection supports `auto`, `cpu`, `mps`, and `cuda`. `auto` uses CUDA only
when available, otherwise Apple-silicon MPS when available, otherwise CPU. Example
MPS dry-run:

```bash
python src/02_segment_lungs_totalseg.py \
  --input outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --input-type nifti \
  --modality mr \
  --case-id mri_first_case \
  --device mps \
  --dry-run
```

CT dry-run example:

```bash
python src/02_segment_lungs_totalseg.py \
  --input outputs/nifti/ct_first_case/image.nii.gz \
  --input-type nifti \
  --modality ct \
  --case-id ct_first_case \
  --device auto \
  --dry-run
```

## Segmentation Validation

Validate segmentation quality before computing bounding boxes or crops. The QC
stage checks that the source image and lung mask have identical size, spacing,
origin, and direction; it fails rather than resampling mismatched data.

Use `src/02b_validate_segmentation.py` to create reproducible matplotlib QC
figures and a metrics JSON. Matplotlib remains the tool for saved QC contact
sheets in this stage; OsiriX integration comes later.

The QC renderer uses display-only scaling:

- CT: lung-focused windowing, default center `-500`, width `1500`.
- MRI: robust finite-voxel percentile scaling, default `1`st to `99`th
  percentiles.

It saves both filled-mask overlays and contour overlays, plus
`segmentation_qc.json` with advisory warnings, selected slice indices, mask
volume, connected-component metrics, centroid, boundary-touch flags, and
coordinate conventions. These warnings are not diagnoses; they are prompts for
manual review.

MRI dry-run example:

```bash
python src/02b_validate_segmentation.py \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/mri_first_case/lungs/combined_lungs.nii.gz \
  --modality mr \
  --case-id mri_first_case \
  --dry-run
```

MRI QC with individual masks:

```bash
python src/02b_validate_segmentation.py \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/mri_first_case/lungs/combined_lungs.nii.gz \
  --individual-mask lung_left=outputs/totalsegmentator/mri_first_case/lungs/lung_left.nii.gz \
  --individual-mask lung_right=outputs/totalsegmentator/mri_first_case/lungs/lung_right.nii.gz \
  --modality mr \
  --case-id mri_first_case
```

The QC stage reads existing image and mask files only. It does not rerun
TotalSegmentator and does not inspect or save patient metadata.

## Bounding Box And Cropping

The new bbox/crop entry point is `src/03_compute_bbox.py`. The older
`src/03_bbox.py` is legacy lungmask-based code and is kept temporarily for
comparison only.

The bbox stage reads an existing source image and validated binary lung mask. It
does not rerun TotalSegmentator. It computes the smallest 3D box enclosing every
nonzero mask voxel, applies optional padding, then crops the source image,
combined mask, and optional individual masks.

Indexing conventions are explicit:

- NumPy arrays use `z, y, x`.
- SimpleITK indices use `x, y, z`.
- Bboxes are stored as half-open intervals `[min:max)`, matching Python slicing.

Padding can be requested in voxels or millimeters. Millimeter padding is
converted to voxel padding with `ceil(padding_mm / spacing)` independently for
each axis, so anisotropic spacing is respected. Padding is clamped at image
boundaries.

The JSON stores both voxel and physical bboxes. Physical bounds are computed with
SimpleITK transforms and respect image origin, spacing, and direction; they do
not assume identity direction matrices.

Outputs:

```text
outputs/bbox/<case_id>/bbox.json
outputs/crops/<case_id>/image_crop.nii.gz
outputs/crops/<case_id>/combined_lungs_crop.nii.gz
outputs/crops/<case_id>/<individual_mask>_crop.nii.gz
```

MRI dry-run:

```bash
python src/03_compute_bbox.py \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/mri_first_case/lungs/combined_lungs.nii.gz \
  --case-id mri_first_case \
  --padding-mm 5 \
  --dry-run
```

MRI crop with individual masks:

```bash
python src/03_compute_bbox.py \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/mri_first_case/lungs/combined_lungs.nii.gz \
  --case-id mri_first_case \
  --padding-mm 5 \
  --crop-individual-masks \
  --individual-mask outputs/totalsegmentator/mri_first_case/lungs/lung_left.nii.gz \
  --individual-mask outputs/totalsegmentator/mri_first_case/lungs/lung_right.nii.gz
```

## OsiriX Export

NIfTI masks cannot simply be opened as guaranteed DICOM overlays in OsiriX
because they do not carry references to the original DICOM SOP Instance UIDs and
Frame of Reference. For aligned review in OsiriX, this project exports a
referenced DICOM object when the original DICOM series has enough structural
metadata.

The selected export format is RT Structure Set. It is used because OsiriX
workflows commonly display RTSTRUCT ROI contours and the project needs named
review ROIs for the lung mask and computational crop box. The exporter uses
`pydicom`, the existing DICOM dependency, to construct and validate the object.
DICOM SEG is not implemented in this stage.

Before export, run a read-only capability inspection:

```bash
python src/04_export_osirix.py inspect \
  --dicom-dir data/raw/MRI_cases/first_case/dicom
```

The inspector reads only structural tags needed for references and geometry. It
does not print patient identifiers or metadata dumps. Secondary Capture images
are not rejected automatically, but RTSTRUCT export is refused unless
FrameOfReferenceUID, UIDs, orientation, positions, spacing, dimensions, and
slice mapping are sufficient.

Dry-run export:

```bash
python src/04_export_osirix.py export \
  --dicom-dir data/raw/MRI_cases/first_case/dicom \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/mri_first_case/lungs/combined_lungs.nii.gz \
  --individual-mask "Lung Left=outputs/totalsegmentator/mri_first_case/lungs/lung_left.nii.gz" \
  --individual-mask "Lung Right=outputs/totalsegmentator/mri_first_case/lungs/lung_right.nii.gz" \
  --bbox-json outputs/bbox/mri_first_case/bbox.json \
  --case-id mri_first_case \
  --format auto \
  --dry-run
```

If inspect and dry-run report RTSTRUCT compatibility, run the same command
without `--dry-run`. Then manually import both the original DICOM series and
`outputs/osirix/<case_id>/lung_segmentation_and_bbox_rtstruct.dcm` into OsiriX.
Automatic OsiriX launching is not implemented.

If a standards-compliant referenced object cannot be guaranteed, the exporter can
create a fallback review package containing NIfTI files, QC figures if already
available, bbox JSON, and a README. This fallback is for manual research review
only and is not an aligned DICOM overlay.

Generated DICOM-derived objects remain private medical data and must stay under
ignored `outputs/`. This export is for research validation, not diagnosis.

## Bounding-Box Validation

Segmentation QC and bbox QC are separate. The segmentation figures show the lung
mask and contours, but they do not automatically draw the computed crop box.
Use `src/03b_visualize_bbox.py` to create a reproducible PNG overlay of the
already-computed bbox without rerunning TotalSegmentator or changing the bbox.

The bbox JSON stores half-open NumPy bounds in `z, y, x`. The visualization maps
them into:

- axial views: `y` versus `x`
- coronal views: `z` versus `x`
- sagittal views: `z` versus `y`

The rectangle is drawn around the included voxel edges using the half-open stop
bound, so `[min:max)` includes voxels `min` through `max - 1`. CT uses the same
lung windowing defaults as segmentation QC. MRI uses percentile display scaling.

This PNG workflow is important for the current MRI because the source DICOMs may
be Secondary Capture objects without enough Frame of Reference metadata for a
guaranteed aligned RTSTRUCT overlay in OsiriX.

MRI bbox dry-run:

```bash
python src/03b_visualize_bbox.py \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/mri_first_case/lungs/combined_lungs.nii.gz \
  --bbox-json outputs/bbox/mri_first_case/bbox.json \
  --crop-image outputs/crops/mri_first_case/image_crop.nii.gz \
  --crop-mask outputs/crops/mri_first_case/combined_lungs_crop.nii.gz \
  --case-id mri_first_case \
  --modality mr \
  --dry-run
```
