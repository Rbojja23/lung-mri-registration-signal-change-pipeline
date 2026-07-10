# CT/MRI Lung Bounding Box Pipeline

Research pipeline for loading CT or MRI medical images, segmenting lungs with
TotalSegmentator, validating the segmentation, computing a geometry-preserving
3D lung bounding box, cropping image/mask volumes, creating QC figures, and
exporting OsiriX-compatible review artifacts when the source DICOM series safely
supports them.

This project is for research validation only. It is not a diagnostic tool.

## Privacy

Medical image data must remain local. `data/raw/`, `outputs/`, DICOM, NIfTI, and
generated arrays are ignored by Git. Generated RTSTRUCT objects and review
packages are also private medical data and must stay under ignored `outputs/`.

The pipeline avoids printing patient names, patient IDs, accession numbers, birth
dates, and full DICOM metadata dumps.

## Installation

Use a Python environment with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

TotalSegmentator may download model weights on first real use. Inference runs
locally.

## Directory Structure

```text
data/raw/                         private local DICOM/NIfTI data
outputs/nifti/<case_id>/          DICOM-aligned working NIfTI images
outputs/totalsegmentator/<case>/  raw and derived lung masks
outputs/figures/<case>/           segmentation and bbox QC figures
outputs/bbox/<case>/              bbox.json
outputs/crops/<case>/             cropped image and masks
outputs/osirix/<case>/            RTSTRUCT or fallback review package
outputs/logs/<case>/              non-identifying pipeline run logs
legacy/                           deprecated CT-only lungmask scripts
```

## Geometry

Geometry is preserved through SimpleITK image metadata:

- SimpleITK index order: `x, y, z`
- NumPy array order: `z, y, x`
- Physical coordinates: millimeters in the image coordinate frame
- Bounding boxes: half-open intervals `[min:max)` in NumPy `z, y, x`

DICOM-aligned NIfTI files are preferred for processing because they preserve a
clear mapping back to the source DICOM slices.

## TotalSegmentator

The active segmentation stage is `src/02_segment_lungs_totalseg.py` and the
unified CLI command is `segment`.

- CT uses TotalSegmentator task `total`
- MRI uses TotalSegmentator task `total_mr`

The old CT-only lungmask scripts were moved to `legacy/` and are deprecated.
`lungmask` is no longer an active dependency.

## Validation And Cropping

Segmentation QC creates filled-mask and contour overlays plus
`segmentation_qc.json`. Bbox QC is separate: segmentation figures do not
automatically display the crop box.

The bbox stage validates a binary combined lung mask, computes a half-open 3D
bbox, supports millimeter or voxel padding, saves `bbox.json`, and crops image
and mask NIfTI files while updating crop origins correctly.

## OsiriX Export

NIfTI masks cannot be guaranteed DICOM overlays in OsiriX because they lack
source SOP Instance UID and Frame of Reference references. The export stage first
validates the original DICOM series using non-identifying structural tags.

When safe, it writes an RT Structure Set containing:

- Whole Lung Segmentation
- optional individual lung ROIs
- Lung Bounding Box

Secondary Capture images are not rejected automatically, but RTSTRUCT export is
refused if FrameOfReferenceUID, geometry, or one-to-one slice mapping is missing.
If RTSTRUCT is unsafe, a fallback review package is created instead.

## Unified CLI

Version:

```bash
python src/pipeline.py --version
```

MRI run-all:

```bash
python src/pipeline.py run-all \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --input-type nifti \
  --modality mr \
  --case-id mri_first_case \
  --dicom-dir data/raw/MRI_cases/first_case/dicom \
  --device mps \
  --padding-mm 5 \
  --no-launch
```

Resume:

```bash
python src/pipeline.py run-all \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --input-type nifti \
  --modality mr \
  --case-id mri_first_case \
  --dicom-dir data/raw/MRI_cases/first_case/dicom \
  --resume \
  --no-launch
```

Configuration defaults live in `config/default.yaml`; command-line arguments
override configuration values.

## Individual Stage Examples

Segmentation QC:

```bash
python src/pipeline.py segmentation-qc \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/mri_first_case/lungs/combined_lungs.nii.gz \
  --individual-mask lung_left=outputs/totalsegmentator/mri_first_case/lungs/lung_left.nii.gz \
  --individual-mask lung_right=outputs/totalsegmentator/mri_first_case/lungs/lung_right.nii.gz \
  --modality mr \
  --case-id mri_first_case
```

BBox QC:

```bash
python src/pipeline.py bbox-qc \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/mri_first_case/lungs/combined_lungs.nii.gz \
  --bbox-json outputs/bbox/mri_first_case/bbox.json \
  --crop-image outputs/crops/mri_first_case/image_crop.nii.gz \
  --crop-mask outputs/crops/mri_first_case/combined_lungs_crop.nii.gz \
  --modality mr \
  --case-id mri_first_case
```

Manual viewer launch:

```bash
python src/pipeline.py launch \
  --dicom-dir data/raw/MRI_cases/first_case/dicom \
  --rtstruct outputs/osirix/mri_first_case/lung_segmentation_and_bbox_rtstruct.dcm \
  --viewer osirix-lite \
  --dry-run
```

## Troubleshooting

- If NIfTI geometry does not match DICOM geometry, use the DICOM-aligned NIfTI
  created from the same source series.
- If RTSTRUCT export is refused, inspect the DICOM reference report and use the
  fallback review package for manual research review.
- Existing outputs are not overwritten unless `--overwrite` is supplied.
- Use `--resume` with `run-all` to reuse complete validated outputs.

See [docs/architecture.md](docs/architecture.md) for the data-flow diagram and
coordinate details.
