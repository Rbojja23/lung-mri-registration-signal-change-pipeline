# CT/MRI Lung Bounding Box Pipeline

## Project overview

Version 1.1 is a research pipeline for geometry-preserving lung segmentation,
quality control, 3D bounding-box calculation, cropping, and interactive NIfTI
review. ITK-SNAP is the primary viewer. Tested RTSTRUCT export remains an
optional, compatibility-gated integration for suitable original DICOM series.

## Current capabilities

- Load CT or MR DICOM series and NIfTI images without reorientation.
- Run TotalSegmentator and derive binary whole-lung and semantic masks.
- Generate independent segmentation and bounding-box QC artifacts.
- Compute padded half-open bboxes and geometry-preserving image/mask crops.
- Prepare MRI left/right lung and bbox labels for ITK-SNAP.
- Optionally launch ITK-SNAP using argument lists and `shell=False`.
- Inspect DICOM spatial-reference capability and optionally export RTSTRUCT.

## Installation

Create a Python environment and install the active dependencies:

```bash
pip install -r requirements.txt
```

TotalSegmentator may download model weights on first inference. `lungmask` is
not an active dependency. ITK-SNAP is an external application and requires no
Python package.

## ITK-SNAP installation

Install ITK-SNAP on macOS at `/Applications/ITK-SNAP.app` (preferred) or
`~/Applications/ITK-SNAP.app`. Detection reads `Contents/Info.plist`, resolves
`CFBundleExecutable`, and verifies the executable. `itksnap-wt` is not required.

Read-only health check:

```bash
python src/pipeline.py launch --viewer itksnap --health-check
```

## deedsBCV setup checkpoint

[deedsBCV](https://github.com/mattiaspaul/deedsBCV) is an external MIT-licensed
registration project. This checkpoint provides reproducible source setup and
help-only executable health checks; registration itself is not implemented yet.
The pinned upstream revision is
`00b46539aa03d23988a241173d8f1f6c0c68dc6a`.

On Apple Silicon, do not use the normal upstream Makefile unchanged: its AVX2
and SSE4.2 flags target x86 CPUs. The setup script builds all four tools with
`clang++`, arm64-safe flags, and zlib. Homebrew `libomp` is used when available;
the script never installs it silently. A serial fallback is available with
`--without-openmp`. Cloned source and compiled binaries remain under ignored
directories; provenance is recorded in `third_party/deedsBCV/UPSTREAM.md`.

Run diagnostics without cloning or compiling:

```bash
scripts/setup_deedsbcv_macos.sh --check-only
```

Build with OpenMP, or explicitly select the serial fallback:

```bash
scripts/setup_deedsbcv_macos.sh
scripts/setup_deedsbcv_macos.sh --without-openmp
```

Run safe discovery and help-only health probes:

```bash
python src/check_deedsbcv.py --health-check
python src/check_deedsbcv.py --health-check --json
```

Compiled deedsBCV binaries, third-party build files, medical inputs, and future
registration outputs must not be committed.

## Registration input preparation

Registration roles are always explicit: callers must supply both `--fixed` and
`--moving`. The preparation tool does not infer roles from filenames, ordering,
dimensions, or intensities. Provisional assignments can be documented with
`--role-assignment-source`; omit `--roles-confirmed` until a researcher confirms
the direction. The moving image will eventually be transformed into the fixed
image's coordinate space.

This checkpoint accepts scalar 3D `.nii` and `.nii.gz` images, validates finite
values and geometry, and creates role-named `.nii.gz` working copies without
casting, normalization, reorientation, or resampling. Preparation requires a
matching fixed/moving voxel grid and approximately isotropic spacing. deedsBCV
does not properly account for anisotropic spacing, so anisotropic or mismatched
grids are reported in dry-run and rejected in real preparation until a later
resampling checkpoint.

Preview preparation without creating directories:

```bash
python src/prepare_registration_inputs.py \
  --fixed data/raw/registration_test/<case>/<fixed>.nii \
  --moving data/raw/registration_test/<case>/<moving>.nii \
  --case-id <non-identifying-case-id> \
  --experiment full \
  --role-assignment-source manual \
  --roles-confirmed \
  --dry-run
```

Prepared files are written under:

```text
outputs/registration/<case_id>/<experiment>/preprocessed/
    fixed.nii.gz
    moving.nii.gz
    preprocessing_manifest.json
```

The local manifest records role-confirmation status, safe geometry, streamed
source/output checksums, and post-write verification. It contains no arbitrary
NIfTI metadata. No registration binary is run by this preparation command.

## Data privacy

Medical data must remain local. `data/raw/`, `outputs/`, DICOM, NIfTI, and NumPy
arrays are ignored by Git. Reports use project-relative paths when possible and
exclude patient names, IDs, birth dates, accession numbers, full DICOM metadata,
environment dumps, and subprocess output.

## Directory structure

```text
data/raw/                         private input data
outputs/nifti/<case_id>/          geometry-preserving working NIfTI
outputs/totalsegmentator/<case>/  semantic and combined lung masks
outputs/figures/<case>/           segmentation and bbox QC
outputs/bbox/<case>/              bbox.json
outputs/crops/<case>/             image and mask crops
outputs/itksnap/<case>/           primary viewer products
outputs/osirix/<case>/            optional RTSTRUCT/fallback package
outputs/logs/<case>/              non-identifying pipeline log
legacy/                           unsupported historical lungmask scripts
```

## Input formats

The loader supports CT/MR DICOM directories and `.nii`/`.nii.gz` images. NIfTI
modality must be specified because its header does not reliably encode CT versus
MR. A DICOM-aligned NIfTI is sufficient for ITK-SNAP; the original DICOM is only
needed for optional RTSTRUCT work.

Geometry conventions:

- SimpleITK indices and size: `x,y,z`
- NumPy arrays: `z,y,x`
- Physical coordinates: millimeters in the image coordinate frame
- Bounding boxes: half-open `[min:max)` in NumPy `z,y,x`

## Pipeline stages

Default MR `run-all` order:

1. `inspect`
2. `segment`
3. `segmentation-qc`
4. `bbox`
5. `bbox-qc`
6. `prepare-itksnap`
7. optional `launch`

Viewer products are generated even with `--no-launch`. OsiriX export is not a
default stage; `--export-rtstruct` is an explicit opt-in.

## MRI workflow

MRI is fully supported through ITK-SNAP viewer preparation. TotalSegmentator's
`total_mr` task supplies semantic `lung_left` and `lung_right` masks. Viewer
labels are `1 Lung Left`, `2 Lung Right`, and `10 Lung Bounding Box`.

The supplied Secondary Capture MRI lacks `FrameOfReferenceUID`, so RTSTRUCT is
not appropriate. This does not affect the geometry-matched NIfTI/ITK-SNAP path.

## CT workflow and limitation

CT image loading, TotalSegmentator segmentation, QC, bbox calculation, and crop
commands remain supported. TotalSegmentator produces the five known lobe masks.
Version 1.1 does not yet configure those masks as an ITK-SNAP label table, so
unified CT `run-all` fails explicitly before claiming a complete viewer package.
Use individual CT stages until the five-lobe viewer mapping is validated.

## TotalSegmentator tasks

- CT: `total`
- MR: `total_mr`

The active pipeline accepts `auto`, `cpu`, `mps`, or `cuda` device selection and
records non-identifying software/task provenance.

## Segmentation QC

Segmentation QC validates binary masks and exact image geometry, then creates
filled and contour overlays plus `segmentation_qc.json`. It is independent of
bbox QC.

## Bounding box and crop

The bbox is calculated from every nonzero combined-lung voxel, supports physical
or voxel padding, and is stored as `[z_min:z_max, y_min:y_max, x_min:x_max)`.
SimpleITK region-of-interest cropping preserves spacing/direction and updates the
crop origin. Bbox QC independently verifies containment and crop consistency.

## ITK-SNAP viewer products

```text
outputs/itksnap/<case_id>/
    lungs_multilabel.nii.gz
    bbox_label.nii.gz
    lungs_and_bbox_labels.nii.gz
    label_descriptions.txt
    launch_manifest.json
```

- `lungs_multilabel.nii.gz` is the authoritative semantic segmentation.
- `bbox_label.nii.gz` is the authoritative one-voxel, six-face bbox shell.
- `lungs_and_bbox_labels.nii.gz` is a visualization convenience product.
- Label `10` wins on bbox/lung overlap only in the combined viewer image.
- Do not use the combined image for quantitative lung-volume measurements.
- Every label image preserves reference size, spacing, origin, and direction.

The shell marks each minimum face and each exclusive maximum minus one, making
the half-open bbox visible throughout axial, coronal, and sagittal traversal.

## Unified run-all examples

Dry-run without application launch:

```bash
python src/pipeline.py run-all \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --input-type nifti --modality mr --case-id mri_first_case \
  --device mps --padding-mm 5 --viewer itksnap --no-launch --dry-run
```

Resume and prepare without launch:

```bash
python src/pipeline.py run-all \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --input-type nifti --modality mr --case-id mri_first_case \
  --resume --viewer itksnap --no-launch
```

Resume and launch:

```bash
python src/pipeline.py run-all \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --input-type nifti --modality mr --case-id mri_first_case \
  --resume --launch --viewer itksnap
```

`--dry-run` validates known source inputs, plans every downstream output and
launch role, but writes nothing, runs no inference, and launches no application.

## Individual-stage examples

```bash
python src/pipeline.py segmentation-qc \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/mri_first_case/lungs/combined_lungs.nii.gz \
  --individual-mask lung_left=outputs/totalsegmentator/mri_first_case/lungs/lung_left.nii.gz \
  --individual-mask lung_right=outputs/totalsegmentator/mri_first_case/lungs/lung_right.nii.gz \
  --modality mr --case-id mri_first_case
```

```bash
python src/pipeline.py bbox-qc \
  --image outputs/nifti/mri_first_case/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/mri_first_case/lungs/combined_lungs.nii.gz \
  --bbox-json outputs/bbox/mri_first_case/bbox.json \
  --crop-image outputs/crops/mri_first_case/image_crop.nii.gz \
  --crop-mask outputs/crops/mri_first_case/combined_lungs_crop.nii.gz \
  --modality mr --case-id mri_first_case
```

Manual ITK-SNAP preparation and launch are available through `prepare-itksnap`
and `launch`; use each command's `--help` for its complete argument contract.

## Optional RTSTRUCT/OsiriX path

This tested compatibility path is only for source DICOM series with complete,
consistent spatial references, unique SOP Instance UIDs, and one-to-one slice
mapping. Inspect capability before export:

```bash
python src/pipeline.py inspect --dicom-dir data/raw/<case>/dicom
```

The explicit `export` command or `run-all --export-rtstruct --dicom-dir ...`
performs validation. Missing `FrameOfReferenceUID` prevents safe RTSTRUCT but is
irrelevant to ITK-SNAP. OsiriX/Horos launch remains available explicitly and is
never selected as a silent fallback.

## Outputs

Generated NIfTI, JSON, figures, logs, RTSTRUCT objects, and review packages are
private research artifacts under ignored `outputs/`. Existing outputs are not
replaced without `--overwrite`, except that `--resume` may regenerate only an
invalid/partial selected-case ITK-SNAP package.

## Troubleshooting

- Run `launch --viewer itksnap --health-check` for installation diagnostics.
- Use the DICOM-aligned NIfTI if image/mask geometry differs.
- Use `--resume` only with products from the same source image and geometry.
- A refused RTSTRUCT indicates unsafe DICOM references; use ITK-SNAP instead.
- CT `run-all` intentionally reports the unconfigured viewer-label limitation.

## Known limitations

- ITK-SNAP detection currently targets macOS application bundles.
- CT five-lobe ITK-SNAP viewer preparation is not configured in v1.1.
- Secondary Capture DICOM without `FrameOfReferenceUID` cannot safely support
  RTSTRUCT export.
- Outputs require research validation and are not diagnostic annotations.

## Research-use-only disclaimer

This software is for research validation only. It is not a medical device and
must not be used for diagnosis or clinical decision-making.
