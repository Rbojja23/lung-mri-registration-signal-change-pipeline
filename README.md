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
registration project. The project pins upstream revision
`00b46539aa03d23988a241173d8f1f6c0c68dc6a` and applies the tracked minimal
patch in `third_party/deedsBCV/patches/` during setup. The patch fixes confirmed
C++ undefined behavior that could produce zero NIfTI dimensions; it does not
change registration mathematics.

On Apple Silicon, do not use the normal upstream Makefile unchanged: its AVX2
and SSE4.2 flags target x86 CPUs. The setup script requires `linearBCV` and
`deedsBCV`; `applyBCV` and `applyBCVfloat` are optional because their upstream
x86 intrinsic headers may be unavailable on arm64. Optional failures are
reported without invalidating a successful required build. Homebrew `libomp`
is used when available; `--without-openmp` selects the serial fallback.
Cloned source, build-state metadata, and compiled binaries remain ignored;
patch provenance is recorded in `third_party/deedsBCV/UPSTREAM.md`.

Run diagnostics without cloning or compiling:

```bash
scripts/setup_deedsbcv_macos.sh --check-only
```

Fresh setup with OpenMP, or explicitly with the serial fallback:

```bash
scripts/setup_deedsbcv_macos.sh
scripts/setup_deedsbcv_macos.sh --without-openmp
```

Run safe discovery and help-only health probes:

```bash
python src/check_deedsbcv.py --health-check
python src/check_deedsbcv.py --health-check --json
```

To force a clean rebuild safely, first save or confirm that the ignored source
contains no needed changes beyond the tracked patch, then run:

```bash
scripts/setup_deedsbcv_macos.sh --force
```

The script refuses unrelated source modifications instead of silently
discarding them. Registration outputs remain research-only and non-diagnostic.

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

## Shared-crop registration preparation

The first cropped experiment uses the fixed image's validated half-open lung
bbox for both fixed and moving images. It never computes an independent moving
bbox: that could select different voxel regions and invalidate the controlled
comparison. Fixed and moving full-volume inputs must already share the exact
size, spacing, origin, and direction. The crop converts bbox order from NumPy
`z,y,x` to SimpleITK `x,y,z` and uses `RegionOfInterest`, without resampling or
reorientation.

Preview the Case 1 crop without writing files:

```bash
python src/prepare_shared_registration_crop.py \
  --fixed outputs/registration/registration_case_1/full/preprocessed/fixed.nii.gz \
  --moving outputs/registration/registration_case_1/full/preprocessed/moving.nii.gz \
  --bbox-json outputs/bbox/registration_case_1_fixed/bbox.json \
  --case-id registration_case_1 \
  --source-experiment full \
  --experiment cropped \
  --role-assignment-source inferred-from-repeated-bimg1 \
  --dry-run
```

Remove `--dry-run` to create:

```text
outputs/registration/registration_case_1/cropped/preprocessed/
    fixed.nii.gz
    moving.nii.gz
    preprocessing_manifest.json
    shared_crop_manifest.json
```

Boundary-touch flags are informational. When a bbox reaches an image edge,
requested anatomical padding may have been clipped; this does not change the
shared-crop rule. Existing products require `--overwrite`, which replaces only
the four cropped preprocessing products and preserves the full experiment.

After preparation, preview the existing registration runner without executing
registration:

```bash
python src/run_deedsbcv_registration.py \
  --prepared-dir outputs/registration/registration_case_1/cropped/preprocessed \
  --case-id registration_case_1 \
  --experiment cropped \
  --bin-dir third_party/deedsBCV/build-arm64 \
  --dry-run
```

## Registration QC and experiment comparison

Registration QC compares the fixed image with the original moving image and
then with the registered moving image. It reports 3D MSE and MAE (lower is
better), global normalized cross-correlation and deterministic histogram-based
mutual information (higher is generally better), runtime, represented volume,
and the final deedsBCV SSD diagnostic when it can be parsed. These intensity
metrics do not prove anatomical correctness. A validated memory-efficient 3D
structural-similarity implementation is not provided, so SSIM is deliberately
omitted instead of averaging unrelated 2D scores.

Optional aligned binary masks enable fixed-mask and intersection-domain
metrics. Dice is reported only when corresponding anatomical masks are
provided; it cannot be inferred from intensity images.

Evaluate an existing experiment without rerunning registration:

```bash
python src/evaluate_registration.py \
  --case-id registration_case_1 \
  --experiment full \
  --experiment-dir outputs/registration/registration_case_1/full \
  --fixed outputs/registration/registration_case_1/full/preprocessed/fixed.nii.gz \
  --moving outputs/registration/registration_case_1/full/preprocessed/moving.nii.gz \
  --registered outputs/registration/registration_case_1/full/deformable/moving_registered.nii.gz \
  --registration-manifest outputs/registration/registration_case_1/full/registration_manifest.json
```

Each experiment receives `qc/registration_qc.json` plus before/after overlays,
checkerboards, and matched-scale difference images. ITK-SNAP review can be
requested with `--launch-itksnap`, which opens fixed as the main image and both
moving images as intensity overlays. Use `--dry-run-launch` to inspect the
launch plan without opening an application.

Compare completed full and cropped QC manifests with:

```bash
python src/evaluate_registration.py \
  --compare \
  --case-id registration_case_1 \
  --full-qc outputs/registration/registration_case_1/full/qc/registration_qc.json \
  --cropped-qc outputs/registration/registration_case_1/cropped/qc/registration_qc.json \
  --output-dir outputs/registration/registration_case_1/comparison
```

The comparison reports metric-wise and runtime winners but never declares an
overall anatomical winner without expert visual review. All outputs remain
research-only and non-diagnostic.

## Preliminary Registered Intensity-Change Mapping

The standalone intensity-change checkpoint generates a **preliminary voxelwise
registered intensity-change map**, not a validated ventilation biomarker. It
computes signed `fixed - registered`, absolute `abs(fixed - registered)`, and,
only with an explicit denominator policy, a normalized map. Analytical values
are restricted to the fixed combined-lung mask and remain NaN outside their
valid domains.

The preliminary intensity-change command's normalized option requires either an
explicit absolute fixed-intensity threshold or an explicitly selected
exploratory lung-intensity percentile. The separate normalized-map checkpoint
described below also supports the professor-requested epsilon equation. Use
`--skip-normalized` when only signed and absolute maps are wanted.

Analytical NIfTIs preserve unclipped floating-point values and fixed-image
geometry. Separate display NIfTIs are finite, zero outside the lung mask, and
clipped at a documented lung-only display percentile. PNGs retain fixed anatomy
outside the lungs and show colored overlays only for valid lung voxels.
deedsBCV's trilinear interpolation may smooth or mix registered intensities.

Moving modality, respiratory state/ordering, and intensity calibration remain
unconfirmed. These products are research-only and non-diagnostic. They may be
viewed as ITK-SNAP intensity overlays, for example:

```bash
open -a "ITK-SNAP" --args \
  -g outputs/registration/<case_id>/<experiment>/preprocessed/fixed.nii.gz \
  -o outputs/registration/<case_id>/<experiment>/intensity_change/signed_difference_display.nii.gz \
  -s outputs/totalsegmentator/<fixed_case>/lungs/combined_lungs.nii.gz
```

## Standalone Registration Metric Analysis

The metric-analysis checkpoint explains an existing registration without
running or changing registration. It reports before/after MSE, MAE, RMSE, NCC,
mutual information, and SSD for the whole image and, when an aligned lung mask
is supplied, for lung-only and outside-lung domains. Each metric includes its
direction, absolute improvement, and percentage improvement.

The report also summarizes signed and absolute voxelwise errors, compares
moving and registered intensity distributions, and records the effect expected
from deedsBCV's trilinear interpolation. MSE emphasizes large errors, MAE
responds directly to absolute intensity and interpolation changes, RMSE retains
the original intensity units, NCC measures centered pattern agreement, and MI
measures statistical dependence. No metric is treated as universally best.

The command writes `registration_metric_report.json`, a metric summary,
error-distribution plots, and matched-scale grayscale-anatomy error overlays to
an explicitly selected analysis directory. Preview validation and calculations
without writing anything using `--dry-run`:

```bash
python src/analyze_registration_metrics.py \
  --fixed outputs/registration/<case_id>/<experiment>/preprocessed/fixed.nii.gz \
  --moving outputs/registration/<case_id>/<experiment>/preprocessed/moving.nii.gz \
  --registered outputs/registration/<case_id>/<experiment>/deformable/moving_registered.nii.gz \
  --lung-mask outputs/totalsegmentator/<fixed_case>/lungs/combined_lungs.nii.gz \
  --output-dir outputs/registration/<case_id>/<experiment>/metric_analysis \
  --dry-run --json
```

## Exploratory Functional-Lung Analysis

The standalone functional-lung checkpoint explores whether spatially coherent
regional intensity changes exist after registration. Despite its name, it does
not calculate or claim validated ventilation, physiology, or a clinical
biomarker. It computes only signed `fixed - registered` and absolute
`abs(fixed - registered)` intensity change. It does not normalize by fixed
intensity.

All calculations are restricted to the combined fixed-space lung mask. Supplied
left/right masks enable semantic regional summaries; if they are omitted, the
tool creates explicitly labeled geometric physical-x partitions that are not
semantic segmentations. It reports whole/left/right descriptive statistics,
axial slice trends, descriptive left-right asymmetry, and the largest
6-connected positive and negative sign regions without biological
interpretation.

Outputs include `functional_lung_report.json`, axial/coronal/sagittal overlays,
a multi-slice panel, slice mean/standard-deviation plots, and lung-only
histograms. Every overlay uses fixed grayscale anatomy, lung-only color, and a
shared symmetric scale centered at zero. Respiratory state, moving modality,
intensity comparability, registration error, and interpolation effects remain
important unresolved limitations.

Preview the analysis without writing files:

```bash
python src/analyze_functional_lung.py \
  --fixed outputs/registration/<case_id>/<experiment>/preprocessed/fixed.nii.gz \
  --registered outputs/registration/<case_id>/<experiment>/deformable/moving_registered.nii.gz \
  --lung-mask outputs/totalsegmentator/<fixed_case>/lungs/combined_lungs.nii.gz \
  --left-lung outputs/totalsegmentator/<fixed_case>/lungs/lung_left.nii.gz \
  --right-lung outputs/totalsegmentator/<fixed_case>/lungs/lung_right.nii.gz \
  --output-dir outputs/registration/<case_id>/<experiment>/functional_lung \
  --dry-run --json
```

## Exploratory Normalized Registered Intensity-Change Map

The standalone exploratory checkpoint creates an **Exploratory normalized
registered intensity-change map**. It supports the original threshold policy,
`(fixed - registered) / fixed` where `abs(fixed) >= threshold`, and the
professor-requested epsilon policy, `(fixed - registered) / (fixed + epsilon)`.
It is not a validated ventilation measurement or biomarker and makes no
physiological claim. Respiratory phase, moving modality, MR intensity
calibration, and cross-acquisition intensity comparability remain unresolved;
registration and trilinear interpolation also contribute to observed values.

Both policies calculate only in the combined fixed-space lung mask. Absolute
mode requires a finite, strictly positive threshold and leaves excluded
denominators as NaN. Epsilon mode performs no threshold-based exclusion and
requires either `--epsilon machine` or an explicit finite, strictly positive
number. It uses float64 computation; `machine` means exactly
`2.220446049250313e-16`, NumPy's float64 machine epsilon near 1. This protects
an exact zero denominator numerically, but does not meaningfully stabilize
genuinely small intensities or establish a scientifically preferred value.
Larger explicit values are supported for sensitivity analysis and are never
selected automatically.

The requested `fixed + epsilon` equation is preserved exactly. A positive
epsilon is asymmetric around zero: it moves negative fixed intensities toward
zero and is undefined if `fixed == -epsilon`. The report records negative,
exact-zero, and descriptively near-zero fixed-lung voxel counts so this
limitation is visible. No interpolation or resampling is performed here.

`normalized_difference.nii.gz` is an unclipped float32 analytical map with fixed
geometry. `normalized_difference_display.nii.gz` is a separate finite display
copy, zero outside valid lung voxels and symmetrically clipped around zero.
Axial, coronal, sagittal, and multi-slice figures use grayscale anatomy with a
transparent valid-lung overlay and one shared diverging scale.

Preview the Exploratory normalized registered intensity-change map without
writing files in the backward-compatible absolute-threshold mode:

```bash
python src/generate_normalized_intensity_map.py \
  --fixed outputs/registration/<case_id>/<experiment>/preprocessed/fixed.nii.gz \
  --registered outputs/registration/<case_id>/<experiment>/deformable/moving_registered.nii.gz \
  --lung-mask outputs/totalsegmentator/<fixed_case>/lungs/combined_lungs.nii.gz \
  --left-lung outputs/totalsegmentator/<fixed_case>/lungs/lung_left.nii.gz \
  --right-lung outputs/totalsegmentator/<fixed_case>/lungs/lung_right.nii.gz \
  --output-dir outputs/registration/<case_id>/<experiment>/normalized_intensity \
  --denominator-policy absolute \
  --denominator-threshold <VALUE_TO_CONFIRM> \
  --dry-run --json
```

Preview the epsilon mode in a separate directory so threshold-based products
are not overwritten:

```bash
python src/generate_normalized_intensity_map.py \
  --fixed outputs/registration/<case_id>/<experiment>/preprocessed/fixed.nii.gz \
  --registered outputs/registration/<case_id>/<experiment>/deformable/moving_registered.nii.gz \
  --lung-mask outputs/totalsegmentator/<fixed_case>/lungs/combined_lungs.nii.gz \
  --left-lung outputs/totalsegmentator/<fixed_case>/lungs/lung_left.nii.gz \
  --right-lung outputs/totalsegmentator/<fixed_case>/lungs/lung_right.nii.gz \
  --output-dir outputs/registration/<case_id>/<experiment>/normalized_intensity_epsilon \
  --denominator-policy epsilon \
  --epsilon machine \
  --dry-run --json
```

The analytical NIfTI is float32, unclipped, fixed-geometry, and NaN outside its
lung-domain calculation. The separate display NIfTI is finite, zero outside the
lung, and symmetrically clipped only for visualization. Output replacement
still requires `--overwrite`; choosing the dedicated epsilon directory keeps
the original threshold products intact by default. All outputs are
research-only and non-diagnostic.

## Default deedsBCV registration

Registration requires a completed `preprocessed/` directory from the input
preparation checkpoint. The standalone runner validates its manifest and
checksums, then runs full-volume affine registration followed by deformable
registration. Fixed and moving roles are never inferred or reversed.

The baseline uses upstream defaults by omission. The generated commands contain
only the required `-F`, `-M`, and `-O` arguments, plus the affine matrix `-A` for
the deformable stage; no `-G`, `-L`, `-Q`, `-a`, or segmentation option is added.

Preview the exact commands without creating files or executing binaries:

```bash
python src/run_deedsbcv_registration.py \
  --prepared-dir outputs/registration/<case_id>/full/preprocessed \
  --case-id <case_id> \
  --experiment full \
  --bin-dir third_party/deedsBCV/build-arm64 \
  --dry-run
```

Registration products are scoped to the selected experiment:

```text
outputs/registration/<case_id>/full/
    preprocessed/                     preserved inputs
    affine/
        affine_matrix.txt
        linear_stdout.log
        linear_stderr.log
    deformable/
        moving_registered.nii.gz
        displacements.dat
        deeds_stdout.log
        deeds_stderr.log
    registration_manifest.json
```

The timeout applies independently to each stage and defaults to
`registration.timeout_seconds` in `config/default.yaml`. A timeout,
interruption, nonzero exit, or malformed output leaves an incomplete manifest
and is never reported as success. `--resume` reuses only semantically matching,
verified stages; `--overwrite` replaces only registration products for the
selected case and experiment while preserving prepared inputs. These options
are mutually exclusive.

This workflow is research-only and non-diagnostic. Output plausibility checks
do not establish anatomical correctness; registration QC is the next planned
checkpoint. Cropped registration is not implemented here.

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
