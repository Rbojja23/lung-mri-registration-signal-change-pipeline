# CT/MRI Lung Segmentation, Registration, and Signal-Change Pipeline

## Table of contents

1. [Project summary](#project-summary)
2. [Research scope](#research-scope)
3. [Workflow overview](#workflow-overview)
4. [Installation](#installation)
5. [Default configuration](#default-configuration)
6. [Quick start](#quick-start)
7. [Image loading](#image-loading)
8. [Lung segmentation](#lung-segmentation)
9. [Segmentation quality control](#segmentation-quality-control)
10. [Bounding box and cropping](#bounding-box-and-cropping)
11. [ITK-SNAP preparation](#itk-snap-preparation)
12. [Registration input preparation](#registration-input-preparation)
13. [deedsBCV registration](#deedsbcv-registration)
14. [Registration quality control](#registration-quality-control)
15. [Cross-case difference comparison](#cross-case-difference-comparison)
16. [Registered intensity-change analysis](#registered-intensity-change-analysis)
17. [Fractional MRI signal-change proxy](#fractional-mri-signal-change-proxy)
18. [Native, mean, and Gaussian comparison](#native-mean-and-gaussian-comparison)
19. [Horos DICOM export](#horos-dicom-export)
20. [Presentation workflow](#presentation-workflow)
21. [Inputs and outputs](#inputs-and-outputs)
22. [Testing](#testing)
23. [Data privacy](#data-privacy)
24. [Recorded limitations](#recorded-limitations)
25. [Future directions](#future-directions)

---

## Project summary

This repository provides a research-only workflow for:

- Loading CT or MRI data while preserving spatial geometry.
- Segmenting the lungs with TotalSegmentator.
- Validating segmentation masks.
- Calculating padded three-dimensional lung bounding boxes.
- Producing geometry-preserving full and cropped images.
- Preparing images and labels for ITK-SNAP.
- Registering moving MRI volumes to a fixed MRI with deedsBCV.
- Evaluating registration quality.
- Calculating exploratory registered-space MRI intensity-change maps.
- Comparing native, box-mean-filtered, and Gaussian-weighted maps.
- Exporting presentation-ready figures.
- Exporting fixed-anatomy RGB DICOM Secondary Capture series for Horos or OsiriX.

The pipeline keeps analytical outputs, visualization products, validity maps, provenance reports, and registration quality-control products separate.

---

## Research scope

This software is intended for research validation only.

The MRI ratio products must be described as:

> Exploratory registered-space fractional MRI signal-change proxies.

They must not be described as:

- Validated fractional ventilation
- Validated perfusion
- CT density change
- A clinical biomarker
- A diagnostic measurement

MRI signal intensity is generally acquisition-dependent and is not physically calibrated like CT Hounsfield units. Physiological interpretation requires independent validation.

---

## Workflow overview

```text
CT or MRI input
    |
    v
Geometry-preserving loading
    |
    v
TotalSegmentator lung segmentation
    |
    v
Segmentation quality control
    |
    v
Lung bounding box and geometry-preserving crop
    |
    +--------------------------+
    |                          |
    v                          v
ITK-SNAP products       Registration preparation
                               |
                               v
                    Affine + deformable registration
                               |
                               v
                      Registration quality control
                               |
                               v
                  Registered intensity-change analyses
                               |
             +-----------------+------------------+
             |                 |                  |
             v                 v                  v
          Native         7×7×7 mean        Gaussian weighted
             |                 |                  |
             +-----------------+------------------+
                               |
                               v
               Shared-window comparison figures
                               |
                               v
             Optional Horos RGB DICOM export
```

The original moving image is retained for registration quality control, but the registered moving image must be used for voxelwise comparison with the fixed image.

---

## Installation

Create a Python environment and install the active dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- NumPy
- SciPy
- SimpleITK
- Matplotlib
- NiBabel
- pydicom
- TotalSegmentator
- PyYAML

TotalSegmentator may download model weights during its first run.

ITK-SNAP and Horos are external applications and are not installed through `requirements.txt`.

---

## Default configuration

Defaults are stored in:

```text
config/default.yaml
```

Current configuration:

```yaml
output_root: outputs

totalsegmentator:
  device: auto
  fast: false

bbox:
  padding_mm: 5

display:
  ct_window_center: -500
  ct_window_width: 1500
  mr_lower_percentile: 1
  mr_upper_percentile: 99

launch:
  enabled: false
  viewer: itksnap

registration:
  deeds_bin_dir: null
  timeout_seconds: 3600
```

General safety behavior:

- Existing products are protected by default.
- Use `--dry-run` to validate and preview without writing.
- Use `--json` when available for machine-readable output.
- Use `--overwrite` only when replacing the selected output is intentional.
- Source NIfTIs are treated as read-only.
- Geometry mismatches are rejected rather than silently resampled.

---

## Quick start

### Inspect an image

```bash
python src/pipeline.py inspect \
  --input data/raw/<case>/image.nii.gz \
  --input-type nifti \
  --modality mr
```

### Run the MRI segmentation and viewer-preparation workflow

Preview:

```bash
python src/pipeline.py run-all \
  --image data/raw/<case>/image.nii.gz \
  --input-type nifti \
  --modality mr \
  --case-id <case_id> \
  --device auto \
  --padding-mm 5 \
  --viewer itksnap \
  --no-launch \
  --dry-run
```

Run:

```bash
python src/pipeline.py run-all \
  --image data/raw/<case>/image.nii.gz \
  --input-type nifti \
  --modality mr \
  --case-id <case_id> \
  --device auto \
  --padding-mm 5 \
  --viewer itksnap \
  --no-launch
```

### Generate the native, mean, and Gaussian proxy comparison

Preview:

```bash
python src/generate_fractional_signal_change_proxy.py \
  --dry-run \
  --json
```

Generate:

```bash
python src/generate_fractional_signal_change_proxy.py
```

### Export the six Horos DICOM series

Preview:

```bash
python src/export_fractional_proxy_dicom.py \
  --dry-run \
  --json
```

Generate:

```bash
python src/export_fractional_proxy_dicom.py
```

---

## Image loading

### Purpose

The image-loading component accepts CT or MRI inputs while preserving:

- Image size
- Voxel spacing
- Origin
- Direction matrix
- Physical coordinate system

Supported inputs:

- DICOM directories
- `.nii`
- `.nii.gz`

### Defaults

- Input type can be selected automatically.
- NIfTI modality must be supplied because NIfTI headers do not reliably distinguish CT from MRI.
- No automatic reorientation is performed.
- No automatic intensity normalization is performed.

### Main inputs

- Input image or DICOM directory
- Input type
- Modality
- Non-identifying case ID

### Adjustable parameters

- `--input-type auto|dicom|nifti`
- `--modality auto|ct|mr`
- `--case-id`
- `--output-root`
- `--overwrite`
- `--dry-run`

### Example

```bash
python src/pipeline.py inspect \
  --input data/raw/<case>/image.nii.gz \
  --input-type nifti \
  --modality mr
```

### Typical output

```text
outputs/nifti/<case_id>/
    geometry-preserving working NIfTI
```

### Limitations

- Incorrect source metadata is not automatically repaired.
- NIfTI modality cannot be inferred reliably.
- DICOM loading depends on a consistent source series.
- Conversion to NIfTI does not create missing clinical reference metadata.

---

## Lung segmentation

### Purpose

TotalSegmentator produces semantic lung masks.

### Defaults

For MRI:

```text
Task: total_mr
```

For CT:

```text
Task: total
```

Other defaults:

```text
Device: auto
Fast mode: disabled
```

### Inputs

- Geometry-preserving CT or MRI image
- Modality
- Case ID

### Adjustable parameters

- `--device auto|cpu|mps|cuda`
- `--fast`
- `--output-root`
- `--overwrite`

### Example

```bash
python src/pipeline.py segment \
  --image outputs/nifti/<case_id>/image_from_dicom.nii.gz \
  --input-type nifti \
  --modality mr \
  --case-id <case_id> \
  --device auto
```

### Output

For MRI:

```text
outputs/totalsegmentator/<case_id>/lungs/
    lung_left.nii.gz
    lung_right.nii.gz
    combined_lungs.nii.gz
```

For CT, TotalSegmentator produces the supported lung-lobe masks.

### Limitations

- Automated segmentation is not equivalent to expert validation.
- MRI artifacts and low contrast can affect segmentation.
- CT five-lobe ITK-SNAP label integration is not fully configured.
- Segmentation uncertainty is not currently propagated into downstream statistics.

---

## Segmentation quality control

### Purpose

Segmentation quality control verifies:

- Binary mask values
- Nonempty masks
- Exact geometry agreement
- Left/right mask separation
- Combined-mask consistency
- Visual alignment with the source image

### Inputs

- Source image
- Combined lung mask
- Optional left and right lung masks
- Modality
- Case ID

### Example

```bash
python src/pipeline.py segmentation-qc \
  --image outputs/nifti/<case_id>/image_from_dicom.nii.gz \
  --mask outputs/totalsegmentator/<case_id>/lungs/combined_lungs.nii.gz \
  --individual-mask lung_left=outputs/totalsegmentator/<case_id>/lungs/lung_left.nii.gz \
  --individual-mask lung_right=outputs/totalsegmentator/<case_id>/lungs/lung_right.nii.gz \
  --modality mr \
  --case-id <case_id>
```

### Outputs

```text
outputs/figures/<case_id>/
    filled overlays
    contour overlays
    segmentation_qc.json
```

### Limitations

- Geometry validation does not prove anatomical accuracy.
- Visual quality control remains necessary.
- A segmentation may be technically valid but anatomically incomplete.

---

## Bounding box and cropping

### Purpose

The bounding-box component finds the smallest three-dimensional region containing every nonzero lung-mask voxel and adds configurable padding.

Bounding boxes use half-open NumPy intervals:

```text
[z_min:z_max, y_min:y_max, x_min:x_max)
```

### Default

```text
Physical padding: 5 mm
```

### Inputs

- Source image
- Combined lung mask
- Optional individual masks

### Adjustable parameters

- `--padding-mm`
- `--padding-voxels`
- `--crop-individual-masks`
- `--individual-mask`
- `--output-root`
- `--overwrite`

### Outputs

```text
outputs/bbox/<case_id>/
    bbox.json

outputs/crops/<case_id>/
    image_crop.nii.gz
    combined_lungs_crop.nii.gz
    optional individual-mask crops
```

SimpleITK region-of-interest cropping preserves spacing and direction and updates the cropped-image origin.

### Limitations

- Padding may be clipped at image boundaries.
- Cropping reduces surrounding anatomical context.
- Full and cropped registration metrics describe different voxel domains.
- A correct bounding box still depends on a correct segmentation.

---

## ITK-SNAP preparation

### Purpose

This component creates geometry-matched labels for interactive review.

### MRI label mapping

| Value | Meaning |
|---:|---|
| 0 | Background |
| 1 | Left lung |
| 2 | Right lung |
| 10 | Lung bounding-box shell |

### Inputs

- Source image
- Left lung mask
- Right lung mask
- Bounding-box JSON
- Modality
- Case ID

### Outputs

```text
outputs/itksnap/<case_id>/
    lungs_multilabel.nii.gz
    bbox_label.nii.gz
    lungs_and_bbox_labels.nii.gz
    label_descriptions.txt
    launch_manifest.json
```

### Interpretation

- `lungs_multilabel.nii.gz` is the authoritative semantic segmentation.
- `bbox_label.nii.gz` is the authoritative six-face bounding-box shell.
- `lungs_and_bbox_labels.nii.gz` is a visualization convenience.
- Label 10 takes display priority where the shell and lung labels overlap.
- The combined display label should not be used for lung-volume measurement.

### Health check

```bash
python src/pipeline.py launch \
  --viewer itksnap \
  --health-check
```

### Limitations

- Automatic ITK-SNAP discovery currently targets macOS application bundles.
- CT five-lobe viewer labels are not yet configured in the unified workflow.

---

## Registration input preparation

### Purpose

Registration preparation creates explicit fixed and moving working images.

The moving image is transformed into the fixed image’s coordinate system.

### Required inputs

- Explicit fixed NIfTI
- Explicit moving NIfTI
- Non-identifying case ID

### Defaults

```text
Experiment: full
Output root: outputs/registration
Role-assignment source: manual
Isotropy tolerance: 1e-5
```

No casting, normalization, reorientation, or resampling is performed.

### Example

```bash
python src/prepare_registration_inputs.py \
  --fixed data/raw/registration_test/<case>/<fixed>.nii \
  --moving data/raw/registration_test/<case>/<moving>.nii \
  --case-id <case_id> \
  --experiment full \
  --role-assignment-source manual \
  --roles-confirmed \
  --dry-run
```

Remove `--dry-run` to create:

```text
outputs/registration/<case_id>/full/preprocessed/
    fixed.nii.gz
    moving.nii.gz
    preprocessing_manifest.json
```

### Adjustable parameters

- `--experiment`
- `--output-root`
- `--role-assignment-source`
- `--roles-confirmed`
- `--isotropy-tolerance`
- `--overwrite`

### Limitations

- Fixed and moving roles must be scientifically confirmed.
- Roles are not safely inferable from filenames alone.
- Mismatched voxel grids are rejected.
- deedsBCV does not properly account for arbitrary anisotropic spacing.

### Shared cropped registration preparation

The cropped experiment applies one fixed-image bounding box to both fixed and moving images. It does not calculate an independent moving-image crop.

Defaults:

```text
Source experiment: full
Destination experiment: cropped
Role source: inferred-from-repeated-bimg1
```

Example:

```bash
python src/prepare_shared_registration_crop.py \
  --fixed outputs/registration/<case_id>/full/preprocessed/fixed.nii.gz \
  --moving outputs/registration/<case_id>/full/preprocessed/moving.nii.gz \
  --bbox-json outputs/bbox/<fixed_case_id>/bbox.json \
  --case-id <case_id> \
  --source-experiment full \
  --experiment cropped \
  --dry-run
```

Output:

```text
outputs/registration/<case_id>/cropped/preprocessed/
    fixed.nii.gz
    moving.nii.gz
    preprocessing_manifest.json
    shared_crop_manifest.json
```

Limitations:

- Cropping may alter deformable-registration behavior.
- Reduced context may increase boundary effects.
- Cropped and full metrics should not be compared as if they cover identical domains.

---

## deedsBCV registration

### Purpose

deedsBCV performs:

1. Affine registration
2. Deformable registration

### Setup

Check the local environment:

```bash
scripts/setup_deedsbcv_macos.sh --check-only
```

Build with OpenMP where available:

```bash
scripts/setup_deedsbcv_macos.sh
```

Use the serial fallback:

```bash
scripts/setup_deedsbcv_macos.sh --without-openmp
```

Check the compiled tools:

```bash
python src/check_deedsbcv.py \
  --health-check \
  --json
```

### Defaults

```text
Experiment: full
Timeout: 3600 seconds per registration stage
```

The baseline registration uses upstream defaults by omission. It does not silently add optional optimization or segmentation parameters.

### Preview

```bash
python src/run_deedsbcv_registration.py \
  --prepared-dir outputs/registration/<case_id>/full/preprocessed \
  --case-id <case_id> \
  --experiment full \
  --bin-dir third_party/deedsBCV/build-arm64 \
  --dry-run
```

### Adjustable parameters

- `--experiment`
- `--bin-dir`
- `--timeout-seconds`
- `--resume`
- `--overwrite`

`--resume` and `--overwrite` are mutually exclusive.

### Outputs

```text
outputs/registration/<case_id>/<experiment>/
    preprocessed/
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

### Limitations

- Successful execution does not establish correct anatomy.
- Interpolation changes the registered-moving intensity distribution.
- Registration behavior depends on contrast, field of view, respiratory state, and acquisition consistency.
- Registration deformation is not automatically a physiological ventilation measurement.

---

## Registration quality control

### Purpose

Registration quality control compares:

```text
fixed versus original moving
fixed versus registered moving
```

### Metrics

- Mean squared error
- Mean absolute error
- Normalized cross-correlation
- Mutual information
- Optional mask Dice
- Registration runtime
- Represented volume
- deedsBCV SSD diagnostic when available

### Default

```text
Histogram bins: 64
```

### Example

```bash
python src/evaluate_registration.py \
  --case-id <case_id> \
  --experiment full \
  --experiment-dir outputs/registration/<case_id>/full \
  --fixed outputs/registration/<case_id>/full/preprocessed/fixed.nii.gz \
  --moving outputs/registration/<case_id>/full/preprocessed/moving.nii.gz \
  --registered outputs/registration/<case_id>/full/deformable/moving_registered.nii.gz \
  --registration-manifest outputs/registration/<case_id>/full/registration_manifest.json
```

### Outputs

```text
outputs/registration/<case_id>/<experiment>/qc/
    registration_qc.json
    before_overlay.png
    after_overlay.png
    before_checkerboard.png
    after_checkerboard.png
    before_difference.png
    after_difference.png
```

### Adjustable parameters

- Histogram bins
- Fixed, moving, and registered masks
- Output directory
- ITK-SNAP launch
- Dry-run launch
- Overwrite behavior

### Limitations

- Intensity metrics do not prove anatomical correctness.
- Interpolation can improve error metrics without improving anatomy.
- Dice requires valid corresponding masks.
- No metric is treated as a universal registration score.
- SSIM is intentionally omitted because a validated memory-efficient 3-D implementation is not provided.

### Detailed registration metric analysis

Defaults:

```text
Histogram bins: 100
Mutual-information bins: 64
Display percentile: 99
```

Example:

```bash
python src/analyze_registration_metrics.py \
  --fixed outputs/registration/<case_id>/full/preprocessed/fixed.nii.gz \
  --moving outputs/registration/<case_id>/full/preprocessed/moving.nii.gz \
  --registered outputs/registration/<case_id>/full/deformable/moving_registered.nii.gz \
  --lung-mask outputs/totalsegmentator/<fixed_case_id>/lungs/combined_lungs.nii.gz \
  --output-dir outputs/registration/<case_id>/full/metric_analysis \
  --dry-run \
  --json
```

Outputs include:

- Registration metric report
- Metric summary figure
- Error histograms
- Difference-statistics figures
- Whole-image and lung-domain summaries

---

## Cross-case difference comparison

### Purpose

This component compares Case 1 and Case 2 using a single pooled display window.

It computes:

```text
abs(fixed - original_moving)
abs(fixed - registered_moving)
```

### Defaults

```text
Pooled display percentile: 99
Overlay opacity: 0.45
Positive-magnitude colormap: turbo
```

### Generate

```bash
python src/compare_registration_differences.py \
  --dry-run \
  --json

python src/compare_registration_differences.py
```

### Adjustable parameters

- Case directories
- Output directory
- Display percentile
- Overlay opacity
- Overwrite behavior

### Outputs

```text
outputs/registration/case_comparison/equal_window_registration_differences/
    matched_window_grayscale_axial.png
    matched_window_turbo_axial.png
    matched_window_turbo_overlay_axial.png
    matched_window_turbo_multiplanar.png
    equal_window_registration_difference_report.json
```

### Limitations

- These are absolute-difference quality-control products.
- They are not the fractional MRI signal-change proxy.
- Difference magnitude can contain acquisition change, registration error, noise, and interpolation effects.
- `turbo` is used only for nonnegative magnitudes and should not be used for signed proxy maps.

---

## Registered intensity-change analysis

### Preliminary signed and absolute maps

Definitions:

```text
signed difference = fixed - registered
absolute difference = abs(fixed - registered)
```

Defaults:

```text
Experiment: full
Display percentile: 99
Overlay opacity: 0.45
Histogram bins: 100
```

Example:

```bash
python src/generate_intensity_change_map.py \
  --fixed outputs/registration/<case_id>/full/preprocessed/fixed.nii.gz \
  --registered outputs/registration/<case_id>/full/deformable/moving_registered.nii.gz \
  --lung-mask outputs/totalsegmentator/<fixed_case_id>/lungs/combined_lungs.nii.gz \
  --case-id <case_id> \
  --experiment full \
  --skip-normalized \
  --dry-run \
  --json
```

Adjustable parameters:

- Denominator policy
- Absolute denominator threshold
- Denominator percentile
- Display percentile
- Overlay opacity
- Histogram bins
- Figures-only regeneration
- Output root

Limitations:

- Signal difference does not directly measure ventilation.
- MRI signal may not be comparable across acquisitions.
- Registration interpolation affects the registered image.
- Analytical values outside the valid lung domain are NaN.

### Descriptive regional lung analysis

The historical functional-lung component describes spatial patterns without claiming validated function.

Defaults:

```text
Display percentile: 99
Overlay opacity: 0.45
Histogram bins: 100
```

Example:

```bash
python src/analyze_functional_lung.py \
  --fixed outputs/registration/<case_id>/full/preprocessed/fixed.nii.gz \
  --registered outputs/registration/<case_id>/full/deformable/moving_registered.nii.gz \
  --lung-mask outputs/totalsegmentator/<fixed_case_id>/lungs/combined_lungs.nii.gz \
  --left-lung outputs/totalsegmentator/<fixed_case_id>/lungs/lung_left.nii.gz \
  --right-lung outputs/totalsegmentator/<fixed_case_id>/lungs/lung_right.nii.gz \
  --output-dir outputs/registration/<case_id>/full/functional_lung \
  --dry-run \
  --json
```

Outputs include:

- Axial, coronal, and sagittal overlays
- Multislice figures
- Slice trends
- Histograms
- Regional statistics
- `functional_lung_report.json`

Limitations:

- Connected positive and negative regions have no automatic physiological interpretation.
- Left/right masks should be supplied for semantic anatomical summaries.
- Geometric physical-x partitions are not semantic lung segmentations.

### Historical normalized intensity maps

Two denominator policies are supported.

Absolute-threshold formula:

```text
(fixed - registered) / fixed
```

where:

```text
abs(fixed) >= threshold
```

Epsilon formula:

```text
(fixed - registered) / (fixed + epsilon)
```

Machine epsilon is:

```text
2.220446049250313e-16
```

Defaults:

```text
Denominator policy: must be explicitly selected
Display percentile: 99
Overlay opacity: 0.45
```

Machine epsilon prevents exact numerical division by zero but does not stabilize genuinely small MRI denominators. These historical maps use a different equation from the dedicated fractional MRI signal-change proxy.

---

## Fractional MRI signal-change proxy

### Purpose

The dedicated analysis calculates:

```text
P(x) = (M_R(x) - F(x)) / M_R(x)
```

where:

- `F` is the fixed MRI.
- `M_R` is the registered moving MRI.
- `P` is the exploratory fractional MRI signal-change proxy.

The original moving MRI is not used in the voxelwise formula because it is not in fixed-image anatomical correspondence.

### Interpretation

- `P > 0`: registered-moving signal is greater than fixed signal.
- `P < 0`: registered-moving signal is lower than fixed signal.
- `P = 0`: the two signals are equal.

The sign describes registered MRI signal change. It does not automatically describe physiological ventilation.

### Default denominator policy

A voxel is valid when:

```text
abs(M_R(x)) >=
0.01 × median(abs(M_R)) within the fixed-space lung mask
```

Default settings:

```text
Denominator-threshold fraction: 0.01
Display percentile: 99
Overlay opacity: 0.62
Invalid analytical voxels: NaN
Outside-lung analytical voxels: NaN
```

Machine epsilon is not used to hide unstable denominators.

### Adjustable parameters

- `--denominator-threshold`
- `--denominator-threshold-fraction`
- `--display-percentile`
- `--overlay-opacity`
- `--gaussian-sigma-mm`
- `--gaussian-truncate`
- `--gaussian-support-radius-mm`
- Case directories
- Lung-mask paths
- Prior proxy directory
- Output directory
- Overwrite behavior

Use an absolute denominator threshold only when supported by an MRI noise or calibration analysis:

```bash
python src/generate_fractional_signal_change_proxy.py \
  --denominator-threshold <CONFIRMED_POSITIVE_VALUE> \
  --output-dir outputs/registration/case_comparison/fractional_proxy_absolute_threshold
```

### Inputs

For each case:

- Fixed MRI
- Registered moving MRI
- Exact fixed-space combined-lung mask

Exact size, spacing, origin, and direction agreement are required.

### Outputs

```text
outputs/registration/case_comparison/
  fractional_mri_signal_change_proxy_gaussian_mm_comparison/
    case_1/
      native_fractional_proxy.nii.gz
      native_validity.nii.gz
      mean_filtered_fractional_proxy.nii.gz
      mean_filtered_validity.nii.gz
      gaussian_filtered_fractional_proxy.nii.gz
      gaussian_filtered_validity.nii.gz
    case_2/
      native_fractional_proxy.nii.gz
      native_validity.nii.gz
      mean_filtered_fractional_proxy.nii.gz
      mean_filtered_validity.nii.gz
      gaussian_filtered_fractional_proxy.nii.gz
      gaussian_filtered_validity.nii.gz
    fractional_proxy_axial_comparison.png
    fractional_proxy_multiplanar_comparison.png
    fractional_proxy_report.json
```

---

## Native, mean, and Gaussian comparison

### Native method

The formula is calculated directly at every valid lung voxel.

Advantages:

- Retains the greatest local variation.
- Does not impose spatial averaging.

Limitations:

- Most sensitive to voxel-scale noise.
- Produces the greatest number of extreme values.

### 7×7×7 box mean

The fixed and registered-moving images are filtered separately before calculating the ratio.

Every valid contributing voxel in the neighborhood receives equal weight.

```text
Maximum neighborhood: 7×7×7 voxels
Maximum voxel count: 343
```

Near the lung boundary or an invalid denominator, fewer than 343 voxels may contribute. The filter divides by the number of valid contributors, preventing outside-lung background from lowering the average.

Advantages:

- Strongly reduces voxel-scale variation.
- Simple and reproducible.

Limitations:

- Applies equal weight to near and distant neighbors.
- Produces the strongest spatial averaging of the three primary methods.
- Reduces effective spatial resolution.

### Gaussian-weighted filter

The fixed and registered-moving images are filtered separately using:

```text
filtered_I(x) =
Gaussian(I(x) × V(x)) / Gaussian(V(x))
```

where `V` is the analytical-validity mask.

The proxy is then calculated as:

```text
P_G(x) =
(filtered_M_R(x) - filtered_F(x)) / filtered_M_R(x)
```

Default Gaussian parameters:

```text
Sigma: 2.5 mm
Truncation: 1.5 sigma
```

For the current reference grid, these settings produce a 7×7×7 discrete support. This support is comparable to, but mathematically different from, the box mean.

Advantages:

- Nearby voxels receive more weight.
- Reduces voxel-scale variation.
- Retains slightly more local variation than the equal-weight box mean.

Limitations:

- Sigma is exploratory and not physiologically validated.
- Filtering reduces effective spatial resolution.
- Different sigma and truncation choices can change the result.

### Filtering order

The primary filtered experiments use:

```text
filter fixed image
filter registered-moving image
calculate ratio from filtered inputs
```

They do not use:

```text
calculate native ratio
smooth the ratio afterward
```

Post-ratio smoothing must be treated as a separate experiment if it is studied.

### Visualization policy

The comparison figures use:

```text
grayscale fixed MRI anatomy everywhere
+ signed proxy color only inside valid lung voxels
+ exact lung-mask contour
```

The absolute-difference image is not used as the anatomical background.

A signed, zero-centered `RdBu_r` colormap is used. `turbo` is not used because it does not naturally distinguish signed positive and negative values around zero.

All Case 1 and Case 2 native, mean, and Gaussian maps use one pooled symmetric display interval.

### Current reference-run values

The following are results from the current two-case dataset. They are not universal defaults and must be recalculated for new data.

```text
Voxel spacing:
approximately 1.29770994 mm isotropic

Mean support:
7×7×7 voxels
up to 343 valid contributors

Gaussian sigma:
2.5 mm

Gaussian truncation:
1.5 sigma

Gaussian voxel sigma:
approximately 1.92647 voxels per axis

Gaussian discrete support:
7×7×7 voxels

Voxel-cell physical support:
approximately 9.08397 mm per axis

First-to-last center span:
approximately 7.78626 mm per axis
```

Current denominator thresholds:

```text
Case 1: 9.073836059570313
Case 2: 10.60737060546875
```

Current six-map shared 99th-percentile display interval:

```text
[-0.6633644104003906, +0.6633644104003906]
```

Current valid lung count per case and method:

```text
522,794 voxels
```

These values should be read from `fractional_proxy_report.json` rather than hardcoded into new analyses.

---

## Horos DICOM export

### Purpose

The completed comparison can be exported into six independent, scrollable Horos-compatible RGB DICOM Secondary Capture series:

- Case 1 native
- Case 1 mean
- Case 1 Gaussian
- Case 2 native
- Case 2 mean
- Case 2 Gaussian

### Rendering design

Every slice uses:

```text
fixed MRI grayscale anatomy everywhere
+ proxy color inside valid exact-mask lung voxels
```

No signed- or absolute-difference image is loaded or used as the background.

### Defaults

```text
Overlay opacity: 0.62
Lung-mask contour: enabled
Colormap: RdBu_r
Display limits: read from fractional_proxy_report.json
Slice order: ascending fixed-image axial order
```

### Generate

```bash
python src/export_fractional_proxy_dicom.py \
  --dry-run \
  --json

python src/export_fractional_proxy_dicom.py
```

### Adjustable parameters

- `--comparison-report`
- `--output-dir`
- `--opacity`
- `--no-contour`
- `--overwrite`

### Outputs

```text
outputs/registration/case_comparison/
  fractional_mri_signal_change_proxy_gaussian_mm_comparison/
    horos_rgb_dicom_secondary_capture/
      case_1/
        native/
        mean/
        gaussian/
      case_2/
        native/
        mean/
        gaussian/
      fractional_proxy_rgb_secondary_capture_manifest.json
```

The current export contains 360 axial slices per series.

### Open Case 1 in Horos

```bash
open -a "Horos" \
  outputs/registration/case_comparison/fractional_mri_signal_change_proxy_gaussian_mm_comparison/horos_rgb_dicom_secondary_capture/case_1/native \
  outputs/registration/case_comparison/fractional_mri_signal_change_proxy_gaussian_mm_comparison/horos_rgb_dicom_secondary_capture/case_1/mean \
  outputs/registration/case_comparison/fractional_mri_signal_change_proxy_gaussian_mm_comparison/horos_rgb_dicom_secondary_capture/case_1/gaussian
```

### Open Case 2 in Horos

```bash
open -a "Horos" \
  outputs/registration/case_comparison/fractional_mri_signal_change_proxy_gaussian_mm_comparison/horos_rgb_dicom_secondary_capture/case_2/native \
  outputs/registration/case_comparison/fractional_mri_signal_change_proxy_gaussian_mm_comparison/horos_rgb_dicom_secondary_capture/case_2/mean \
  outputs/registration/case_comparison/fractional_mri_signal_change_proxy_gaussian_mm_comparison/horos_rgb_dicom_secondary_capture/case_2/gaussian
```

### DICOM limitations

These files are:

- Anonymized RGB Secondary Capture objects
- Irreversibly windowed visualization products
- Derived from NIfTI data
- Not original clinical DICOM objects
- Not quantitative DICOM Parametric Maps
- Not DICOM SEG objects
- Not RTSTRUCT objects
- Not clinical overlays

Quantitative proxy values cannot be recovered accurately from their RGB pixels. Use the analytical NIfTIs and JSON report for quantitative analysis.

---

## Presentation workflow

The presentation launcher opens completed artifacts without recomputing them.

List available stages:

```bash
./presentation_commands.sh help
```

Examples:

```bash
./presentation_commands.sh original
./presentation_commands.sh segmentation
./presentation_commands.sh bbox
./presentation_commands.sh full-registration
./presentation_commands.sh full-registration-qc
./presentation_commands.sh metric-analysis
./presentation_commands.sh functional-lung
./presentation_commands.sh normalized-map
./presentation_commands.sh case-comparison
./presentation_commands.sh professor-update
```

Rehearse without opening applications:

```bash
PRESENTATION_DRY_RUN=1 \
  ./presentation_commands.sh professor-update
```

Open all fractional-proxy PNG figures together:

```bash
open -a "Preview" \
  outputs/registration/case_comparison/fractional_mri_signal_change_proxy_gaussian_mm_comparison/*.png
```

Detailed presentation commands and speaking notes are available in:

```text
docs/registration_presentation_commands.md
```

---

## Inputs and outputs

### Private inputs

```text
data/raw/
```

### Main output structure

```text
outputs/
    nifti/                       working NIfTIs
    totalsegmentator/            semantic and combined lung masks
    figures/                     segmentation and bbox QC
    bbox/                        bounding-box JSON files
    crops/                       cropped images and masks
    itksnap/                     ITK-SNAP viewer products
    osirix/                      optional compatibility exports
    registration/
        <case_id>/
            full/
            cropped/
        case_comparison/
```

### Source structure

```text
config/                          default configuration
docs/                            architecture and presentation documentation
legacy/                          unsupported historical scripts
scripts/                         setup utilities
src/                             active implementation
tests/                           automated tests
third_party/deedsBCV/            pinned source provenance and patches
```

### Coordinate conventions

- SimpleITK indices and sizes use `x,y,z`.
- SimpleITK arrays exposed to NumPy use `z,y,x`.
- Physical coordinates are expressed in millimeters.
- Direction matrices are preserved.
- Bounding boxes use half-open NumPy intervals.

---

## Testing

Run the fractional-proxy tests:

```bash
pytest -q tests/test_fractional_signal_change_proxy.py
```

Run the Horos export tests:

```bash
pytest -q tests/test_fractional_proxy_rgb_secondary_capture.py
```

Run the complete test suite:

```bash
pytest -q
```

Compile Python sources:

```bash
python -m compileall -q src tests
```

Check patch formatting:

```bash
git diff --check
```

Before accepting analytical outputs, verify:

- Fixed and moving role assignments
- Source checksums
- Exact geometry
- Lung-mask confinement
- Invalid denominator handling
- NaN preservation
- Shared-window provenance
- Filter definition
- Slice locations
- DICOM readability
- DICOM identifier uniqueness
- Output overwrite protection
- Visual orientation and slice ordering

---

## Data privacy

Medical data must remain local.

Do not commit:

- Raw DICOM data
- NIfTI images
- NumPy arrays
- Segmentations
- Registration outputs
- Derived DICOM series
- Patient metadata
- Compiled third-party binaries

Reports should use non-identifying case IDs and should not include:

- Patient names
- Medical record numbers
- Dates of birth
- Accession numbers
- Unnecessary DICOM metadata
- Environment-variable dumps

---

## Recorded limitations

1. This software is research-only and non-diagnostic.
2. MRI intensities are not physically calibrated like CT Hounsfield units.
3. Fixed and moving roles require researcher confirmation.
4. Respiratory phase and scan ordering may be uncertain.
5. Registration metrics do not prove anatomical correctness.
6. Trilinear interpolation changes registered-moving intensities.
7. Registration error can appear as regional signal change.
8. Lung-mask errors propagate into every masked analysis.
9. The fractional-proxy denominator policy is exploratory.
10. Machine epsilon does not scientifically stabilize small MRI denominators.
11. Box and Gaussian filtering reduce effective spatial resolution.
12. Gaussian sigma and truncation are not physiologically validated.
13. The fractional MRI signal-change proxy has not been validated against physiological ground truth.
14. The current reference analysis contains too few cases for population-level inference.
15. Full and cropped registration metrics cover different domains.
16. RGB Secondary Capture DICOMs are visualization-only.
17. Quantitative values cannot be recovered from rendered RGB exports.
18. CT five-lobe ITK-SNAP integration is incomplete.
19. macOS is the best-tested viewer and registration-development environment.
20. Source DICOM without complete reference metadata cannot safely support RTSTRUCT export.

---

## Future directions

### Scientific validation

- Confirm fixed and moving acquisition roles.
- Confirm respiratory state and acquisition ordering.
- Record MRI sequence and reconstruction parameters.
- Validate the proxy against an independent physiological reference.
- Compare against spirometry, hyperpolarized-gas MRI, CT ventilation, or another appropriate reference.
- Evaluate scan-rescan repeatability.
- Expand the analysis to a larger cohort.

### Registration validation

- Add expert anatomical landmarks.
- Evaluate deformation-field Jacobians separately from intensity-derived proxies.
- Add inverse-consistency checks.
- Add deformation-topology checks.
- Compare alternative registration methods.
- Quantify the effects of cropping and interpolation.

### Signal-processing validation

- Perform denominator-threshold sensitivity analysis.
- Compare multiple Gaussian sigma values.
- Quantify effective spatial resolution.
- Evaluate post-ratio filtering only as a separately named experiment.
- Investigate edge-preserving filters.
- Add uncertainty and confidence maps.
- Quantify partial-volume effects near lung boundaries.

### MRI intensity standardization

- Measure background noise.
- Develop acquisition-informed denominator thresholds.
- Evaluate bias-field correction.
- Investigate scanner- and sequence-specific normalization.
- Determine whether intensities are comparable across sessions.

### Visualization and interoperability

- Consider quantitative DICOM Parametric Map export.
- Preserve analytical values separately from RGB products.
- Improve cross-platform viewer support.
- Complete CT five-lobe label support.
- Automate provenance-rich presentation reports.

### Reproducibility

- Add containerized environments.
- Add dataset-level manifests.
- Record software and model versions in every report.
- Add continuous integration for synthetic tests.
- Add standardized experiment configuration files.
- Add formal release tags for analytical definitions.

---

## Research-use-only disclaimer

This software and every generated output are intended exclusively for research validation.

They must not be used for:

- Diagnosis
- Treatment planning
- Clinical decision-making
- Patient monitoring
- Claims of validated regional ventilation
- Claims of validated perfusion
- Claims of clinical biomarker performance

Scientific interpretation requires independent validation, confirmed acquisition roles, appropriate statistical analysis, and expert review.
