# Registration presentation commands

Run these commands from any directory by invoking the script with its absolute path, or from the repository root as shown below. The launcher is read-only: it validates existing artifacts, opens them, and prints JSON; it never computes pipeline results. For rehearsal without GUI windows, prefix a GUI command with `PRESENTATION_DRY_RUN=1`.

Case 1 remains the backward-compatible default. Select Case 2 by inserting
`--case registration_case_2` before the stage name, for example:

```bash
./presentation_commands.sh --case registration_case_2 full-registration
PRESENTATION_DRY_RUN=1 ./presentation_commands.sh --case registration_case_2 normalized-map
```

All paths passed to ITK-SNAP are resolved from the repository location and are
therefore absolute. The script reads completed products only; it never runs
segmentation, registration, QC, or analysis code.

## Professor Update — New MRI Results

Command:

```bash
./presentation_commands.sh professor-update
```

This global, read-only mode presents only the newest MRI results from the most
recent professor meeting. It validates every required figure and report before
opening the first window, prints values directly from the existing reports, and
opens one presentation figure at a time in this exact order:

1. Case 2 post-registration QC overlay (`full/qc/after_overlay.png`)
2. Case 2 signed registered lung intensity difference
3. Case 2 absolute registered lung intensity difference
4. Case 2 anatomical/functional lung multi-slice overlay
5. Case 2 machine-epsilon normalized multi-slice map
6. Case 1 versus Case 2 matched-window grayscale axial differences
7. Case 1 versus Case 2 matched-window `turbo` axial differences
8. Case 1 versus Case 2 matched-window transparent `turbo` anatomy overlay
9. Case 1 versus Case 2 matched-window `turbo` multiplanar differences
10. Case 1 versus Case 2 lung-masked `turbo` axial differences
11. Case 1 versus Case 2 lung-masked `turbo` multiplanar differences
12. Case 1 versus Case 2 registration runtime comparison
13. Case 1 versus Case 2 registration-quality comparison
14. Case 1 versus Case 2 intensity-change comparison
15. Case 1 versus Case 2 normalized-intensity comparison

For the new equal-window figures, say:

> The original QC difference images shared a window within each case but not
> between cases. These new comparisons use one pooled 99th-percentile window
> for every Case 1 and Case 2 panel, allowing a fair visual comparison. The
> transparent turbo version retains the fixed anatomy for spatial context.

Every difference panel uses `abs(fixed - candidate)` and the exact shared
window `[0, 1005.4111328125]`. The transparent `turbo` anatomy overlay uses
opacity `0.45`.

For the professor-requested lung-masked composite, say:

> The absolute difference is calculated over the entire image. Inside the
> fixed-space lung mask, the difference is emphasized with the turbo colormap
> at 0.45 opacity. Outside the lung mask, the absolute difference remains
> visible in grayscale, preserving changes in the chest wall and other
> surrounding anatomy. Every Case 1 and Case 2 view uses the same pooled
> 99th-percentile window, so the displayed magnitudes are directly comparable.

The terminal summary parses full and cropped runtimes, runtime and voxel
reductions, signed and absolute means, signed positive/negative fractions,
normalized means, epsilon policy, and epsilon value from
`outputs/registration/case_comparison/case_1_vs_case_2.json`. The Case 2
normalized formula and epsilon are also read from its existing
`normalized_intensity_report.json`; they are not hardcoded. The presentation
retains the warning that normalized intensity change is exploratory and is not
validated ventilation. It does not select a universal registration winner, and
it states that full and cropped MSE/MAE values are from different spatial
domains.

For a rehearsal that validates and prints the exact open sequence without
launching Preview:

```bash
PRESENTATION_DRY_RUN=1 ./presentation_commands.sh professor-update
```

Backup-path command:

```bash
./presentation_commands.sh professor-update-backup
```

The backup command validates and prints absolute paths without opening them. It
includes the Case 2 pre-registration overlay, before/after checkerboards,
before/after difference images, metric summary, error histograms, functional
lung report, normalized intensity report, the Case 1-versus-Case 2 report, the
equal-window report, and the RGB Secondary Capture manifest. It is also safe to
run with `PRESENTATION_DRY_RUN=1`.

## Equal-window Case 1 versus Case 2 differences

Command:

```bash
./presentation_commands.sh equal-window-differences
```

This global read-only stage validates the completed equal-window report and
all four figures before opening them in this order:

1. `matched_window_grayscale_axial.png`
2. `matched_window_turbo_axial.png`
3. `matched_window_turbo_overlay_axial.png`
4. `matched_window_turbo_multiplanar.png`

The formula is `abs(fixed - candidate)`, where the candidate is the original
moving image before registration and the registered moving image afterward.
All panels use `[0, 1005.4111328125]`, calculated as the pooled 99th percentile
of all four Case 1/Case 2 before/after difference volumes. The transparent
`turbo` overlay has opacity `0.45` and retains grayscale fixed anatomy for
spatial context.

Rehearse without opening Preview. The command is global, so both case selectors
validate the same cross-case products:

```bash
PRESENTATION_DRY_RUN=1 ./presentation_commands.sh --case registration_case_1 equal-window-differences
PRESENTATION_DRY_RUN=1 ./presentation_commands.sh --case registration_case_2 equal-window-differences
```

## RGB DICOM Secondary Capture series

Command:

```bash
./presentation_commands.sh rgb-dicom-series
```

This validates the completed RGB export manifest and every expected DICOM file
for both 360-slice series, then reveals the Case 1 and Case 2 directories in
Finder for import into a DICOM viewer. It never regenerates DICOM or analytical
products. Rehearse both case-selecting forms without opening Finder:

```bash
PRESENTATION_DRY_RUN=1 ./presentation_commands.sh --case registration_case_1 rgb-dicom-series
PRESENTATION_DRY_RUN=1 ./presentation_commands.sh --case registration_case_2 rgb-dicom-series
```

The series contain derived 8-bit RGB Secondary Capture images for visual
review. They show grayscale fixed anatomy blended with `turbo`-colored
`abs(fixed - registered)` at opacity `0.45`, using the same shared window
`[0, 1005.4111328125]` for both cases. They are not quantitative DICOM
Parametric Maps, DICOM SEG objects, RTSTRUCTs, legacy DICOM Overlay Planes, or
clinical overlays; quantitative difference values cannot be recovered from
the RGB pixels.

Direct OsiriX MD import commands are:

```bash
open -a "OsiriX MD" outputs/registration/case_comparison/equal_window_registration_differences/rgb_dicom_secondary_capture/case_1
open -a "OsiriX MD" outputs/registration/case_comparison/equal_window_registration_differences/rgb_dicom_secondary_capture/case_2
```

## Lung-masked turbo differences

Generate only after a successful dry run:

```bash
python src/create_lung_masked_difference_visualization.py --dry-run --json
python src/create_lung_masked_difference_visualization.py
```

Present the axial and multiplanar figures, in that order:

```bash
./presentation_commands.sh lung-masked-differences
PRESENTATION_DRY_RUN=1 ./presentation_commands.sh --case registration_case_1 lung-masked-differences
PRESENTATION_DRY_RUN=1 ./presentation_commands.sh --case registration_case_2 lung-masked-differences
```

The background is grayscale `abs(fixed - candidate)` across the whole image,
not fixed anatomy. Inside the exact fixed-space combined-lung mask, the same
difference values are blended toward `turbo` at opacity `0.45`. Outside the
mask, grayscale difference remains visible. All components use the stored
shared limits `[0, 1005.4111328125]`.

## Lung-masked RGB DICOM Secondary Capture series

Generate only after a successful dry run:

```bash
python src/export_lung_masked_difference_dicom.py --dry-run --json
python src/export_lung_masked_difference_dicom.py
```

Validate and reveal both series for import:

```bash
./presentation_commands.sh lung-masked-dicom-series
PRESENTATION_DRY_RUN=1 ./presentation_commands.sh --case registration_case_1 lung-masked-dicom-series
PRESENTATION_DRY_RUN=1 ./presentation_commands.sh --case registration_case_2 lung-masked-dicom-series
```

Direct OsiriX MD import commands are:

```bash
open -a "OsiriX MD" outputs/registration/case_comparison/equal_window_registration_differences/lung_masked_turbo_difference/rgb_dicom_secondary_capture/case_1
open -a "OsiriX MD" outputs/registration/case_comparison/equal_window_registration_differences/lung_masked_turbo_difference/rgb_dicom_secondary_capture/case_2
```

In ITK-SNAP, choose **File → Open Main Image → DICOM series**, navigate to a
case directory above, and select the detected series. The RGB color is already
rendered and is not an adjustable quantitative `turbo` lookup table.

The series are derived 8-bit RGB Secondary Capture objects for visual review.
They are not quantitative Parametric Maps, DICOM SEG objects, RTSTRUCTs,
legacy Overlay Planes, or clinical overlays, and the underlying quantitative
difference values cannot be recovered from RGB pixels.

## Original

Command: `./presentation_commands.sh original`

Displays the full fixed scan, original moving scan, and registered moving scan as ITK-SNAP intensity layers. Say aloud: “This view establishes the fixed anatomy, the unregistered moving image, and the final registered result on the same full grid.” Expected files: `full/preprocessed/fixed.nii.gz`, `full/preprocessed/moving.nii.gz`, and `full/deformable/moving_registered.nii.gz`. Caution: fixed/moving role assignment was inferred and has not been researcher-confirmed.

## Segmentation

Command: `./presentation_commands.sh segmentation`

Displays the full fixed image with the verified left/right lung multilabel segmentation and its label descriptions. Say aloud: “These fixed-space lung labels define the anatomical region used by the downstream lung analyses.” Expected files: the full fixed image plus `outputs/itksnap/registration_case_1_fixed/lungs_multilabel.nii.gz` and `label_descriptions.txt`. Caution: segmentation quality is not equivalent to clinical validation.

## Bounding box

Command: `./presentation_commands.sh bbox`

Displays the fixed image with the combined lung and bounding-box label image. Say aloud: “The yellow shell shows the shared fixed-image crop applied to both registration inputs.” Expected files: the full fixed image, `lungs_and_bbox_labels.nii.gz`, and `label_descriptions.txt`. Caution: the bbox touches image boundaries, so requested padding may be clipped.

## Cropped inputs

Command: `./presentation_commands.sh cropped-inputs`

Displays the cropped fixed and moving inputs as intensity layers. Say aloud: “Both scans received the same geometry-preserving region-of-interest crop without resampling.” Expected files: `cropped/preprocessed/fixed.nii.gz` and `moving.nii.gz`. Caution: cropping changes the evaluation domain, so full and cropped metric values are not directly interchangeable.

## Full registration

Command: `./presentation_commands.sh full-registration`

Displays fixed, original moving, and registered moving images for the full experiment. Say aloud: “This is the full-volume registration result, with the pre-registration moving image retained for direct comparison.” Expected files: the three full-volume NIfTI files. Caution: visual agreement and intensity metrics do not by themselves establish anatomical correctness.

## Cropped registration

Command: `./presentation_commands.sh cropped-registration`

Displays fixed, original moving, and registered moving images for the cropped experiment. Say aloud: “This is the same comparison on the lung-focused cropped grid.” Expected files: the three cropped-experiment NIfTI files. Caution: boundary behavior and reduced context can affect deformable registration.

## Full registration QC

Command: `./presentation_commands.sh full-registration-qc`

Opens the six before/after overlay, checkerboard, and difference figures and prints `registration_qc.json`. Say aloud: “The full-volume QC compares alignment before and after registration using complementary visual checks.” Expected files: all six PNGs and `full/qc/registration_qc.json`. Caution: QC figures are research evidence, not diagnostic review.

## Cropped registration QC

Command: `./presentation_commands.sh cropped-registration-qc`

Opens the equivalent cropped QC figures and prints the cropped QC JSON. Say aloud: “These plots repeat the QC assessment within the cropped experiment’s smaller field of view.” Expected files: all six PNGs and `cropped/qc/registration_qc.json`. Caution: the cropped domain differs from the full domain.

## Comparison

Command: `./presentation_commands.sh comparison`

Opens the full-versus-cropped comparison figure and prints its JSON. Say aloud: “Cropping reduced the computation domain and runtime while both experiments improved their within-domain similarity metrics.” Expected files: `comparison/registration_comparison.png` and `.json`. Caution: cross-domain metric magnitudes should not be treated as a controlled anatomical winner.

## Metric analysis

Command: `./presentation_commands.sh metric-analysis`

Opens the metric summary, error histograms, and difference-statistics plots and prints the metric report. Say aloud: “Multiple metrics are shown because each captures a different aspect of registration behavior.” Expected files: three PNGs and `full/metric_analysis/registration_metric_report.json`. Caution: interpolation-induced smoothing can improve intensity errors without guaranteeing better anatomy.

## Signed difference

Command: `./presentation_commands.sh signed-difference`

Displays the signed-difference map over fixed anatomy with the combined lung mask as segmentation. Say aloud: “The sign indicates whether fixed intensity is above or below registered moving intensity at each lung voxel.” Expected files: fixed anatomy, `signed_difference_display.nii.gz`, and `combined_lungs.nii.gz`. Caution: this is not a validated ventilation map.

## Absolute difference

Command: `./presentation_commands.sh absolute-difference`

Displays the absolute-difference map over fixed anatomy with the lung mask. Say aloud: “This map removes direction and emphasizes the magnitude of local registered intensity differences.” Expected files: fixed anatomy, `absolute_difference_display.nii.gz`, and `combined_lungs.nii.gz`. Caution: magnitude can include registration error and interpolation effects.

## Intensity-change figures

Command: `./presentation_commands.sh intensity-change-figures`

Opens both overlays, the signed contact sheet, and histogram, then prints the manifest. Say aloud: “These figures summarize the spatial distribution and whole-lung distribution of signed and absolute differences.” Expected files: four named PNGs and `intensity_change_manifest.json`. Caution: respiratory phase, modality, calibration, and intensity comparability remain unresolved.

## Functional lung

Command: `./presentation_commands.sh functional-lung`

Opens orthogonal overlays, the multi-slice panel, slice summaries, and lung histograms, then prints the report. Say aloud: “This exploratory regional analysis describes spatial intensity-change patterns without assigning physiological meaning.” Expected files: seven named PNGs and `functional_lung_report.json`. Caution: the output is explicitly not a validated functional or ventilation biomarker.

## Normalized map

Command: `./presentation_commands.sh normalized-map`

If completed outputs exist, displays the machine-epsilon normalized map over
fixed anatomy, opens every normalized overlay PNG, and prints the report. It
reads from `full/normalized_intensity_epsilon_machine/`. Say aloud: “This is the
exploratory registered intensity-change ratio `(fixed - registered) / (fixed +
epsilon)` using the same float64 machine epsilon in both cases.” Expected files:
`normalized_difference_display.nii.gz`, normalized overlay PNGs,
`normalized_intensity_report.json`, fixed anatomy, and the lung mask. Caution:
machine epsilon provides exact-zero numerical protection but does not
scientifically stabilize genuinely small denominators; this is not validated
ventilation.

## Case comparison

Command: `./presentation_commands.sh case-comparison`

Opens the existing Case 1-versus-Case 2 runtime, registration-quality,
intensity-change, functional-lung, and compatible machine-epsilon normalized
comparison figures, then prints `case_1_vs_case_2.json`. This stage is global;
`--case` does not change its inputs. The launcher uses absolute paths and never
recomputes the comparison or any upstream analysis. Caution: the results are
descriptive, full and cropped metrics use different domains, and no biological,
clinical, or ventilation winner is assigned.

## Tests

Command: `./presentation_commands.sh tests`

Prints focused test commands without running them. Say aloud: “These are the focused verification targets for the viewer and completed analysis components.” Expected files: the listed test modules. Caution: presentation mode deliberately does not run the full suite.

## Results summary

Command: `./presentation_commands.sh results-summary`

Prints verified runtimes, reductions, before/after metrics, whole-lung difference summaries, and unresolved assumptions from existing JSON. Say aloud: “This summary reports the recorded results while keeping the scientific limitations visible.” Expected files: comparison, registration, intensity-change, and functional-lung JSON reports. Caution: results are research-only, role assignment is unconfirmed, and the intensity changes are not physiological measurements.

## All

Command: `./presentation_commands.sh all`

Prints the recommended presentation sequence without opening any GUI windows. Say aloud: “I will step through the artifacts one stage at a time to keep the comparison legible.” Expected files: none are opened by this command. Caution: use each printed command separately rather than launching all windows together.

## Help

Command: `./presentation_commands.sh help`

Prints the supported stage names and dry-run option. Say aloud: “The launcher exposes only read-only presentation and inspection actions.” Expected files: only `presentation_commands.sh`. Caution: `PRESENTATION_DRY_RUN=1` validates GUI command construction but does not test application rendering.
