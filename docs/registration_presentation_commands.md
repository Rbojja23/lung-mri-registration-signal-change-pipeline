# Case 1 registration presentation commands

Run these commands from any directory by invoking the script with its absolute path, or from the repository root as shown below. The launcher is read-only: it validates existing artifacts, opens them, and prints JSON; it never computes pipeline results. For rehearsal without GUI windows, prefix a GUI command with `PRESENTATION_DRY_RUN=1`.

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

If completed outputs exist, displays the normalized map over fixed anatomy, opens every normalized overlay PNG, and prints the report; otherwise it prints the verified unresolved-threshold message. Say aloud: “The normalized implementation exists, but Case 1 was intentionally not generated until a defensible denominator threshold is chosen.” Expected files when available: `normalized_difference_display.nii.gz`, normalized overlay PNGs, `normalized_intensity_report.json`, fixed anatomy, and the lung mask. Caution: near-zero denominators can amplify noise, and no threshold has been scientifically resolved for Case 1.

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
