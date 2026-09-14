#!/usr/bin/env bash
set -euo pipefail

# Read-only presentation launcher for completed registration-case artifacts.
# Set PRESENTATION_DRY_RUN=1 to validate and print GUI commands without opening them.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_ID="${PRESENTATION_CASE:-registration_case_1}"
if [[ "${1:-}" == "--case" ]]; then
  if [[ -z "${2:-}" ]]; then
    printf '%s\n' 'Error: --case requires a case ID.' >&2
    exit 2
  fi
  CASE_ID="$2"
  shift 2
fi
if [[ ! "$CASE_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'Error: invalid case ID: %s\n' "$CASE_ID" >&2
  exit 2
fi
CASE_DIR="$ROOT_DIR/outputs/registration/$CASE_ID"
FULL_DIR="$CASE_DIR/full"
CROPPED_DIR="$CASE_DIR/cropped"
FIXED_CASE_ID="${CASE_ID}_fixed"
FIXED_LABEL_DIR="$ROOT_DIR/outputs/itksnap/$FIXED_CASE_ID"
LUNG_MASK="$ROOT_DIR/outputs/totalsegmentator/$FIXED_CASE_ID/lungs/combined_lungs.nii.gz"
CASE_COMPARISON_DIR="$ROOT_DIR/outputs/registration/case_comparison"
EQUAL_WINDOW_DIR="$CASE_COMPARISON_DIR/equal_window_registration_differences"
EQUAL_WINDOW_REPORT="$EQUAL_WINDOW_DIR/equal_window_registration_difference_report.json"
RGB_DICOM_DIR="$EQUAL_WINDOW_DIR/rgb_dicom_secondary_capture"
RGB_DICOM_REPORT="$RGB_DICOM_DIR/rgb_secondary_capture_manifest.json"
LUNG_MASKED_DIR="$EQUAL_WINDOW_DIR/lung_masked_turbo_difference"
LUNG_MASKED_REPORT="$LUNG_MASKED_DIR/lung_masked_turbo_difference_report.json"
LUNG_MASKED_DICOM_DIR="$LUNG_MASKED_DIR/rgb_dicom_secondary_capture"
LUNG_MASKED_DICOM_REPORT="$LUNG_MASKED_DICOM_DIR/lung_masked_rgb_secondary_capture_manifest.json"
DRY_RUN="${PRESENTATION_DRY_RUN:-0}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf 'Error: required existing file is missing: %s\n' "$path" >&2
    return 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    printf 'Error: required existing directory is missing: %s\n' "$path" >&2
    return 1
  fi
}

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    printf 'Error: required command is unavailable: %s\n' "$name" >&2
    return 1
  fi
}

print_command() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

launch_itksnap() {
  local image="$1"
  shift
  require_file "$image"
  local path
  for path in "$@"; do
    [[ "$path" == "-o" || "$path" == "-s" || "$path" == "-l" ]] || require_file "$path"
  done
  local command=(open -a "ITK-SNAP" --args -g "$image" "$@")
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${command[@]}"
  else
    require_command open
    "${command[@]}"
  fi
}

open_files() {
  local path
  for path in "$@"; do require_file "$path"; done
  local command=(open -a Preview "$@")
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${command[@]}"
  else
    require_command open
    "${command[@]}"
  fi
}

reveal_directories() {
  local path
  for path in "$@"; do
    require_dir "$path"
    local command=(open -R "$path")
    if [[ "$DRY_RUN" == "1" ]]; then
      print_command "${command[@]}"
    else
      require_command open
      "${command[@]}"
    fi
  done
}

display_json() {
  local path="$1"
  require_file "$path"
  require_command jq
  printf '\n%s\n' "--- $path ---"
  jq . "$path"
}

stage_original() {
  launch_itksnap \
    "$FULL_DIR/preprocessed/fixed.nii.gz" \
    -o "$FULL_DIR/preprocessed/moving.nii.gz" \
       "$FULL_DIR/deformable/moving_registered.nii.gz"
}

stage_segmentation() {
  launch_itksnap \
    "$FULL_DIR/preprocessed/fixed.nii.gz" \
    -s "$FIXED_LABEL_DIR/lungs_multilabel.nii.gz" \
    -l "$FIXED_LABEL_DIR/label_descriptions.txt"
}

stage_bbox() {
  launch_itksnap \
    "$FULL_DIR/preprocessed/fixed.nii.gz" \
    -s "$FIXED_LABEL_DIR/lungs_and_bbox_labels.nii.gz" \
    -l "$FIXED_LABEL_DIR/label_descriptions.txt"
}

stage_cropped_inputs() {
  launch_itksnap \
    "$CROPPED_DIR/preprocessed/fixed.nii.gz" \
    -o "$CROPPED_DIR/preprocessed/moving.nii.gz"
}

stage_full_registration() {
  launch_itksnap \
    "$FULL_DIR/preprocessed/fixed.nii.gz" \
    -o "$FULL_DIR/preprocessed/moving.nii.gz" \
       "$FULL_DIR/deformable/moving_registered.nii.gz"
}

stage_cropped_registration() {
  launch_itksnap \
    "$CROPPED_DIR/preprocessed/fixed.nii.gz" \
    -o "$CROPPED_DIR/preprocessed/moving.nii.gz" \
       "$CROPPED_DIR/deformable/moving_registered.nii.gz"
}

stage_qc() {
  local experiment="$1"
  local qc_dir="$CASE_DIR/$experiment/qc"
  open_files \
    "$qc_dir/before_overlay.png" "$qc_dir/after_overlay.png" \
    "$qc_dir/before_checkerboard.png" "$qc_dir/after_checkerboard.png" \
    "$qc_dir/difference_before.png" "$qc_dir/difference_after.png"
  display_json "$qc_dir/registration_qc.json"
}

stage_comparison() {
  open_files "$CASE_DIR/comparison/registration_comparison.png"
  display_json "$CASE_DIR/comparison/registration_comparison.json"
}

stage_metric_analysis() {
  local dir="$FULL_DIR/metric_analysis"
  open_files "$dir/metric_summary.png" "$dir/error_histograms.png" "$dir/difference_statistics.png"
  display_json "$dir/registration_metric_report.json"
}

stage_signed_difference() {
  launch_itksnap "$FULL_DIR/preprocessed/fixed.nii.gz" \
    -o "$FULL_DIR/intensity_change/signed_difference_display.nii.gz" \
    -s "$LUNG_MASK"
}

stage_absolute_difference() {
  launch_itksnap "$FULL_DIR/preprocessed/fixed.nii.gz" \
    -o "$FULL_DIR/intensity_change/absolute_difference_display.nii.gz" \
    -s "$LUNG_MASK"
}

stage_intensity_change_figures() {
  local dir="$FULL_DIR/intensity_change"
  open_files "$dir/signed_difference_overlay.png" "$dir/absolute_difference_overlay.png" \
    "$dir/signed_difference_contact_sheet.png" "$dir/intensity_change_histogram.png"
  display_json "$dir/intensity_change_manifest.json"
}

stage_functional_lung() {
  local dir="$FULL_DIR/functional_lung"
  open_files "$dir/axial_overlay.png" "$dir/coronal_overlay.png" "$dir/sagittal_overlay.png" \
    "$dir/multi_slice_panel.png" "$dir/slice_mean.png" "$dir/slice_std.png" "$dir/lung_histograms.png"
  display_json "$dir/functional_lung_report.json"
}

stage_normalized_map() {
  local dir="$FULL_DIR/normalized_intensity_epsilon_machine"
  local display="$dir/normalized_difference_display.nii.gz"
  local report="$dir/normalized_intensity_report.json"
  if [[ ! -f "$display" || ! -f "$report" ]]; then
    printf 'Machine-epsilon normalized-map outputs have not been generated for %s.\n' "$CASE_ID"
    return 0
  fi
  local overlays=("$dir"/normalized_*.png)
  local existing=()
  local path
  for path in "${overlays[@]}"; do [[ -f "$path" ]] && existing+=("$path"); done
  if (( ${#existing[@]} == 0 )); then
    printf 'Error: normalized NIfTI/report exist, but no normalized overlay PNGs were found in: %s\n' "$dir" >&2
    return 1
  fi
  launch_itksnap "$FULL_DIR/preprocessed/fixed.nii.gz" -o "$display" -s "$LUNG_MASK"
  open_files "${existing[@]}"
  display_json "$report"
}

stage_case_comparison() {
  local figures=(
    "$CASE_COMPARISON_DIR/registration_runtime_comparison.png"
    "$CASE_COMPARISON_DIR/registration_quality_comparison.png"
    "$CASE_COMPARISON_DIR/intensity_change_comparison.png"
    "$CASE_COMPARISON_DIR/functional_lung_comparison.png"
  )
  if [[ -f "$CASE_COMPARISON_DIR/normalized_intensity_comparison.png" ]]; then
    figures+=("$CASE_COMPARISON_DIR/normalized_intensity_comparison.png")
  fi
  open_files "${figures[@]}"
  display_json "$CASE_COMPARISON_DIR/case_1_vs_case_2.json"
}

require_complete_report() {
  local path="$1"
  require_file "$path"
  if ! jq -e '.completion_status == "complete"' "$path" >/dev/null; then
    printf 'Error: report is not marked complete: %s\n' "$path" >&2
    return 1
  fi
}

validate_equal_window_report() {
  require_command jq
  require_complete_report "$EQUAL_WINDOW_REPORT"
  if ! jq -e '
    .analysis_type == "Case 1-versus-Case 2 equal-window registration difference visualization" and
    .formulas.before == "abs(fixed - moving)" and
    .formulas.after == "abs(fixed - registered)" and
    .shared_difference_window.percentile == 99 and
    .shared_difference_window.vmin == 0 and
    .shared_difference_window.vmax == 1005.4111328125 and
    .shared_difference_window.identical_for_every_difference_panel == true and
    .display.overlay_opacity == 0.45
  ' "$EQUAL_WINDOW_REPORT" >/dev/null; then
    printf 'Error: equal-window report is incompatible with the professor presentation.\n' >&2
    return 1
  fi
}

validate_rgb_dicom_series() {
  local case count index filename
  require_command jq
  require_complete_report "$RGB_DICOM_REPORT"
  if ! jq -e '
    .object_classification.dicom_sop_class == "Secondary Capture Image Storage" and
    .object_classification.visualization_only == true and
    .object_classification.is_parametric_map == false and
    .object_classification.is_legacy_overlay_plane == false and
    .formula == "abs(fixed - registered)" and
    .display.difference_colormap == "turbo" and
    .display.opacity == 0.45 and
    .display.shared_difference_limits.vmin == 0 and
    .display.shared_difference_limits.vmax == 1005.4111328125 and
    .display.identical_map_normalization_across_cases == true and
    .validation.status == "passed"
  ' "$RGB_DICOM_REPORT" >/dev/null; then
    printf 'Error: RGB Secondary Capture manifest is incomplete or incompatible.\n' >&2
    return 1
  fi
  for case in case_1 case_2; do
    require_dir "$RGB_DICOM_DIR/$case"
    count="$(jq -er ".slice_count.${case}" "$RGB_DICOM_REPORT")"
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
      printf 'Error: invalid RGB DICOM slice count for %s: %s\n' "$case" "$count" >&2
      return 1
    fi
    for ((index = 1; index <= count; index++)); do
      printf -v filename 'IM_%04d.dcm' "$index"
      require_file "$RGB_DICOM_DIR/$case/$filename"
    done
  done
}

validate_lung_masked_report() {
  require_command jq
  require_complete_report "$LUNG_MASKED_REPORT"
  if ! jq -e '
    .analysis_type == "Case 1-versus-Case 2 lung-masked turbo difference visualization" and
    .formulas.before == "abs(fixed - moving)" and
    .formulas.after == "abs(fixed - registered)" and
    .shared_difference_window.vmin == 0 and
    .shared_difference_window.vmax == 1005.4111328125 and
    .shared_difference_window.identical_for_every_panel_and_component == true and
    .display.opacity == 0.45 and
    .display.fixed_anatomy_used_as_background == false and
    .display.lung_mask_restricts_analysis == false and
    .display.inside_mask_colormap == "turbo"
  ' "$LUNG_MASKED_REPORT" >/dev/null; then
    printf 'Error: lung-masked turbo report is incomplete or incompatible.\n' >&2
    return 1
  fi
}

validate_lung_masked_dicom_series() {
  local case count index filename
  require_command jq
  require_complete_report "$LUNG_MASKED_DICOM_REPORT"
  if ! jq -e '
    .object_classification.dicom_sop_class == "Secondary Capture Image Storage" and
    .object_classification.visualization_only == true and
    .object_classification.is_parametric_map == false and
    .object_classification.is_legacy_overlay_plane == false and
    .formula == "abs(fixed - registered)" and
    .display.difference_colormap_inside_mask == "turbo" and
    .display.opacity == 0.45 and
    .display.shared_difference_limits.vmin == 0 and
    .display.shared_difference_limits.vmax == 1005.4111328125 and
    .display.fixed_anatomy_used_as_background == false and
    .display.lung_mask_restricts_analysis == false and
    .display.identical_map_normalization_across_cases == true and
    .validation.status == "passed"
  ' "$LUNG_MASKED_DICOM_REPORT" >/dev/null; then
    printf 'Error: lung-masked RGB Secondary Capture manifest is incomplete or incompatible.\n' >&2
    return 1
  fi
  for case in case_1 case_2; do
    require_dir "$LUNG_MASKED_DICOM_DIR/$case"
    count="$(jq -er ".slice_count.${case}" "$LUNG_MASKED_DICOM_REPORT")"
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
      printf 'Error: invalid lung-masked DICOM slice count for %s: %s\n' "$case" "$count" >&2
      return 1
    fi
    for ((index = 1; index <= count; index++)); do
      printf -v filename 'IM_%04d.dcm' "$index"
      require_file "$LUNG_MASKED_DICOM_DIR/$case/$filename"
    done
  done
}

print_equal_window_explanation() {
  printf '%s\n' 'The original QC difference images shared a window within each case but not between cases. These new comparisons use one pooled 99th-percentile window for every Case 1 and Case 2 panel, allowing a fair visual comparison. The transparent turbo version retains the fixed anatomy for spatial context.'
  printf '%s\n' 'Formula: abs(fixed - candidate)'
  printf '%s\n' 'Shared window: [0, 1005.4111328125]'
  printf '%s\n' 'Transparent turbo opacity: 0.45'
}

stage_equal_window_differences() {
  local figures=(
    "$EQUAL_WINDOW_DIR/matched_window_grayscale_axial.png"
    "$EQUAL_WINDOW_DIR/matched_window_turbo_axial.png"
    "$EQUAL_WINDOW_DIR/matched_window_turbo_overlay_axial.png"
    "$EQUAL_WINDOW_DIR/matched_window_turbo_multiplanar.png"
  )
  local path
  validate_equal_window_report
  for path in "${figures[@]}"; do require_file "$path"; done
  print_equal_window_explanation
  open_files "${figures[@]}"
  display_json "$EQUAL_WINDOW_REPORT"
}

stage_rgb_dicom_series() {
  validate_rgb_dicom_series
  printf '%s\n' 'Derived RGB DICOM Secondary Capture for visual review only; not a quantitative Parametric Map or clinical overlay.'
  printf 'Case 1 DICOM import directory: %s\n' "$RGB_DICOM_DIR/case_1"
  printf 'Case 2 DICOM import directory: %s\n' "$RGB_DICOM_DIR/case_2"
  reveal_directories "$RGB_DICOM_DIR/case_1" "$RGB_DICOM_DIR/case_2"
}

print_lung_masked_explanation() {
  printf '%s\n' 'The absolute difference is calculated over the entire image. Inside the fixed-space lung mask, the difference is emphasized with the turbo colormap at 0.45 opacity. Outside the lung mask, the absolute difference remains visible in grayscale, preserving changes in the chest wall and other surrounding anatomy. Every Case 1 and Case 2 view uses the same pooled 99th-percentile window, so the displayed magnitudes are directly comparable.'
  printf '%s\n' 'Background: grayscale abs(fixed - candidate), not fixed anatomy'
  printf '%s\n' 'Shared window: [0, 1005.4111328125]'
}

stage_lung_masked_differences() {
  local figures=(
    "$LUNG_MASKED_DIR/lung_masked_turbo_axial_comparison.png"
    "$LUNG_MASKED_DIR/lung_masked_turbo_multiplanar_comparison.png"
  )
  local path
  validate_lung_masked_report
  for path in "${figures[@]}"; do require_file "$path"; done
  print_lung_masked_explanation
  open_files "${figures[@]}"
  display_json "$LUNG_MASKED_REPORT"
}

stage_lung_masked_dicom_series() {
  validate_lung_masked_dicom_series
  printf '%s\n' 'Derived RGB Secondary Capture: grayscale difference outside lungs and masked turbo difference inside; visualization only.'
  printf 'Case 1 DICOM import directory: %s\n' "$LUNG_MASKED_DICOM_DIR/case_1"
  printf 'Case 2 DICOM import directory: %s\n' "$LUNG_MASKED_DICOM_DIR/case_2"
  reveal_directories "$LUNG_MASKED_DICOM_DIR/case_1" "$LUNG_MASKED_DICOM_DIR/case_2"
}

print_professor_update_summary() {
  local comparison_report="$CASE_COMPARISON_DIR/case_1_vs_case_2.json"
  jq -r '
    "NEW RESULTS\n\n" +
    "Case 2:\n" +
    "- registration complete\n" +
    "- signed/absolute intensity mapping complete\n" +
    "- functional lung mapping complete\n" +
    "- epsilon normalization complete\n\n" +
    "Cross-case:\n" +
    "- Case 1 full runtime: \(.registration.case_1.full.runtime_seconds.total) seconds\n" +
    "- Case 2 full runtime: \(.registration.case_2.full.runtime_seconds.total) seconds\n" +
    "- Case 1 cropped runtime: \(.registration.case_1.cropped.runtime_seconds.total) seconds\n" +
    "- Case 2 cropped runtime: \(.registration.case_2.cropped.runtime_seconds.total) seconds\n" +
    "- Case 1 signed mean: \(.intensity_change_comparison.cases.case_1.signed_whole_lung.mean)\n" +
    "- Case 2 signed mean: \(.intensity_change_comparison.cases.case_2.signed_whole_lung.mean)\n" +
    "- Case 1 normalized mean: \(.epsilon_normalized_comparison.cases.case_1.whole_lung.mean)\n" +
    "- Case 2 normalized mean: \(.epsilon_normalized_comparison.cases.case_2.whole_lung.mean)"
  ' "$comparison_report"
}

print_runtime_details() {
  jq -r '
    "Case 1 full runtime: \(.registration.case_1.full.runtime_seconds.total) seconds\n" +
    "Case 1 cropped runtime: \(.registration.case_1.cropped.runtime_seconds.total) seconds\n" +
    "Case 1 runtime reduction: \(.registration.case_1.derived.runtime_reduction_percent)%\n" +
    "Case 1 voxel reduction: \(.registration.case_1.derived.voxel_reduction_percent)%\n" +
    "Case 2 full runtime: \(.registration.case_2.full.runtime_seconds.total) seconds\n" +
    "Case 2 cropped runtime: \(.registration.case_2.cropped.runtime_seconds.total) seconds\n" +
    "Case 2 runtime reduction: \(.registration.case_2.derived.runtime_reduction_percent)%\n" +
    "Case 2 voxel reduction: \(.registration.case_2.derived.voxel_reduction_percent)%"
  ' "$CASE_COMPARISON_DIR/case_1_vs_case_2.json"
}

print_intensity_details() {
  jq -r '
    "Case 1 signed mean: \(.intensity_change_comparison.cases.case_1.signed_whole_lung.mean)\n" +
    "Case 2 signed mean: \(.intensity_change_comparison.cases.case_2.signed_whole_lung.mean)\n" +
    "Case 1 absolute mean: \(.intensity_change_comparison.cases.case_1.absolute_whole_lung.mean)\n" +
    "Case 2 absolute mean: \(.intensity_change_comparison.cases.case_2.absolute_whole_lung.mean)\n" +
    "Case 1 positive/negative fractions: \(.intensity_change_comparison.cases.case_1.signed_whole_lung.positive_fraction) / \(.intensity_change_comparison.cases.case_1.signed_whole_lung.negative_fraction)\n" +
    "Case 2 positive/negative fractions: \(.intensity_change_comparison.cases.case_2.signed_whole_lung.positive_fraction) / \(.intensity_change_comparison.cases.case_2.signed_whole_lung.negative_fraction)"
  ' "$CASE_COMPARISON_DIR/case_1_vs_case_2.json"
}

print_normalized_details() {
  local comparison_report="$CASE_COMPARISON_DIR/case_1_vs_case_2.json"
  local epsilon_value
  epsilon_value="$(jq -er '.epsilon_normalized_comparison.validated_epsilon' "$comparison_report")"
  jq -r '
    "Case 1 normalized mean: \(.epsilon_normalized_comparison.cases.case_1.whole_lung.mean)\n" +
    "Case 2 normalized mean: \(.epsilon_normalized_comparison.cases.case_2.whole_lung.mean)\n" +
    "Epsilon policy: \(.epsilon_normalized_comparison.cases.case_1.epsilon_source)"
  ' "$comparison_report"
  printf 'Epsilon value: %.15e\n' "$epsilon_value"
  printf '%s\n' 'Exploratory normalized registered intensity-change comparison; not validated ventilation.'
}

present_figure() {
  local number="$1"
  local label="$2"
  local path="$3"
  printf '\n%s. %s\nFigure: %s\n' "$number" "$label" "$path"
  open_files "$path"
}

stage_professor_update() {
  local case_2_full="$ROOT_DIR/outputs/registration/registration_case_2/full"
  local qc="$case_2_full/qc"
  local intensity="$case_2_full/intensity_change"
  local functional="$case_2_full/functional_lung"
  local normalized="$case_2_full/normalized_intensity_epsilon_machine"
  local comparison_report="$CASE_COMPARISON_DIR/case_1_vs_case_2.json"
  local required_figures=(
    "$qc/after_overlay.png"
    "$intensity/signed_difference_overlay.png"
    "$intensity/absolute_difference_overlay.png"
    "$functional/multi_slice_panel.png"
    "$normalized/normalized_multi_slice.png"
    "$EQUAL_WINDOW_DIR/matched_window_grayscale_axial.png"
    "$EQUAL_WINDOW_DIR/matched_window_turbo_axial.png"
    "$EQUAL_WINDOW_DIR/matched_window_turbo_overlay_axial.png"
    "$EQUAL_WINDOW_DIR/matched_window_turbo_multiplanar.png"
    "$LUNG_MASKED_DIR/lung_masked_turbo_axial_comparison.png"
    "$LUNG_MASKED_DIR/lung_masked_turbo_multiplanar_comparison.png"
    "$CASE_COMPARISON_DIR/registration_runtime_comparison.png"
    "$CASE_COMPARISON_DIR/registration_quality_comparison.png"
    "$CASE_COMPARISON_DIR/intensity_change_comparison.png"
    "$CASE_COMPARISON_DIR/normalized_intensity_comparison.png"
  )
  local path

  require_command jq
  if [[ "$DRY_RUN" != "1" ]]; then require_command open; fi
  for path in "${required_figures[@]}"; do require_file "$path"; done
  require_file "$qc/before_overlay.png"
  require_file "$qc/difference_before.png"
  require_file "$qc/difference_after.png"
  require_complete_report "$qc/registration_qc.json"
  require_complete_report "$intensity/intensity_change_manifest.json"
  require_complete_report "$functional/functional_lung_report.json"
  require_complete_report "$normalized/normalized_intensity_report.json"
  require_complete_report "$comparison_report"
  validate_equal_window_report
  validate_rgb_dicom_series
  validate_lung_masked_report
  validate_lung_masked_dicom_series

  print_professor_update_summary

  present_figure 1 "Case 2 registration result" "${required_figures[0]}"
  printf 'Backup overlay before: %s\n' "$qc/before_overlay.png"
  printf 'Backup difference before: %s\n' "$qc/difference_before.png"
  printf 'Backup difference after: %s\n' "$qc/difference_after.png"

  present_figure 2 "Case 2 signed difference" "${required_figures[1]}"
  present_figure 3 "Case 2 absolute difference" "${required_figures[2]}"
  present_figure 4 "Case 2 anatomical/functional lung overlay" "${required_figures[3]}"

  present_figure 5 "Case 2 epsilon-normalized intensity map" "${required_figures[4]}"
  local epsilon_value
  jq -r '"Formula: \(.formula.expression)"' "$normalized/normalized_intensity_report.json"
  epsilon_value="$(jq -er '.denominator_policy.epsilon' "$normalized/normalized_intensity_report.json")"
  printf 'Epsilon: %.15e\n' "$epsilon_value"
  printf 'Normalized report: %s\n' "$normalized/normalized_intensity_report.json"
  printf '%s\n' 'Exploratory normalized registered intensity-change map; not validated ventilation.'

  print_equal_window_explanation
  present_figure 6 "Case 1 vs Case 2 matched-window grayscale axial differences" "${required_figures[5]}"
  present_figure 7 "Case 1 vs Case 2 matched-window turbo axial differences" "${required_figures[6]}"
  present_figure 8 "Case 1 vs Case 2 matched-window transparent turbo anatomy overlay" "${required_figures[7]}"
  present_figure 9 "Case 1 vs Case 2 matched-window turbo multiplanar differences" "${required_figures[8]}"
  printf '%s\n' 'RGB Secondary Capture series are validated and ready for DICOM viewer import.'
  printf 'Reveal command: %q %q\n' "./presentation_commands.sh" "rgb-dicom-series"

  print_lung_masked_explanation
  present_figure 10 "Case 1 vs Case 2 lung-masked turbo axial differences" "${required_figures[9]}"
  present_figure 11 "Case 1 vs Case 2 lung-masked turbo multiplanar differences" "${required_figures[10]}"
  printf '%s\n' 'Lung-masked RGB Secondary Capture series are validated and ready for DICOM viewer import.'
  printf 'Reveal command: %q %q\n' "./presentation_commands.sh" "lung-masked-dicom-series"

  present_figure 12 "Case 1 vs Case 2 runtime comparison" "${required_figures[11]}"
  print_runtime_details

  present_figure 13 "Case 1 vs Case 2 registration-quality comparison" "${required_figures[12]}"
  printf '%s\n' 'Full and cropped domains differ; absolute MSE/MAE are not directly comparable across those spatial domains.'

  present_figure 14 "Case 1 vs Case 2 intensity-change comparison" "${required_figures[13]}"
  print_intensity_details

  present_figure 15 "Case 1 vs Case 2 normalized comparison" "${required_figures[14]}"
  print_normalized_details
}

stage_professor_update_backup() {
  local case_2_full="$ROOT_DIR/outputs/registration/registration_case_2/full"
  local backup_paths=(
    "$case_2_full/qc/before_overlay.png"
    "$case_2_full/qc/before_checkerboard.png"
    "$case_2_full/qc/after_checkerboard.png"
    "$case_2_full/qc/difference_before.png"
    "$case_2_full/qc/difference_after.png"
    "$case_2_full/metric_analysis/metric_summary.png"
    "$case_2_full/metric_analysis/error_histograms.png"
    "$case_2_full/functional_lung/functional_lung_report.json"
    "$case_2_full/normalized_intensity_epsilon_machine/normalized_intensity_report.json"
    "$CASE_COMPARISON_DIR/case_1_vs_case_2.json"
    "$EQUAL_WINDOW_REPORT"
    "$RGB_DICOM_REPORT"
    "$LUNG_MASKED_REPORT"
    "$LUNG_MASKED_DICOM_REPORT"
  )
  local path
  for path in "${backup_paths[@]}"; do require_file "$path"; done
  printf '%s\n' 'PROFESSOR UPDATE BACKUP FIGURES AND REPORTS'
  for path in "${backup_paths[@]}"; do printf 'Backup: %s\n' "$path"; done
}

stage_tests() {
  cat <<'EOF'
Focused test commands (printed only; nothing is run):
  python -m pytest tests/test_viewer_launcher_itksnap.py
  python -m pytest tests/test_registration_qc.py
  python -m pytest tests/test_registration_metric_analysis.py
  python -m pytest tests/test_intensity_change_map.py
  python -m pytest tests/test_functional_lung_analysis.py
  python -m pytest tests/test_normalized_intensity_map.py
  python -m pytest tests/test_case_comparison.py
EOF
}

stage_results_summary() {
  local comparison="$CASE_DIR/comparison/registration_comparison.json"
  local intensity="$FULL_DIR/intensity_change/intensity_change_manifest.json"
  local full_manifest="$FULL_DIR/registration_manifest.json"
  local functional="$FULL_DIR/functional_lung/functional_lung_report.json"
  require_command jq
  require_file "$comparison"; require_file "$intensity"; require_file "$full_manifest"; require_file "$functional"
  jq -n --slurpfile c "$comparison" --slurpfile i "$intensity" --slurpfile r "$full_manifest" --slurpfile f "$functional" '
    ($c[0].experiments.full) as $full |
    ($c[0].experiments.cropped) as $cropped |
    {
      runtime_seconds: {full: $full.runtime.total_seconds, cropped: $cropped.runtime.total_seconds},
      reductions_percent: {
        voxels: (100 * (1 - ($cropped.voxel_count / $full.voxel_count))),
        runtime: (100 * (1 - ($cropped.runtime.total_seconds / $full.runtime.total_seconds)))
      },
      full_metrics: {before: $full.metrics_before, after: $full.metrics_after},
      cropped_metrics: {before: $cropped.metrics_before, after: $cropped.metrics_after},
      whole_lung_intensity_difference: {
        signed: $i[0].statistics.signed.combined_lung,
        absolute: $i[0].statistics.absolute.combined_lung
      },
      unresolved_scientific_assumptions: (($r[0].warnings + $i[0].warnings + $f[0].warnings) | unique)
    }'
}

print_all() {
  local prefix=(./presentation_commands.sh)
  if [[ "$CASE_ID" != "registration_case_1" ]]; then
    prefix+=(--case "$CASE_ID")
  fi
  printf 'Recommended presentation order for %s (run one at a time):\n' "$CASE_ID"
  local stage
  for stage in original segmentation bbox cropped-inputs full-registration \
    full-registration-qc cropped-registration cropped-registration-qc comparison \
    metric-analysis signed-difference absolute-difference intensity-change-figures \
    functional-lung normalized-map case-comparison equal-window-differences \
    rgb-dicom-series lung-masked-differences lung-masked-dicom-series \
    professor-update professor-update-backup results-summary tests; do
    printf '  '
    printf '%q ' "${prefix[@]}"
    printf '%s\n' "$stage"
  done
}

usage() {
  cat <<'EOF'
Usage: ./presentation_commands.sh [--case CASE_ID] STAGE

Stages:
  original  segmentation  bbox  cropped-inputs
  full-registration  cropped-registration
  full-registration-qc  cropped-registration-qc  comparison
  metric-analysis  signed-difference  absolute-difference
  intensity-change-figures  functional-lung  normalized-map
  case-comparison  equal-window-differences  rgb-dicom-series
  lung-masked-differences  lung-masked-dicom-series
  professor-update  professor-update-backup
  tests  results-summary  all  help

Set PRESENTATION_DRY_RUN=1 to verify and print GUI commands without launching applications.
The default case is registration_case_1. PRESENTATION_CASE may also set it.
EOF
}

case "${1:-help}" in
  original) stage_original ;;
  segmentation) stage_segmentation ;;
  bbox) stage_bbox ;;
  cropped-inputs) stage_cropped_inputs ;;
  full-registration) stage_full_registration ;;
  cropped-registration) stage_cropped_registration ;;
  full-registration-qc) stage_qc full ;;
  cropped-registration-qc) stage_qc cropped ;;
  comparison) stage_comparison ;;
  metric-analysis) stage_metric_analysis ;;
  signed-difference) stage_signed_difference ;;
  absolute-difference) stage_absolute_difference ;;
  intensity-change-figures) stage_intensity_change_figures ;;
  functional-lung) stage_functional_lung ;;
  normalized-map) stage_normalized_map ;;
  case-comparison) stage_case_comparison ;;
  equal-window-differences) stage_equal_window_differences ;;
  rgb-dicom-series) stage_rgb_dicom_series ;;
  lung-masked-differences) stage_lung_masked_differences ;;
  lung-masked-dicom-series) stage_lung_masked_dicom_series ;;
  professor-update) stage_professor_update ;;
  professor-update-backup) stage_professor_update_backup ;;
  tests) stage_tests ;;
  results-summary) stage_results_summary ;;
  all) print_all ;;
  help|-h|--help) usage ;;
  *) printf 'Error: unknown stage: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac
