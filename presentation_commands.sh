#!/usr/bin/env bash
set -euo pipefail

# Read-only presentation launcher for completed registration_case_1 artifacts.
# Set PRESENTATION_DRY_RUN=1 to validate and print GUI commands without opening them.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$ROOT_DIR/outputs/registration/registration_case_1"
FULL_DIR="$CASE_DIR/full"
CROPPED_DIR="$CASE_DIR/cropped"
FIXED_LABEL_DIR="$ROOT_DIR/outputs/itksnap/registration_case_1_fixed"
LUNG_MASK="$ROOT_DIR/outputs/totalsegmentator/registration_case_1_fixed/lungs/combined_lungs.nii.gz"
DRY_RUN="${PRESENTATION_DRY_RUN:-0}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf 'Error: required existing file is missing: %s\n' "$path" >&2
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
  local dir="$FULL_DIR/normalized_intensity"
  local display="$dir/normalized_difference_display.nii.gz"
  local report="$dir/normalized_intensity_report.json"
  if [[ ! -f "$display" || ! -f "$report" ]]; then
    printf '%s\n' 'Normalized map implementation exists, but Case 1 outputs have not been generated because the denominator threshold remains unresolved.'
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

stage_tests() {
  cat <<'EOF'
Focused test commands (printed only; nothing is run):
  python -m pytest tests/test_viewer_launcher_itksnap.py
  python -m pytest tests/test_registration_qc.py
  python -m pytest tests/test_registration_metric_analysis.py
  python -m pytest tests/test_intensity_change_map.py
  python -m pytest tests/test_functional_lung_analysis.py
  python -m pytest tests/test_normalized_intensity_map.py
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
  cat <<'EOF'
Recommended presentation order (run one at a time):
  ./presentation_commands.sh original
  ./presentation_commands.sh segmentation
  ./presentation_commands.sh bbox
  ./presentation_commands.sh cropped-inputs
  ./presentation_commands.sh full-registration
  ./presentation_commands.sh full-registration-qc
  ./presentation_commands.sh cropped-registration
  ./presentation_commands.sh cropped-registration-qc
  ./presentation_commands.sh comparison
  ./presentation_commands.sh metric-analysis
  ./presentation_commands.sh signed-difference
  ./presentation_commands.sh absolute-difference
  ./presentation_commands.sh intensity-change-figures
  ./presentation_commands.sh functional-lung
  ./presentation_commands.sh normalized-map
  ./presentation_commands.sh results-summary
  ./presentation_commands.sh tests
EOF
}

usage() {
  cat <<'EOF'
Usage: ./presentation_commands.sh STAGE

Stages:
  original  segmentation  bbox  cropped-inputs
  full-registration  cropped-registration
  full-registration-qc  cropped-registration-qc  comparison
  metric-analysis  signed-difference  absolute-difference
  intensity-change-figures  functional-lung  normalized-map
  tests  results-summary  all  help

Set PRESENTATION_DRY_RUN=1 to verify and print GUI commands without launching applications.
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
  tests) stage_tests ;;
  results-summary) stage_results_summary ;;
  all) print_all ;;
  help|-h|--help) usage ;;
  *) printf 'Error: unknown stage: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac
