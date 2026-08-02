#!/usr/bin/env bash
set -euo pipefail

# Read-only presentation launcher for the completed registration_case_1 outputs.
# This script never invokes segmentation, cropping, registration, or QC generation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf 'Error: required existing output is missing: %s\n' "$path" >&2
    return 1
  fi
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Error: required command is unavailable: %s\n' "$command_name" >&2
    return 1
  fi
}

launch_itksnap() {
  require_command "$PYTHON_BIN"
  require_file "src/pipeline.py"
  "$PYTHON_BIN" src/pipeline.py launch --viewer itksnap "$@"
}

open_pngs() {
  require_command open
  local path
  for path in "$@"; do
    require_file "$path"
  done
  open -a Preview "$@"
}

# Stage 1: Original fixed and moving scans before registration
stage_original() {
  local fixed="outputs/registration/registration_case_1/full/preprocessed/fixed.nii.gz"
  local moving="outputs/registration/registration_case_1/full/preprocessed/moving.nii.gz"
  require_file "$fixed"
  require_file "$moving"
  launch_itksnap --image "$fixed" --overlay "$moving"
}

# Stage 2: Lung segmentation
stage_segmentation() {
  local image="outputs/registration/registration_case_1/full/preprocessed/fixed.nii.gz"
  local labels="outputs/itksnap/registration_case_1_fixed/lungs_multilabel.nii.gz"
  local descriptions="outputs/itksnap/registration_case_1_fixed/label_descriptions.txt"
  require_file "$image"
  require_file "$labels"
  require_file "$descriptions"
  launch_itksnap --image "$image" --segmentation "$labels" --labels "$descriptions"
}

# Stage 3: Bounding box and lung-focused region
stage_bbox() {
  local image="outputs/registration/registration_case_1/full/preprocessed/fixed.nii.gz"
  local labels="outputs/itksnap/registration_case_1_fixed/lungs_and_bbox_labels.nii.gz"
  local descriptions="outputs/itksnap/registration_case_1_fixed/label_descriptions.txt"
  require_file "$image"
  require_file "$labels"
  require_file "$descriptions"
  launch_itksnap --image "$image" --segmentation "$labels" --labels "$descriptions"
}

# Stage 4: Shared cropped fixed and moving images
stage_crop() {
  local fixed="outputs/registration/registration_case_1/cropped/preprocessed/fixed.nii.gz"
  local moving="outputs/registration/registration_case_1/cropped/preprocessed/moving.nii.gz"
  require_file "$fixed"
  require_file "$moving"
  launch_itksnap --image "$fixed" --overlay "$moving"
}

# Stage 5: Full-volume registration result
stage_full_registration() {
  local fixed="outputs/registration/registration_case_1/full/preprocessed/fixed.nii.gz"
  local registered="outputs/registration/registration_case_1/full/deformable/moving_registered.nii.gz"
  require_file "$fixed"
  require_file "$registered"
  launch_itksnap --image "$fixed" --overlay "$registered"
}

# Stage 6: Cropped registration result
stage_cropped_registration() {
  local fixed="outputs/registration/registration_case_1/cropped/preprocessed/fixed.nii.gz"
  local registered="outputs/registration/registration_case_1/cropped/deformable/moving_registered.nii.gz"
  require_file "$fixed"
  require_file "$registered"
  launch_itksnap --image "$fixed" --overlay "$registered"
}

# Stage 7: Full registration QC visualizations
stage_full_qc() {
  open_pngs \
    "outputs/registration/registration_case_1/full/qc/before_overlay.png" \
    "outputs/registration/registration_case_1/full/qc/after_overlay.png" \
    "outputs/registration/registration_case_1/full/qc/before_checkerboard.png" \
    "outputs/registration/registration_case_1/full/qc/after_checkerboard.png" \
    "outputs/registration/registration_case_1/full/qc/difference_before.png" \
    "outputs/registration/registration_case_1/full/qc/difference_after.png"
}

# Stage 8: Cropped registration QC visualizations
stage_cropped_qc() {
  open_pngs \
    "outputs/registration/registration_case_1/cropped/qc/before_overlay.png" \
    "outputs/registration/registration_case_1/cropped/qc/after_overlay.png" \
    "outputs/registration/registration_case_1/cropped/qc/before_checkerboard.png" \
    "outputs/registration/registration_case_1/cropped/qc/after_checkerboard.png" \
    "outputs/registration/registration_case_1/cropped/qc/difference_before.png" \
    "outputs/registration/registration_case_1/cropped/qc/difference_after.png"
}

# Stage 9: Quantitative full-versus-cropped comparison
stage_comparison() {
  open_pngs \
    "outputs/registration/registration_case_1/comparison/registration_comparison.png"
}

# Stage 10: Final result summary
stage_summary() {
  local manifest="outputs/registration/registration_case_1/comparison/registration_comparison.json"
  require_command jq
  require_file "$manifest"
  jq '{
    case_id,
    completion_status,
    runtime_winner,
    metric_winners,
    overall_anatomical_winner,
    interpretation,
    intended_use
  }' "$manifest"
}

# Stage 11: Optional validation evidence
stage_validation() {
  local full="outputs/registration/registration_case_1/full/registration_manifest.json"
  local cropped="outputs/registration/registration_case_1/cropped/registration_manifest.json"
  local full_qc="outputs/registration/registration_case_1/full/qc/registration_qc.json"
  local cropped_qc="outputs/registration/registration_case_1/cropped/qc/registration_qc.json"
  local comparison="outputs/registration/registration_case_1/comparison/registration_comparison.json"
  require_command jq
  require_file "$full"
  require_file "$cropped"
  require_file "$full_qc"
  require_file "$cropped_qc"
  require_file "$comparison"
  jq -n \
    --slurpfile full "$full" \
    --slurpfile cropped "$cropped" \
    --slurpfile full_qc "$full_qc" \
    --slurpfile cropped_qc "$cropped_qc" \
    --slurpfile comparison "$comparison" \
    '{
      full_registration: $full[0].completion_status,
      cropped_registration: $cropped[0].completion_status,
      full_qc: $full_qc[0].completion_status,
      cropped_qc: $cropped_qc[0].completion_status,
      comparison: $comparison[0].completion_status
    }'
}

print_all() {
  cat <<'EOF'
Run these one at a time from the repository root:
  ./presentation_commands.sh original
  ./presentation_commands.sh segmentation
  ./presentation_commands.sh bbox
  ./presentation_commands.sh crop
  ./presentation_commands.sh full-registration
  ./presentation_commands.sh cropped-registration
  ./presentation_commands.sh full-qc
  ./presentation_commands.sh cropped-qc
  ./presentation_commands.sh comparison
  ./presentation_commands.sh summary
  ./presentation_commands.sh validation
EOF
}

usage() {
  printf 'Usage: %s {original|segmentation|bbox|crop|full-registration|cropped-registration|full-qc|cropped-qc|comparison|summary|validation|all}\n' "$0" >&2
}

case "${1:-}" in
  original) stage_original ;;
  segmentation) stage_segmentation ;;
  bbox) stage_bbox ;;
  crop) stage_crop ;;
  full-registration) stage_full_registration ;;
  cropped-registration) stage_cropped_registration ;;
  full-qc) stage_full_qc ;;
  cropped-qc) stage_cropped_qc ;;
  comparison) stage_comparison ;;
  summary) stage_summary ;;
  validation) stage_validation ;;
  all) print_all ;;
  *) usage; exit 2 ;;
esac
