#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_URL="${DEEDSBCV_UPSTREAM_URL:-https://github.com/mattiaspaul/deedsBCV.git}"
PINNED_COMMIT="${DEEDSBCV_PINNED_COMMIT:-00b46539aa03d23988a241173d8f1f6c0c68dc6a}"
REQUIRED_BINARIES=(linearBCV deedsBCV)
OPTIONAL_BINARIES=(applyBCV applyBCVfloat)
EXPECTED_BINARIES=("${REQUIRED_BINARIES[@]}" "${OPTIONAL_BINARIES[@]}")

CHECK_ONLY=0
FORCE=0
WITHOUT_OPENMP=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_deedsbcv_macos.sh [--check-only] [--force] [--without-openmp]

  --check-only       Report prerequisites and local setup state; do not clone or build.
  --force            Replace managed source/build directories after verifying source safety.
  --without-openmp   Compile a serial arm64 build without Homebrew libomp.
EOF
}

while (($#)); do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --force) FORCE=1 ;;
    --without-openmp) WITHOUT_OPENMP=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "deedsBCV setup requires macOS (Darwin)." >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "deedsBCV setup requires Apple Silicon arm64." >&2
  exit 1
fi

for command_name in git clang++ shasum; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DEEDSBCV_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DEEDS_ROOT="$PROJECT_ROOT/third_party/deedsBCV"
SOURCE_DIR="$DEEDS_ROOT/source"
BUILD_DIR="$DEEDS_ROOT/build-arm64"
PATCH_FILE="$DEEDS_ROOT/patches/0001-fix-nifti-writer-and-array-deallocation.patch"
BUILD_STATE="$BUILD_DIR/.deedsbcv-build-state.json"

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Required deedsBCV patch is missing: $PATCH_FILE" >&2
  exit 1
fi
PATCH_SHA256="$(shasum -a 256 "$PATCH_FILE" | awk '{print $1}')"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
printf '#include <zlib.h>\nint main(){return zlibVersion()==0;}\n' > "$TMP_DIR/zlib_check.cpp"
if ! clang++ -arch arm64 "$TMP_DIR/zlib_check.cpp" -lz -o "$TMP_DIR/zlib_check" >/dev/null 2>&1; then
  echo "zlib headers/library were not usable by clang++. Install the macOS Command Line Tools." >&2
  exit 1
fi

OPENMP_FLAGS=()
OPENMP_STATUS="disabled by request"
BUILD_MODE="serial"
if ((WITHOUT_OPENMP == 0)); then
  if ! command -v brew >/dev/null 2>&1; then
    OPENMP_STATUS="Homebrew not found"
    if ((CHECK_ONLY == 0)); then
      echo "Homebrew is required for the recommended OpenMP build." >&2
      echo "Install Homebrew from https://brew.sh, then run: brew install libomp" >&2
      echo "Or rerun with --without-openmp for a serial build." >&2
      exit 1
    fi
  elif LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null)" && [[ -d "$LIBOMP_PREFIX" ]]; then
    OPENMP_FLAGS=(-Xclang -fopenmp -I"$LIBOMP_PREFIX/include" -L"$LIBOMP_PREFIX/lib" -lomp)
    OPENMP_STATUS="enabled from Homebrew libomp"
    BUILD_MODE="openmp"
  else
    OPENMP_STATUS="not found (install with: brew install libomp)"
    if ((CHECK_ONLY == 0)); then
      echo "Homebrew llvm-openmp/libomp was not found." >&2
      echo "Install it explicitly with: brew install libomp" >&2
      echo "Or rerun with --without-openmp for a serial build." >&2
      exit 1
    fi
  fi
fi

echo "Host: macOS arm64"
echo "Compiler: $(command -v clang++)"
echo "zlib: available"
echo "OpenMP: $OPENMP_STATUS"
echo "Pinned commit: $PINNED_COMMIT"
echo "Patch: $PATCH_FILE"
echo "Patch SHA-256: $PATCH_SHA256"
echo "Source directory: $SOURCE_DIR"
echo "Build directory: $BUILD_DIR"

source_patch_state() {
  if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    echo "not-cloned"
    return
  fi
  local current_commit status_file diff_file
  current_commit="$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$current_commit" != "$PINNED_COMMIT" ]]; then
    echo "wrong-commit"
    return
  fi
  status_file="$TMP_DIR/source-status"
  git -C "$SOURCE_DIR" status --porcelain --untracked-files=all > "$status_file"
  if [[ ! -s "$status_file" ]]; then
    echo "not-applied"
    return
  fi
  if grep -q '^??' "$status_file"; then
    echo "unrelated-modifications"
    return
  fi
  diff_file="$TMP_DIR/source.patch"
  git -C "$SOURCE_DIR" diff --binary --no-ext-diff -- src > "$diff_file"
  if cmp -s "$PATCH_FILE" "$diff_file"; then
    echo "applied"
  else
    echo "unrelated-modifications"
  fi
}

if ((CHECK_ONLY)); then
  echo "Patch state: $(source_patch_state)"
  if [[ -f "$BUILD_STATE" ]]; then
    echo "Build state: available"
  else
    echo "Build state: missing"
  fi
  for binary in "${EXPECTED_BINARIES[@]}"; do
    if [[ -x "$BUILD_DIR/$binary" ]]; then
      echo "$binary: available"
    else
      echo "$binary: missing"
    fi
  done
  exit 0
fi

if [[ -e "$SOURCE_DIR" && ! -d "$SOURCE_DIR/.git" ]]; then
  echo "Refusing to replace non-Git source directory: $SOURCE_DIR" >&2
  exit 1
fi

if [[ -d "$SOURCE_DIR/.git" ]]; then
  PATCH_STATE="$(source_patch_state)"
  case "$PATCH_STATE" in
    wrong-commit)
      current="$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null || true)"
      echo "Source is at ${current:-unreadable}, not pinned commit $PINNED_COMMIT." >&2
      exit 1
      ;;
    unrelated-modifications)
      echo "Refusing deedsBCV source with modifications other than the tracked patch." >&2
      exit 1
      ;;
  esac
  if ((FORCE)); then
    rm -rf "$SOURCE_DIR" "$BUILD_DIR"
  fi
elif ((FORCE)); then
  rm -rf "$BUILD_DIR"
fi

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  mkdir -p "$DEEDS_ROOT"
  git clone --no-checkout "$UPSTREAM_URL" "$SOURCE_DIR"
  git -C "$SOURCE_DIR" fetch --depth 1 origin "$PINNED_COMMIT"
  git -C "$SOURCE_DIR" checkout --detach "$PINNED_COMMIT"
fi

REMOTE_URL="$(git -C "$SOURCE_DIR" remote get-url origin)"
if [[ "$REMOTE_URL" != "$UPSTREAM_URL" && "$REMOTE_URL" != "${UPSTREAM_URL%.git}" ]]; then
  echo "Refusing source directory with unexpected origin: $REMOTE_URL" >&2
  exit 1
fi
CURRENT_COMMIT="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$PINNED_COMMIT" ]]; then
  echo "Pinned commit verification failed: expected $PINNED_COMMIT, found $CURRENT_COMMIT" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_DIR/LICENSE" ]]; then
  echo "Pinned source does not contain the expected LICENSE; refusing to build." >&2
  exit 1
fi

PATCH_STATE="$(source_patch_state)"
case "$PATCH_STATE" in
  not-applied)
    if ! git -C "$SOURCE_DIR" apply --check "$PATCH_FILE"; then
      echo "Tracked deedsBCV patch does not apply cleanly to pinned source." >&2
      exit 1
    fi
    git -C "$SOURCE_DIR" apply "$PATCH_FILE"
    echo "Patch status: newly applied"
    ;;
  applied)
    echo "Patch status: already applied"
    ;;
  *)
    echo "Cannot apply tracked patch; source state is $PATCH_STATE." >&2
    exit 1
    ;;
esac

mkdir -p "$BUILD_DIR"
COMMON_FLAGS=(-arch arm64 -std=c++11 -O3 -I"$SOURCE_DIR/src")
SOURCES=(linearBCV.cpp deedsBCV0.cpp applyBCV.cpp applyBCVfloat.cpp)
BUILD_STARTED_EPOCH="$(date +%s)"
BUILD_STARTED_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
REQUIRED_BUILT=()
OPTIONAL_BUILT=()
OPTIONAL_UNAVAILABLE=()

for index in "${!EXPECTED_BINARIES[@]}"; do
  binary="${EXPECTED_BINARIES[$index]}"
  source_file="$SOURCE_DIR/src/${SOURCES[$index]}"
  temporary_output="$TMP_DIR/$binary"
  echo "Building $binary"
  if clang++ "${COMMON_FLAGS[@]}" "$source_file" ${OPENMP_FLAGS[@]+"${OPENMP_FLAGS[@]}"} -lz -o "$temporary_output"; then
    mv "$temporary_output" "$BUILD_DIR/$binary"
    if ((index < ${#REQUIRED_BINARIES[@]})); then
      REQUIRED_BUILT+=("$binary")
    else
      OPTIONAL_BUILT+=("$binary")
    fi
  elif ((index < ${#REQUIRED_BINARIES[@]})); then
    echo "Required binary failed to build: $binary" >&2
    exit 1
  else
    echo "Warning: optional binary unavailable: $binary" >&2
    OPTIONAL_UNAVAILABLE+=("$binary")
  fi
done

join_csv() { local IFS=,; echo "$*"; }
json_array() {
  local item separator=""
  printf '['
  for item in "$@"; do
    printf '%s"%s"' "$separator" "$item"
    separator=,
  done
  printf ']'
}
REQUIRED_BUILT_CSV="$(join_csv ${REQUIRED_BUILT[@]+"${REQUIRED_BUILT[@]}"})"
OPTIONAL_BUILT_CSV="$(join_csv ${OPTIONAL_BUILT[@]+"${OPTIONAL_BUILT[@]}"})"
OPTIONAL_UNAVAILABLE_CSV="$(join_csv ${OPTIONAL_UNAVAILABLE[@]+"${OPTIONAL_UNAVAILABLE[@]}"})"
COMPILER="$(command -v clang++)"
cat > "$BUILD_STATE" <<EOF
{
  "pinned_commit": "$PINNED_COMMIT",
  "patch": "$(basename "$PATCH_FILE")",
  "patch_sha256": "$PATCH_SHA256",
  "build_mode": "$BUILD_MODE",
  "compiler": "$COMPILER",
  "timestamp": "$BUILD_STARTED_ISO",
  "timestamp_epoch": $BUILD_STARTED_EPOCH,
  "required_binaries": $(json_array ${REQUIRED_BUILT[@]+"${REQUIRED_BUILT[@]}"}),
  "optional_binaries": $(json_array ${OPTIONAL_BUILT[@]+"${OPTIONAL_BUILT[@]}"}),
  "optional_unavailable": $(json_array ${OPTIONAL_UNAVAILABLE[@]+"${OPTIONAL_UNAVAILABLE[@]}"})
}
EOF

echo "Build summary:"
echo "  Required binaries built: $REQUIRED_BUILT_CSV"
echo "  Optional binaries built: $OPTIONAL_BUILT_CSV"
echo "  Optional binaries unavailable: $OPTIONAL_UNAVAILABLE_CSV"
echo "  Build state: $BUILD_STATE"
echo "No registration was run."
