#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_URL="https://github.com/mattiaspaul/deedsBCV.git"
PINNED_COMMIT="00b46539aa03d23988a241173d8f1f6c0c68dc6a"
EXPECTED_BINARIES=(linearBCV deedsBCV applyBCV applyBCVfloat)

CHECK_ONLY=0
FORCE=0
WITHOUT_OPENMP=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_deedsbcv_macos.sh [--check-only] [--force] [--without-openmp]

  --check-only       Report prerequisites and local setup state; do not clone or build.
  --force            Replace only the managed source and build directories.
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

for command_name in git clang++; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEEDS_ROOT="$PROJECT_ROOT/third_party/deedsBCV"
SOURCE_DIR="$DEEDS_ROOT/source"
BUILD_DIR="$DEEDS_ROOT/build-arm64"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
printf '#include <zlib.h>\nint main(){return zlibVersion()==0;}\n' > "$TMP_DIR/zlib_check.cpp"
if ! clang++ -arch arm64 "$TMP_DIR/zlib_check.cpp" -lz -o "$TMP_DIR/zlib_check" >/dev/null 2>&1; then
  echo "zlib headers/library were not usable by clang++. Install the macOS Command Line Tools." >&2
  exit 1
fi

OPENMP_FLAGS=()
OPENMP_STATUS="disabled by request"
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
echo "Source directory: $SOURCE_DIR"
echo "Build directory: $BUILD_DIR"

if ((CHECK_ONLY)); then
  if [[ -d "$SOURCE_DIR/.git" ]]; then
    CURRENT_COMMIT="$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null || true)"
    echo "Local source commit: ${CURRENT_COMMIT:-unreadable}"
  else
    echo "Local source: not cloned"
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

if ((FORCE)); then
  rm -rf "$SOURCE_DIR" "$BUILD_DIR"
fi

if [[ -e "$SOURCE_DIR" && ! -d "$SOURCE_DIR/.git" ]]; then
  echo "Refusing to replace non-Git source directory: $SOURCE_DIR (use --force)." >&2
  exit 1
fi

CLONED_NOW=0
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  mkdir -p "$DEEDS_ROOT"
  git clone --no-checkout "$UPSTREAM_URL" "$SOURCE_DIR"
  CLONED_NOW=1
fi

REMOTE_URL="$(git -C "$SOURCE_DIR" remote get-url origin)"
if [[ "$REMOTE_URL" != "$UPSTREAM_URL" && "$REMOTE_URL" != "${UPSTREAM_URL%.git}" ]]; then
  echo "Refusing source directory with unexpected origin: $REMOTE_URL" >&2
  exit 1
fi

git -C "$SOURCE_DIR" fetch --depth 1 origin "$PINNED_COMMIT"
CURRENT_COMMIT="$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "$CURRENT_COMMIT" && "$CURRENT_COMMIT" != "$PINNED_COMMIT" && $FORCE -eq 0 && $CLONED_NOW -eq 0 ]]; then
  echo "Source is at $CURRENT_COMMIT, not the pinned commit; use --force to replace it." >&2
  exit 1
fi
git -C "$SOURCE_DIR" checkout --detach "$PINNED_COMMIT"

if [[ ! -f "$SOURCE_DIR/LICENSE" ]]; then
  echo "Pinned source does not contain the expected LICENSE; refusing to build." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
COMMON_FLAGS=(-arch arm64 -std=c++11 -O3 -I"$SOURCE_DIR/src")
SOURCES=(linearBCV.cpp deedsBCV0.cpp applyBCV.cpp applyBCVfloat.cpp)

for index in "${!EXPECTED_BINARIES[@]}"; do
  binary="${EXPECTED_BINARIES[$index]}"
  source_file="$SOURCE_DIR/src/${SOURCES[$index]}"
  output_file="$BUILD_DIR/$binary"
  if [[ -x "$output_file" && $FORCE -eq 0 ]]; then
    echo "$binary: already built"
    continue
  fi
  echo "Building $binary"
  clang++ "${COMMON_FLAGS[@]}" "$source_file" "${OPENMP_FLAGS[@]}" -lz -o "$output_file"
done

echo "Built deedsBCV binaries:"
for binary in "${EXPECTED_BINARIES[@]}"; do
  echo "  $BUILD_DIR/$binary"
done
echo "No registration was run."
