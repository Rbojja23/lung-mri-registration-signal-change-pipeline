# deedsBCV provenance

- Upstream repository: https://github.com/mattiaspaul/deedsBCV
- Pinned commit: `00b46539aa03d23988a241173d8f1f6c0c68dc6a`
- Upstream license: MIT (the setup script preserves the upstream `LICENSE`)
- Expected binaries: `linearBCV`, `deedsBCV`, `applyBCV`, `applyBCVfloat`

Run `scripts/setup_deedsbcv_macos.sh` to clone the pinned source into the
ignored `source/` directory and compile arm64-safe binaries into the ignored
`build-arm64/` directory. The build deliberately omits the upstream Makefile's
x86-only AVX2 and SSE4.2 flags. OpenMP is used when Homebrew `libomp` is
available; `--without-openmp` selects the serial fallback.

Do not commit the cloned source, generated build files, or compiled binaries.
