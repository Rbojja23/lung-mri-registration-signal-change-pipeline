# deedsBCV provenance

- Upstream repository: https://github.com/mattiaspaul/deedsBCV
- Pinned commit: `00b46539aa03d23988a241173d8f1f6c0c68dc6a`
- Tracked patch: `patches/0001-fix-nifti-writer-and-array-deallocation.patch`
- Upstream license: MIT; the cloned source remains covered by its upstream `LICENSE`
- Patch validation date: 2026-07-15

The tracked patch corrects confirmed C++ undefined behavior in the upstream
NIfTI writers and array deallocations. Three dimension buffers that write five
`short` values are enlarged from four elements to five, and 60 array
allocations are paired with `delete[]` instead of scalar `delete`. It does not
change registration mathematics, preprocessing, command-line behavior, or
output naming.

Before the patch, a completed deformable registration could write a malformed
NIfTI header with `dim = (3,0,0,0,0,1,1,1)`. After the patch, the validated
output header was `dim = (3,360,360,360,1,1,1,1)`.

Case 1 validation completed with affine return code 0 and deformable return
code 0. The 360×360×360 result was readable by SimpleITK and its geometry
matched the fixed image. No patient metadata is recorded here.

The setup script clones the exact pinned revision into ignored `source/`,
applies the tracked patch exactly once, and builds into ignored `build-arm64/`.
Neither cloned upstream source nor compiled binaries should be committed.
