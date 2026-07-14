# Release notes: v1.1.0

- Migrated the primary interactive workflow from OsiriX-first to ITK-SNAP-first.
- Added system-first macOS ITK-SNAP bundle detection, health checking, and safe
  `-g`/`-s`/`-l` argument-list launching.
- Added authoritative MRI left/right multilabel output, an authoritative
  six-face half-open bbox shell, and a combined visualization label image.
- Integrated ITK-SNAP product generation and optional launch into `run-all`.
- Retained tested, explicit RTSTRUCT capability validation/export and
  OsiriX/Horos launching for compatible original DICOM datasets.
- Documented that the supplied Secondary Capture MRI lacks
  `FrameOfReferenceUID`; this blocks RTSTRUCT, not NIfTI/ITK-SNAP review.
- CT loading, segmentation, QC, bbox, and crops remain supported individually;
  five-lobe ITK-SNAP preparation is not configured in v1.1.
- Release validation: 151 synthetic/unit tests.
