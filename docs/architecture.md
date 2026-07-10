# Architecture

## Data Flow

```mermaid
flowchart TD
  A[DICOM or NIfTI input] --> B[Modality-neutral loading]
  B --> C[TotalSegmentator lung segmentation]
  C --> D[Segmentation QC]
  D --> E[Bounding box and crop]
  E --> F[Bounding-box QC]
  F --> G{DICOM reference compatible?}
  G -->|yes| H[RTSTRUCT export]
  G -->|no| I[Fallback review package]
  H --> J[OsiriX/Horos/Slicer launch]
  I --> J
```

## Coordinate Conventions

- SimpleITK indices and image size use `x, y, z`.
- NumPy arrays from SimpleITK use `z, y, x`.
- Physical coordinates are millimeters in the image coordinate frame.
- Bounding boxes are stored as half-open intervals `[min:max)` in NumPy
  `z, y, x` order.

## Geometry Preservation

Images, masks, crops, and overlays are validated with size, spacing, origin, and
direction checks. Crops use SimpleITK region-of-interest operations so the crop
origin is updated to the physical location of the crop start index.

## Export

The OsiriX export stage validates structural DICOM references before writing a
referenced RT Structure Set. It refuses to fabricate missing Frame of Reference
or slice-reference metadata. If a referenced object cannot be guaranteed, it
creates a fallback review package instead.

