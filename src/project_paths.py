from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RAW_DATA_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def project_root() -> Path:
    """Return the repository root derived from this source file location."""
    return PROJECT_ROOT


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_user_path(
    path: str | Path | None,
    *,
    default: Path,
    base: Path = PROJECT_ROOT,
    allow_outside: bool = False,
) -> Path:
    """Resolve a user-supplied path without silently escaping the project root."""
    if path is None:
        candidate = default
    else:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate

    resolved = candidate.resolve(strict=False)
    resolved_base = base.resolve(strict=False)

    if not allow_outside and not _is_relative_to(resolved, resolved_base):
        raise ValueError(f"Path is outside the project root: {display_path(resolved)}")

    return resolved


def display_path(path: str | Path, *, root: Path = PROJECT_ROOT) -> str:
    """Return a project-relative display path when possible."""
    resolved = Path(path).resolve(strict=False)
    resolved_root = root.resolve(strict=False)

    if resolved == resolved_root:
        return "."

    if _is_relative_to(resolved, resolved_root):
        return str(resolved.relative_to(resolved_root))

    return "[external path]"


def ensure_output_dir(path: str | Path, *, create: bool = False) -> Path:
    """Return an output directory path, creating it only when requested."""
    resolved = Path(path).expanduser().resolve(strict=False)
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT
    data_dir: Path = DEFAULT_DATA_DIR
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR

    @classmethod
    def from_overrides(
        cls,
        *,
        data_root: str | Path | None = None,
        output_root: str | Path | None = None,
        root: Path = PROJECT_ROOT,
        allow_external_overrides: bool = True,
    ) -> "ProjectPaths":
        resolved_root = root.resolve(strict=False)
        data_dir = resolve_user_path(
            data_root,
            default=resolved_root / "data",
            base=resolved_root,
            allow_outside=allow_external_overrides,
        )
        outputs_dir = resolve_user_path(
            output_root,
            default=resolved_root / "outputs",
            base=resolved_root,
            allow_outside=allow_external_overrides,
        )
        raw_data_dir = data_dir / "raw"

        return cls(
            root=resolved_root,
            data_dir=data_dir,
            raw_data_dir=raw_data_dir,
            outputs_dir=outputs_dir,
        )

    def output_dir(self, *parts: str, create: bool = False) -> Path:
        return ensure_output_dir(self.outputs_dir.joinpath(*parts), create=create)

    def raw_data_path(self, *parts: str) -> Path:
        return self.raw_data_dir.joinpath(*parts)

