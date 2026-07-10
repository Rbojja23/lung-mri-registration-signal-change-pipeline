from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from project_paths import (  # noqa: E402
    ProjectPaths,
    display_path,
    ensure_output_dir,
    project_root,
    resolve_user_path,
)


def test_project_root_resolution():
    assert project_root() == Path(__file__).resolve().parents[1]


def test_default_path_construction():
    paths = ProjectPaths()

    assert paths.data_dir == project_root() / "data"
    assert paths.raw_data_dir == project_root() / "data" / "raw"
    assert paths.outputs_dir == project_root() / "outputs"


def test_override_path_resolution_inside_temp_root(tmp_path):
    paths = ProjectPaths.from_overrides(
        data_root="local_data",
        output_root="local_outputs",
        root=tmp_path,
        allow_external_overrides=False,
    )

    assert paths.data_dir == tmp_path / "local_data"
    assert paths.raw_data_dir == tmp_path / "local_data" / "raw"
    assert paths.outputs_dir == tmp_path / "local_outputs"


def test_read_only_paths_do_not_create_directories(tmp_path):
    paths = ProjectPaths.from_overrides(root=tmp_path, allow_external_overrides=False)

    assert not paths.data_dir.exists()
    assert not paths.raw_data_dir.exists()
    assert not paths.outputs_dir.exists()


def test_output_directory_creation_only_when_requested(tmp_path):
    paths = ProjectPaths.from_overrides(root=tmp_path, allow_external_overrides=False)
    report_dir = paths.output_dir("reports")

    assert report_dir == tmp_path / "outputs" / "reports"
    assert not report_dir.exists()

    created_dir = paths.output_dir("reports", create=True)

    assert created_dir.exists()
    assert created_dir.is_dir()


def test_raw_data_directories_are_never_created_automatically(tmp_path):
    paths = ProjectPaths.from_overrides(root=tmp_path, allow_external_overrides=False)
    raw_case = paths.raw_data_path("case_a")

    assert raw_case == tmp_path / "data" / "raw" / "case_a"
    assert not raw_case.exists()
    assert not paths.raw_data_dir.exists()


def test_safe_relative_display_of_repository_paths():
    assert display_path(project_root() / "src" / "project_paths.py") == (
        "src/project_paths.py"
    )


def test_resolve_user_path_blocks_implicit_escape(tmp_path):
    outside = tmp_path.parent / "outside"

    try:
        resolve_user_path(
            outside,
            default=tmp_path / "data",
            base=tmp_path,
            allow_outside=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected paths outside the root to be rejected.")

