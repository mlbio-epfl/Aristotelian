"""Pytest fixtures for plotting tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Import style application function
from scripts.plots.experiments import _apply_style


@pytest.fixture(scope="session", autouse=True)
def apply_prh_style() -> None:
    """Apply PRH style for all plotting tests (session-scoped)."""
    _apply_style()


@pytest.fixture
def temp_assets_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test assets and outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def reference_outputs_dir() -> Path:
    """Path to the reference outputs directory for equivalence testing."""
    return Path(__file__).parent / "reference_outputs"


@pytest.fixture
def test_data_dir() -> Path:
    """Path to the test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def populated_assets_dir(
    temp_assets_dir: Path, test_data_dir: Path
) -> Generator[Path, None, None]:
    """Temporary directory pre-populated with test data files."""
    import shutil

    # Copy all test data files to the temp directory
    if test_data_dir.exists():
        for f in test_data_dir.glob("*.npy"):
            shutil.copy(f, temp_assets_dir / f.name)
    yield temp_assets_dir
