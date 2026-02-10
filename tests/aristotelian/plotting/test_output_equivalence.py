"""Tests for output equivalence before and after refactoring.

These tests compare generated plots against reference outputs to ensure
refactoring doesn't change visual output.

Uses perceptual hashing (imagehash) to allow for minor rendering differences
while catching significant visual changes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from PIL import Image


def _pdf_to_png(pdf_path: Path, output_path: Path, dpi: int = 150) -> bool:
    """Convert PDF to PNG using pdftoppm (poppler-utils).

    Returns True if conversion succeeded, False otherwise.
    """
    try:
        # pdftoppm outputs to stdout or with -png flag to file
        result = subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                str(dpi),
                "-singlefile",
                str(pdf_path),
                str(output_path.with_suffix("")),  # pdftoppm adds .png
            ],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _load_image(path: Path) -> "Image.Image | None":
    """Load image from path, handling PDF conversion if needed."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not installed")
        return None

    if path.suffix.lower() == ".pdf":
        # Try to convert PDF to PNG
        png_path = path.with_suffix(".png")
        if not png_path.exists():
            if not _pdf_to_png(path, png_path):
                return None
        path = png_path

    if not path.exists():
        return None

    return Image.open(path)


def _compute_phash(image: "Image.Image", hash_size: int = 16) -> np.ndarray | None:
    """Compute perceptual hash of image.

    Uses a simple DCT-based perceptual hash algorithm.
    Returns None if imagehash is not available, falls back to simple comparison.
    """
    try:
        import imagehash

        return imagehash.phash(image, hash_size=hash_size)
    except ImportError:
        # Fallback: simple resize and threshold
        gray = image.convert("L")
        resized = gray.resize((hash_size, hash_size))
        arr = np.array(resized)
        median = np.median(arr)
        return (arr > median).astype(np.uint8)


def _hash_difference(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Compute Hamming distance between two hashes."""
    try:
        import imagehash

        if isinstance(hash1, imagehash.ImageHash) and isinstance(
            hash2, imagehash.ImageHash
        ):
            return hash1 - hash2
    except ImportError:
        pass

    # Fallback for numpy arrays
    return int(np.sum(hash1 != hash2))


class TestOutputEquivalence:
    """Test output equivalence against reference outputs."""

    # Maximum allowed perceptual hash difference (bits)
    MAX_HASH_DIFF = 5

    @pytest.fixture
    def reference_dir(self) -> Path:
        """Path to reference outputs directory."""
        return Path(__file__).parent / "reference_outputs"

    def _compare_outputs(
        self, generated_path: Path, reference_path: Path
    ) -> tuple[bool, str]:
        """Compare two output files.

        Returns (is_equivalent, message).
        """
        if not generated_path.exists():
            return False, f"Generated file not found: {generated_path}"

        if not reference_path.exists():
            return False, f"Reference file not found: {reference_path}"

        gen_img = _load_image(generated_path)
        ref_img = _load_image(reference_path)

        if gen_img is None:
            return False, f"Could not load generated image: {generated_path}"
        if ref_img is None:
            return False, f"Could not load reference image: {reference_path}"

        gen_hash = _compute_phash(gen_img)
        ref_hash = _compute_phash(ref_img)

        if gen_hash is None or ref_hash is None:
            return False, "Could not compute perceptual hash"

        diff = _hash_difference(gen_hash, ref_hash)
        is_equivalent = diff <= self.MAX_HASH_DIFF

        return (
            is_equivalent,
            f"Hash difference: {diff} bits (max allowed: {self.MAX_HASH_DIFF})",
        )

    @pytest.mark.skipif(
        not (Path(__file__).parent / "reference_outputs").exists()
        or not any((Path(__file__).parent / "reference_outputs").iterdir()),
        reason="No reference outputs available",
    )
    def test_null_drift_gaussian_equivalence(
        self, populated_assets_dir: Path, reference_dir: Path
    ) -> None:
        """Test null_drift_gaussian output matches reference."""
        from scripts.plots.experiments import _plot_null_drift_gaussian

        _plot_null_drift_gaussian(populated_assets_dir, force=True)

        # Check main output
        generated = populated_assets_dir / "null_drift_gaussian_cka_lin.pdf"
        reference = reference_dir / "null_drift_gaussian_cka_lin.pdf"

        if not reference.exists():
            pytest.skip(f"Reference not found: {reference}")

        is_equiv, msg = self._compare_outputs(generated, reference)
        assert is_equiv, f"Output mismatch: {msg}"

    @pytest.mark.skipif(
        not (Path(__file__).parent / "reference_outputs").exists()
        or not any((Path(__file__).parent / "reference_outputs").iterdir()),
        reason="No reference outputs available",
    )
    def test_null_drift_heavy_equivalence(
        self, populated_assets_dir: Path, reference_dir: Path
    ) -> None:
        """Test null_drift_heavy output matches reference."""
        from scripts.plots.experiments import _plot_null_drift_heavy

        # Create heavy data
        from tests.aristotelian.plotting.create_test_data import create_null_drift_data

        n_list = [50, 100, 200, 500]
        d_list = [10, 50, 100, 500]
        data = create_null_drift_data(n_list, d_list)
        np.save(populated_assets_dir / "null_drift_heavy.npy", data)

        _plot_null_drift_heavy(populated_assets_dir, force=True)

        generated = populated_assets_dir / "null_drift_heavy_cka_lin.pdf"
        reference = reference_dir / "null_drift_heavy_cka_lin.pdf"

        if not reference.exists():
            pytest.skip(f"Reference not found: {reference}")

        is_equiv, msg = self._compare_outputs(generated, reference)
        assert is_equiv, f"Output mismatch: {msg}"


class TestOutputConsistency:
    """Test that outputs are consistent across runs."""

    def test_deterministic_output(self, temp_assets_dir: Path) -> None:
        """Test that same input produces same output."""
        from tests.aristotelian.plotting.create_test_data import create_null_drift_data

        # Create test data
        n_list = [50, 100]
        d_list = [10, 50]
        data = create_null_drift_data(n_list, d_list)
        np.save(temp_assets_dir / "null_drift_gaussian.npy", data)

        from scripts.plots.experiments import _plot_null_drift_gaussian

        # Generate first output
        _plot_null_drift_gaussian(temp_assets_dir, force=True)
        output1 = temp_assets_dir / "null_drift_gaussian_cka_lin.pdf"
        assert output1.exists(), "Expected plot output was not generated"

        import shutil

        backup = temp_assets_dir / "output1_backup.pdf"
        shutil.copy(output1, backup)

        # Generate second output
        _plot_null_drift_gaussian(temp_assets_dir, force=True)
        output2 = output1
        assert output2.exists(), "Expected plot output missing on second run"

        # Compare file sizes as quick check
        size1 = backup.stat().st_size
        size2 = output2.stat().st_size

        # Allow small variation due to timestamps in PDF
        assert abs(size1 - size2) < 1000, "Output file sizes differ significantly"


class TestReferenceGeneration:
    """Tests for the reference generation process."""

    def test_reference_generator_imports(self) -> None:
        """Test that reference generator script can be imported."""
        from scripts.plots.references import generate_references

        assert callable(generate_references)

    def test_section_funcs_accessible(self) -> None:
        """Test that SECTION_FUNCS is accessible."""
        from scripts.plots.experiments import SECTION_FUNCS

        assert "null_drift_gaussian" in SECTION_FUNCS
        assert "null_drift_heavy" in SECTION_FUNCS
        assert callable(SECTION_FUNCS["null_drift_gaussian"])
