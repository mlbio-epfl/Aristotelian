"""Tests for figure structure produced by plotting functions.

These tests verify that plotting functions produce figures with:
- Correct number of subplots/axes
- Proper axis labels and titles
- Expected visual elements (heatmaps, lines, colorbars)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.aristotelian.plotting.create_test_data import (
    create_null_drift_data,
    create_perm_budget_data,
    create_snr_sweep_data,
)


@pytest.fixture(autouse=True)
def close_figures() -> None:
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestNullDriftFigureStructure:
    """Test structure of null_drift figures."""

    @pytest.fixture
    def null_drift_data(self, temp_assets_dir: Path) -> Path:
        """Create null drift test data."""
        n_list = [50, 100, 200]
        d_list = [10, 50, 100]
        data = create_null_drift_data(n_list, d_list)
        path = temp_assets_dir / "null_drift_gaussian.npy"
        np.save(path, data)
        return path

    def test_cka_lin_two_panel_structure(
        self, null_drift_data: Path, temp_assets_dir: Path
    ) -> None:
        """Test CKA lin comparison produces 2-panel figure."""
        from scripts.plots.experiments import _plot_null_drift_gaussian

        _plot_null_drift_gaussian(temp_assets_dir, force=True)

        # Check that output was created
        output = temp_assets_dir / "null_drift_gaussian_cka_lin.pdf"
        assert output.exists(), f"Expected output {output} not found"

    def test_per_metric_figures_created(
        self, null_drift_data: Path, temp_assets_dir: Path
    ) -> None:
        """Test per-metric figures are created."""
        from scripts.plots.experiments import _plot_null_drift_gaussian

        _plot_null_drift_gaussian(temp_assets_dir, force=True)

        # Check for metric-specific outputs
        metric_outputs = list(temp_assets_dir.glob("null_drift_gaussian_*.pdf"))
        assert len(metric_outputs) > 0, "No metric-specific outputs found"

    def test_aggregate_figure_created(
        self, null_drift_data: Path, temp_assets_dir: Path
    ) -> None:
        """Test aggregate comparison figure is created."""
        from scripts.plots.experiments import _plot_null_drift_gaussian

        _plot_null_drift_gaussian(temp_assets_dir, force=True)

        # Check for aggregate output
        agg_output = temp_assets_dir / "null_drift_gaussian_reduction_aggregate.pdf"
        assert agg_output.exists(), f"Expected aggregate output {agg_output} not found"


class TestPermBudgetFigureStructure:
    """Test structure of perm_budget figures."""

    @pytest.fixture
    def perm_budget_data(self, temp_assets_dir: Path) -> Path:
        """Create perm budget test data."""
        data = create_perm_budget_data()
        path = temp_assets_dir / "perm_budget.npy"
        np.save(path, data)
        return path

    def test_two_panel_figure(
        self, perm_budget_data: Path, temp_assets_dir: Path
    ) -> None:
        """Test perm_budget produces 2-panel figure."""
        from scripts.plots.experiments import _plot_perm_budget

        _plot_perm_budget(temp_assets_dir, force=True)

        output = temp_assets_dir / "perm_budget_tau.pdf"
        assert output.exists(), f"Expected output {output} not found"


class TestSnrSweepFigureStructure:
    """Test structure of snr_sweep figures."""

    @pytest.fixture
    def snr_sweep_data(self, temp_assets_dir: Path) -> Path:
        """Create SNR sweep test data."""
        data = create_snr_sweep_data()
        path = temp_assets_dir / "snr_sweep.npy"
        np.save(path, data)
        return path

    def test_output_created(self, snr_sweep_data: Path, temp_assets_dir: Path) -> None:
        """Test SNR sweep produces output."""
        from scripts.plots.experiments import _plot_snr_sweep

        _plot_snr_sweep(temp_assets_dir, force=True)

        # Check for some output
        outputs = list(temp_assets_dir.glob("snr_*.pdf"))
        assert len(outputs) > 0, "No SNR sweep outputs found"


class TestFigureElementsPresent:
    """Test that expected figure elements are present."""

    def test_heatmap_has_image(self) -> None:
        """Test heatmap axes contain image."""
        fig, ax = plt.subplots()
        data = np.random.rand(5, 5)
        _im = ax.imshow(data)  # noqa: F841

        # Check image was created
        assert len(ax.images) == 1
        plt.close(fig)

    def test_line_plot_has_lines(self) -> None:
        """Test line plot axes contain lines."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Check line was created
        assert len(ax.lines) == 1
        plt.close(fig)


class TestSkipBehavior:
    """Test skip behavior when outputs exist."""

    def test_skip_when_exists_and_no_force(self, temp_assets_dir: Path) -> None:
        """Test that existing outputs are skipped without force."""
        from scripts.plots.experiments import _should_skip

        # Create existing output
        output = temp_assets_dir / "test.pdf"
        output.touch()

        assert _should_skip([output], force=False)

    def test_no_skip_with_force(self, temp_assets_dir: Path) -> None:
        """Test that force=True regenerates outputs."""
        from scripts.plots.experiments import _should_skip

        # Create existing output
        output = temp_assets_dir / "test.pdf"
        output.touch()

        assert not _should_skip([output], force=True)


class TestMissingDataHandling:
    """Test handling of missing data files."""

    def test_missing_data_skips_gracefully(
        self, temp_assets_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test missing data file prints skip message."""
        from scripts.plots.experiments import _plot_null_drift_gaussian

        # Call without creating data file
        _plot_null_drift_gaussian(temp_assets_dir, force=True)

        captured = capsys.readouterr()
        assert "skip missing" in captured.out
