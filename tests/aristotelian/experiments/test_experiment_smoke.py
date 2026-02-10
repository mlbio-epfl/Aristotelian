"""Smoke tests for experiment sections.

These tests verify that each experiment runs without error on minimal inputs.
They do NOT test correctness - only that the code doesn't crash.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from scripts.experiments.registry import SECTION_FUNCS


@pytest.fixture
def temp_assets_dir():
    """Create a temporary directory for experiment outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TestRegistryCompleteness:
    """Test that all sections are registered correctly."""

    def test_all_sections_have_functions(self):
        """Each section name maps to a callable."""
        for name, fn in SECTION_FUNCS.items():
            assert callable(fn), f"Section {name} is not callable"

    def test_expected_sections_exist(self):
        """Expected experiment sections are registered."""
        expected = {
            "null_drift_gaussian",
            "null_drift_heavy",
            "perm_budget",
            "type1_calibration",
            "snr_sweep",
            "phase_diagram",
            "prh_alignment",
            "v2t_alignment",
            "exp_b_aggregator_calibration",
        }
        actual = set(SECTION_FUNCS.keys())
        missing = expected - actual
        extra = actual - expected
        assert not missing, f"Missing sections: {missing}"
        # Extra sections are OK - they might be new
        if extra:
            print(f"Note: Found extra sections: {extra}")


class TestSectionFunctionSignatures:
    """Test that section functions have correct signatures."""

    def test_all_sections_accept_required_args(self):
        """All sections must accept assets_dir, device, force, and seed."""
        import inspect

        for name, fn in SECTION_FUNCS.items():
            sig = inspect.signature(fn)
            params = sig.parameters
            # Check for the required parameters
            assert "device" in params, f"{name} missing 'device' parameter"
            assert "force" in params, f"{name} missing 'force' parameter"
            assert "seed" in params, f"{name} missing 'seed' parameter"


@pytest.mark.integration
class TestIntegrationSmoke:
    """Integration tests that run actual experiments.

    These are skipped by default. Run with:
        pytest -m integration
    """

    def test_perm_budget_integration(self, temp_assets_dir):
        """Test perm_budget section runs end-to-end."""
        torch.manual_seed(42)
        fn = SECTION_FUNCS["perm_budget"]
        fn(temp_assets_dir, device="cpu", force=True, seed=42)
        assert (temp_assets_dir / "perm_budget.npy").exists()

    def test_type1_calibration_integration(self, temp_assets_dir):
        """Test type1_calibration section runs end-to-end."""
        torch.manual_seed(42)
        fn = SECTION_FUNCS["type1_calibration"]
        fn(temp_assets_dir, device="cpu", force=True, seed=42)
        assert (temp_assets_dir / "type1_calibration.npy").exists()
