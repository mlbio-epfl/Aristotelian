#!/usr/bin/env python
"""Generate reference outputs for plotting equivalence tests.

This script runs BEFORE any refactoring to capture baseline outputs.
These become the "golden" comparison targets for equivalence tests.

Usage:
    python -m scripts.plots.references [--assets-dir assets] [--output-dir tests/plotting/reference_outputs]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scripts.plots.experiments import SECTION_FUNCS, _apply_style

# Sections we want to capture references for
# These are the most critical for verifying refactoring correctness
REFERENCE_SECTIONS = [
    "null_drift_gaussian",
    "null_drift_heavy",
    "perm_budget",
    "type1_calibration",
    "snr_sweep",
    "phase_diagram",
]


def generate_references(
    assets_dir: Path,
    output_dir: Path,
    sections: list[str] | None = None,
) -> None:
    """Generate reference outputs for specified sections.

    Args:
        assets_dir: Directory containing experiment data (.npy files)
        output_dir: Directory to store reference outputs
        sections: List of sections to generate references for (default: REFERENCE_SECTIONS)
    """
    if sections is None:
        sections = REFERENCE_SECTIONS

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply style
    _apply_style()

    # Generate plots for each section
    for section in sections:
        if section not in SECTION_FUNCS:
            print(f"Warning: Unknown section '{section}', skipping")
            continue

        print(f"Generating reference for: {section}")
        try:
            SECTION_FUNCS[section](assets_dir, force=True)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Copy generated outputs to reference directory
    # We look for common output patterns
    output_patterns = ["*.png", "*.pdf"]
    copied = 0
    for pattern in output_patterns:
        for f in assets_dir.glob(pattern):
            # Only copy files that match our reference sections
            section_match = any(section in f.stem for section in sections)
            if section_match:
                dest = output_dir / f.name
                shutil.copy(f, dest)
                print(f"  Copied: {f.name}")
                copied += 1

    print(f"\nGenerated {copied} reference files in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reference outputs for plotting equivalence tests"
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="Directory containing experiment data (.npy files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/plotting/reference_outputs"),
        help="Directory to store reference outputs",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        help="Sections to generate references for (default: critical sections)",
    )
    args = parser.parse_args()

    if args.sections:
        sections = args.sections
    else:
        sections = REFERENCE_SECTIONS

    generate_references(args.assets_dir, args.output_dir, sections)


if __name__ == "__main__":
    main()
