#!/usr/bin/env python3
"""Validate that our metric implementations match the original PRH code.

This script loads pre-extracted features and computes metrics using both:
1. The original PRH implementation (external_code/platonic-rep/metrics.py)
2. Our implementation (src)

It also tests the effect of outlier removal (q=0.95 vs q=1.0) which is a
critical preprocessing step that significantly affects metric values.

Usage:
    python -m scripts.validation.prh_pipeline
    python -m scripts.validation.prh_pipeline --vision vit_base_patch14_dinov2 --language open_llama_7b
    python -m scripts.validation.prh_pipeline --compare-outlier-removal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for direct script execution
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

# Add original PRH code to path
sys.path.insert(0, str(PROJECT_ROOT / "external_code" / "platonic-rep"))
from metrics import AlignmentMetrics as OriginalMetrics

from aristotelian import cka, mutual_knn
from aristotelian.prh.preprocess import prepare_features


def find_feature_file(results_dir: Path, pattern: str) -> Path | None:
    """Find a feature file matching the pattern."""
    features_dir = results_dir / "features" / "minhuh" / "prh" / "wit_1024"
    for f in features_dir.glob("*.pt"):
        if pattern.replace("-", "_").replace("/", "_") in f.name.replace("-", "_"):
            return f
    return None


def compute_max_metrics(v_feats: torch.Tensor, l_feats: torch.Tensor) -> dict:
    """Find max CKA and kNN over all layer pairs."""
    max_cka, max_cka_idx = 0.0, (0, 0)
    max_knn, max_knn_idx = 0.0, (0, 0)

    n_v_layers = v_feats.shape[1]
    n_l_layers = l_feats.shape[1]

    for i in range(n_v_layers):
        for j in range(n_l_layers):
            v_norm = F.normalize(v_feats[:, i, :], dim=-1)
            l_norm = F.normalize(l_feats[:, j, :], dim=-1)

            c = cka(v_norm, l_norm, kernel_metric="ip")
            k = mutual_knn(v_norm, l_norm, topk=10)

            if c > max_cka:
                max_cka, max_cka_idx = c, (i, j)
            if k > max_knn:
                max_knn, max_knn_idx = k, (i, j)

    return {
        "max_cka": max_cka,
        "max_cka_layers": max_cka_idx,
        "max_knn": max_knn,
        "max_knn_layers": max_knn_idx,
    }


def validate_single_layer(
    v: torch.Tensor, lang_feats: torch.Tensor, layer_desc: str = "last"
) -> dict:
    """Validate metrics on a single layer pair."""
    v_norm = F.normalize(v, dim=-1)
    l_norm = F.normalize(lang_feats, dim=-1)

    orig_cka = OriginalMetrics.cka(v_norm, l_norm, kernel_metric="ip")
    our_cka = cka(v_norm, l_norm, kernel_metric="ip")

    orig_knn = OriginalMetrics.mutual_knn(v_norm, l_norm, topk=10)
    our_knn = mutual_knn(v_norm, l_norm, topk=10)

    return {
        "layer": layer_desc,
        "original_cka": orig_cka,
        "our_cka": our_cka,
        "cka_diff": abs(orig_cka - our_cka),
        "original_knn": orig_knn,
        "our_knn": our_knn,
        "knn_diff": abs(orig_knn - our_knn),
    }


def validate_max_over_layers(v_feats: torch.Tensor, l_feats: torch.Tensor) -> dict:
    """Find max metrics over all layer pairs."""
    max_cka, max_cka_idx = 0.0, (0, 0)
    max_knn, max_knn_idx = 0.0, (0, 0)

    n_v_layers = v_feats.shape[1]
    n_l_layers = l_feats.shape[1]

    for i in range(n_v_layers):
        for j in range(n_l_layers):
            v_norm = F.normalize(v_feats[:, i, :], dim=-1)
            l_norm = F.normalize(l_feats[:, j, :], dim=-1)

            c = OriginalMetrics.cka(v_norm, l_norm, kernel_metric="ip")
            k = OriginalMetrics.mutual_knn(v_norm, l_norm, topk=10)

            if c > max_cka:
                max_cka, max_cka_idx = c, (i, j)
            if k > max_knn:
                max_knn, max_knn_idx = k, (i, j)

    # Verify our implementation at max indices
    v_best = F.normalize(v_feats[:, max_cka_idx[0], :], dim=-1)
    l_best = F.normalize(l_feats[:, max_cka_idx[1], :], dim=-1)
    our_max_cka = cka(v_best, l_best, kernel_metric="ip")

    v_best = F.normalize(v_feats[:, max_knn_idx[0], :], dim=-1)
    l_best = F.normalize(l_feats[:, max_knn_idx[1], :], dim=-1)
    our_max_knn = mutual_knn(v_best, l_best, topk=10)

    return {
        "max_cka": max_cka,
        "max_cka_layers": max_cka_idx,
        "our_max_cka": our_max_cka,
        "cka_diff": abs(max_cka - our_max_cka),
        "max_knn": max_knn,
        "max_knn_layers": max_knn_idx,
        "our_max_knn": our_max_knn,
        "knn_diff": abs(max_knn - our_max_knn),
    }


def compare_outlier_removal(
    v_feats: torch.Tensor, l_feats: torch.Tensor, q: float = 0.95
) -> None:
    """Compare metrics with and without outlier removal."""
    print("\n" + "=" * 70)
    print(f"COMPARISON: q=1.0 (no outlier removal) vs q={q} (experiment default)")
    print("=" * 70)

    # Without outlier removal
    print("\n--- WITHOUT outlier removal (q=1.0) ---")
    result_no_outlier = compute_max_metrics(v_feats, l_feats)
    print(
        f"Max CKA:  {result_no_outlier['max_cka']:.6f} at layers {result_no_outlier['max_cka_layers']}"
    )
    print(
        f"Max kNN:  {result_no_outlier['max_knn']:.6f} at layers {result_no_outlier['max_knn_layers']}"
    )

    # With outlier removal
    print(f"\n--- WITH outlier removal (q={q}) ---")
    v_prep = prepare_features(v_feats, q=q)
    l_prep = prepare_features(l_feats, q=q)
    result_with_outlier = compute_max_metrics(v_prep, l_prep)
    print(
        f"Max CKA:  {result_with_outlier['max_cka']:.6f} at layers {result_with_outlier['max_cka_layers']}"
    )
    print(
        f"Max kNN:  {result_with_outlier['max_knn']:.6f} at layers {result_with_outlier['max_knn_layers']}"
    )

    # Difference
    cka_diff = result_with_outlier["max_cka"] - result_no_outlier["max_cka"]
    knn_diff = result_with_outlier["max_knn"] - result_no_outlier["max_knn"]
    cka_pct = (
        100 * cka_diff / result_no_outlier["max_cka"]
        if result_no_outlier["max_cka"] > 0
        else 0
    )
    knn_pct = (
        100 * knn_diff / result_no_outlier["max_knn"]
        if result_no_outlier["max_knn"] > 0
        else 0
    )

    print("\n--- DIFFERENCE ---")
    print(f"CKA diff:  {cka_diff:+.6f} ({cka_pct:+.1f}%)")
    print(f"kNN diff:  {knn_diff:+.6f} ({knn_pct:+.1f}%)")

    print("\n" + "-" * 70)
    print("NOTE: Our experiment pipeline uses q=0.95 by default (PRH_Q_OUTLIER).")
    print("This explains why raw metric values differ from naive computation.")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate PRH pipeline implementations"
    )
    parser.add_argument(
        "--vision",
        default="vit_small_patch14_dinov2",
        help="Vision model name pattern",
    )
    parser.add_argument(
        "--language",
        default="open_llama_3b",
        help="Language model name pattern",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory",
    )
    parser.add_argument(
        "--skip-max",
        action="store_true",
        help="Skip max-over-layers computation (faster)",
    )
    parser.add_argument(
        "--q-outlier",
        type=float,
        default=0.95,
        help="Quantile for outlier removal (default: 1.0 = no removal, experiment uses 0.95)",
    )
    parser.add_argument(
        "--compare-outlier-removal",
        action="store_true",
        help="Compare metrics with and without outlier removal (q=1.0 vs q=0.95)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("=" * 70)
    print("VALIDATION: Comparing OUR pipeline vs ORIGINAL PRH pipeline")
    print("=" * 70)

    # Find feature files
    vision_file = find_feature_file(results_dir, args.vision)
    language_file = find_feature_file(results_dir, args.language)

    if vision_file is None:
        print(f"ERROR: Could not find vision features matching '{args.vision}'")
        print("Available files:")
        features_dir = results_dir / "features" / "minhuh" / "prh" / "wit_1024"
        for f in sorted(features_dir.glob("vit*.pt"))[:10]:
            print(f"  {f.name}")
        return 1

    if language_file is None:
        print(f"ERROR: Could not find language features matching '{args.language}'")
        print("Available files:")
        features_dir = results_dir / "features" / "minhuh" / "prh" / "wit_1024"
        for f in sorted(features_dir.glob("*llama*.pt"))[:10]:
            print(f"  {f.name}")
        return 1

    print(f"\nVision model:   {vision_file.name}")
    print(f"Language model: {language_file.name}")

    # Load features
    print("\nLoading features...")
    v_data = torch.load(vision_file, map_location="cpu", weights_only=False)
    l_data = torch.load(language_file, map_location="cpu", weights_only=False)

    v_feats = v_data["feats"]
    l_feats = l_data["feats"]

    print(f"Vision features:   {v_feats.shape}")
    print(f"Language features: {l_feats.shape}")
    print(
        f"Samples: {v_feats.shape[0]}, "
        f"Vision layers: {v_feats.shape[1]}, "
        f"Language layers: {l_feats.shape[1]}"
    )

    # Apply outlier removal if requested
    if args.q_outlier < 1.0:
        print(f"\nApplying outlier removal with q={args.q_outlier}...")
        v_feats = prepare_features(v_feats, q=args.q_outlier)
        l_feats = prepare_features(l_feats, q=args.q_outlier)

    # Test 1: Single layer (last)
    print("\n" + "-" * 70)
    print("TEST 1: Single layer comparison (last layer of each model)")
    print("-" * 70)

    result = validate_single_layer(v_feats[:, -1, :], l_feats[:, -1, :], "last")
    print(f"Original PRH CKA: {result['original_cka']:.6f}")
    print(f"Our CKA:          {result['our_cka']:.6f}")
    print(f"Difference:       {result['cka_diff']:.2e}")
    print()
    print(f"Original PRH kNN: {result['original_knn']:.6f}")
    print(f"Our kNN:          {result['our_knn']:.6f}")
    print(f"Difference:       {result['knn_diff']:.2e}")

    cka_match = result["cka_diff"] < 1e-5
    knn_match = result["knn_diff"] < 1e-5

    # Test 2: Max over layers
    if not args.skip_max:
        print("\n" + "-" * 70)
        print("TEST 2: Max over all layer pairs (as PRH paper reports)")
        print("-" * 70)

        print("Scanning all layer pairs...")
        result = validate_max_over_layers(v_feats, l_feats)

        print(
            f"Max CKA (original): {result['max_cka']:.6f} "
            f"at layers {result['max_cka_layers']}"
        )
        print(f"Our CKA:            {result['our_max_cka']:.6f}")
        print(f"Difference:         {result['cka_diff']:.2e}")
        print()
        print(
            f"Max kNN (original): {result['max_knn']:.6f} "
            f"at layers {result['max_knn_layers']}"
        )
        print(f"Our kNN:            {result['our_max_knn']:.6f}")
        print(f"Difference:         {result['knn_diff']:.2e}")

        cka_match = cka_match and result["cka_diff"] < 1e-5
        knn_match = knn_match and result["knn_diff"] < 1e-5

    # Summary
    print("\n" + "=" * 70)
    if cka_match and knn_match:
        print("VALIDATION PASSED - All numbers match to ~1e-5 precision")
    else:
        print("VALIDATION FAILED - Some differences exceed 1e-5")
    print("=" * 70)

    # Outlier removal comparison
    if args.compare_outlier_removal:
        # Reload original features for fair comparison
        v_feats_orig = v_data["feats"]
        l_feats_orig = l_data["feats"]
        compare_outlier_removal(v_feats_orig, l_feats_orig, q=0.95)

    return 0 if (cka_match and knn_match) else 1


if __name__ == "__main__":
    sys.exit(main())
