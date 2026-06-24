"""Calibration vs. analytical debiasing comparison (paper Fig. 12 / fig:cka-comparison).

Canonical generator for ``assets/agreement_summary.pdf``. Computes the difference
between our calibrated CKA and two analytical estimators under H1 (signal) and
H0 (null): the debiased CKA of Song et al. (2012) -- using the unbiased-HSIC
("Song") denominator, see ``aristotelian.metrics.estimators.cka_debiased`` -- and
the dep-cols CKA of Chun et al. (2025).

Usage:
    python -m scripts.analyses.cka_debiasing_comparison --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from aristotelian import cka_debiased, cka_depcols, sg_cka_linear
from scripts.experiments.generators import make_gen2_linear_signal, make_pure_noise


def run_fig12_experiment(
    n: int = 1024,
    d_list: list[int] | None = None,
    signal_rank: int = 10,
    signal_strength: float = 2.0,
    num_trials: int = 100,
    num_permutations: int = 200,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Run the Figure 12 experiment: calibrated vs analytical CKA estimators.

    For each d/n ratio, compute calibrated CKA, debiased CKA, and dep-cols CKA
    under both H1 (shared low-rank signal) and H0 (independent random matrices).
    """
    if d_list is None:
        # Match d/n range in paper: ~2^{-2} to 2^3
        d_list = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]

    torch.manual_seed(seed)
    np.random.seed(seed)

    ratio_list = [d / n for d in d_list]
    num_d = len(d_list)

    results = {
        # H1 (signal)
        "h1_gated": np.zeros(num_d),
        "h1_debiased": np.zeros(num_d),
        "h1_depcols": np.zeros(num_d),
        "h1_gated_std": np.zeros(num_d),
        "h1_debiased_std": np.zeros(num_d),
        "h1_depcols_std": np.zeros(num_d),
        # H0 (null)
        "h0_gated": np.zeros(num_d),
        "h0_debiased": np.zeros(num_d),
        "h0_depcols": np.zeros(num_d),
        "h0_gated_std": np.zeros(num_d),
        "h0_debiased_std": np.zeros(num_d),
        "h0_depcols_std": np.zeros(num_d),
    }

    total = 2 * num_d  # H1 + H0
    def _run_h1_trial(dev):
        X, Y, _ = make_gen2_linear_signal(
            n=n, d=d, rank=signal_rank,
            signal_strength=signal_strength, noise_std=1.0,
            noise_type="gaussian", device=dev,
        )
        deb = cka_debiased(X, Y)
        dep = cka_depcols(X, Y)
        res = sg_cka_linear(X, Y, num_permutations=num_permutations, device=dev)
        return deb, dep, res.gated

    def _run_h0_trial(dev):
        X, Y = make_pure_noise(n, d, device=dev)
        deb = cka_debiased(X, Y)
        dep = cka_depcols(X, Y)
        res = sg_cka_linear(X, Y, num_permutations=num_permutations, device=dev)
        return deb, dep, res.gated

    def _try_trial(fn, dev):
        """Run a full trial on dev; on CUDA OOM, retry on CPU."""
        try:
            return fn(dev)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return fn("cpu")

    with tqdm(total=total, desc="Fig 12") as pbar:
        for j, d in enumerate(d_list):
            # --- H1: Signal ---
            gated_vals, debiased_vals, depcols_vals = [], [], []
            for _ in range(num_trials):
                deb, dep, gated = _try_trial(_run_h1_trial, device)
                debiased_vals.append(deb)
                depcols_vals.append(dep)
                gated_vals.append(gated)

            results["h1_gated"][j] = np.mean(gated_vals)
            results["h1_debiased"][j] = np.mean(debiased_vals)
            results["h1_depcols"][j] = np.mean(depcols_vals)
            results["h1_gated_std"][j] = np.std(gated_vals)
            results["h1_debiased_std"][j] = np.std(debiased_vals)
            results["h1_depcols_std"][j] = np.std(depcols_vals)
            pbar.update(1)

            # --- H0: Null ---
            gated_vals, debiased_vals, depcols_vals = [], [], []
            for _ in range(num_trials):
                deb, dep, gated = _try_trial(_run_h0_trial, device)
                debiased_vals.append(deb)
                depcols_vals.append(dep)
                gated_vals.append(gated)

            results["h0_gated"][j] = np.mean(gated_vals)
            results["h0_debiased"][j] = np.mean(debiased_vals)
            results["h0_depcols"][j] = np.mean(depcols_vals)
            results["h0_gated_std"][j] = np.std(gated_vals)
            results["h0_debiased_std"][j] = np.std(debiased_vals)
            results["h0_depcols_std"][j] = np.std(depcols_vals)
            pbar.update(1)

    results["n"] = n
    results["d_list"] = np.array(d_list)
    results["ratio_list"] = np.array(ratio_list)
    results["signal_rank"] = signal_rank
    results["signal_strength"] = signal_strength
    results["num_trials"] = num_trials

    return results


def plot_fig12(results: dict, output_path: Path) -> None:
    """Plot Figure 12: Calibration recovers analytical debiasing."""
    import matplotlib.pyplot as plt
    from aristotelian.style import LINE_COLORS, prh_legend, style_line_axes, use_prh_base

    style_path = Path(__file__).parent.parent.parent / "styles" / "paper.mplstyle"
    use_prh_base(style_path)

    ratio_list = results["ratio_list"]
    num_trials = results["num_trials"]

    color_debiased = LINE_COLORS[0]  # Teal
    color_depcols = LINE_COLORS[2]   # Purple

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.5))

    # --- Left: Under Signal (H1) ---
    style_line_axes(ax1)
    diff_deb = results["h1_gated"] - results["h1_debiased"]
    diff_dep = results["h1_gated"] - results["h1_depcols"]

    # SE for difference (propagated)
    se_deb = np.sqrt(results["h1_gated_std"]**2 + results["h1_debiased_std"]**2) / np.sqrt(num_trials)
    se_dep = np.sqrt(results["h1_gated_std"]**2 + results["h1_depcols_std"]**2) / np.sqrt(num_trials)

    ax1.plot(ratio_list, diff_deb, "o-", color=color_debiased,
             label="Calibrated $-$ Debiased", linewidth=1.5, markersize=4)
    ax1.fill_between(ratio_list, diff_deb - se_deb, diff_deb + se_deb,
                     color=color_debiased, alpha=0.2)
    ax1.plot(ratio_list, diff_dep, "s-", color=color_depcols,
             label="Calibrated $-$ Dep-cols", linewidth=1.5, markersize=4)
    ax1.fill_between(ratio_list, diff_dep - se_dep, diff_dep + se_dep,
                     color=color_depcols, alpha=0.2)

    ax1.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel(r"$d/n$")
    ax1.set_ylabel("Difference under signal")
    ax1.set_xscale("log", base=2)
    prh_legend(ax1, style="square", fontsize=7)

    # --- Right: Under Null (H0) ---
    style_line_axes(ax2)
    diff_deb_null = results["h0_gated"] - results["h0_debiased"]
    diff_dep_null = results["h0_gated"] - results["h0_depcols"]

    se_deb_null = np.sqrt(results["h0_gated_std"]**2 + results["h0_debiased_std"]**2) / np.sqrt(num_trials)
    se_dep_null = np.sqrt(results["h0_gated_std"]**2 + results["h0_depcols_std"]**2) / np.sqrt(num_trials)

    ax2.plot(ratio_list, diff_deb_null, "o-", color=color_debiased,
             label="Calibrated $-$ Debiased", linewidth=1.5, markersize=4)
    ax2.fill_between(ratio_list, diff_deb_null - se_deb_null, diff_deb_null + se_deb_null,
                     color=color_debiased, alpha=0.2)
    ax2.plot(ratio_list, diff_dep_null, "s-", color=color_depcols,
             label="Calibrated $-$ Dep-cols", linewidth=1.5, markersize=4)
    ax2.fill_between(ratio_list, diff_dep_null - se_dep_null, diff_dep_null + se_dep_null,
                     color=color_depcols, alpha=0.2)

    ax2.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel(r"$d/n$")
    ax2.set_ylabel("Difference under null")
    ax2.set_xscale("log", base=2)
    prh_legend(ax2, style="square", fontsize=7)

    # Match x-axis
    xlim = (min(ratio_list) * 0.8, max(ratio_list) * 1.2)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)

    plt.tight_layout()
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibration vs. analytical debiasing comparison (paper Fig. 12)"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--num-permutations", type=int, default=200)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default="./assets")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_fig12_experiment(
        n=args.n,
        num_trials=args.num_trials,
        num_permutations=args.num_permutations,
        device=args.device,
        seed=args.seed,
    )

    # Save raw data
    np.save(output_dir / "agreement_summary_data.npy", results)
    print(f"Saved data to {output_dir / 'agreement_summary_data.npy'}")

    # Generate plot (paper Fig. 12 / fig:cka-comparison)
    plot_fig12(results, output_dir / "agreement_summary.pdf")


if __name__ == "__main__":
    main()
