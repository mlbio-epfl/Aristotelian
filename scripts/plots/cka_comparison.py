"""Plotting functions for CKA Estimator Comparison Experiments.

Generates publication-quality figures comparing:
1. Biased CKA
2. Debiased/Song CKA
3. Dependent-columns CKA
4. Signal-gated CKA (our approach)

Usage:
    python -m scripts.plots.cka_comparison --results-dir ./results/cka_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import PRH style system
from aristotelian.style import (
    LINE_COLORS,
    SEQUENTIAL,
    prh_legend,
    style_heatmap_axes,
    style_line_axes,
    use_prh_base,
)

# Color scheme using PRH palette
COLORS = {
    "biased": LINE_COLORS[1],  # Orange-red
    "debiased": LINE_COLORS[0],  # Teal
    "depcols": LINE_COLORS[2],  # Purple
    "gated": LINE_COLORS[3],  # Green
}
DIVERGING = "RdBu_r"

LABELS = {
    "biased": "Biased CKA",
    "debiased": "Debiased CKA",
    "depcols": "Dep-cols CKA",
    "gated": "Signal-gated CKA",
}

# Labels with citations for main comparison plots
LABELS_CITE = {
    "biased": "Biased CKA",
    "debiased": "Debiased CKA [Murphy+ '24]",
    "depcols": "Dep-cols CKA [Chun+ '25]",
    "gated": "Signal-gated CKA [ours]",
}

ESTIMATORS = ["biased", "debiased", "depcols", "gated"]
OVERLAP_ESTIMATORS = ["debiased", "depcols", "gated"]
MARKERS = {"debiased": "o", "depcols": "s", "gated": "^"}


def _apply_style() -> None:
    """Apply PRH plotting style."""
    use_prh_base(Path(__file__).parent.parent.parent / "styles" / "paper.mplstyle")


def _save_fig(path: Path) -> None:
    """Save figure as PDF."""
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _diverging_vlim(data: np.ndarray, percentile: float = 99.0) -> float:
    """Compute symmetric color limits for diverging heatmaps."""
    max_abs = float(np.nanpercentile(np.abs(data), percentile))
    return max(max_abs, 1e-6)


# =============================================================================
# Plot 1: Null Drift Heatmaps
# =============================================================================


def plot_null_drift(results: dict, output_path: Path) -> None:
    """Plot null drift experiment as heatmaps."""
    n_list = results["n_list"]
    d_list = results["d_list"]

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.5))

    vmin, vmax = -0.1, 1.0
    im = None

    for ax, est in zip(axes.flat, ESTIMATORS):
        style_heatmap_axes(ax)
        data = results[est]

        im = ax.imshow(
            data,
            aspect="auto",
            cmap=SEQUENTIAL,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )

        ax.set_xticks(range(len(d_list)))
        ax.set_xticklabels(d_list)
        ax.set_yticks(range(len(n_list)))
        ax.set_yticklabels(n_list)
        ax.set_xlabel("$d$")
        ax.set_ylabel("$n$")
        ax.set_title(LABELS[est])

        # Add text annotations
        for i in range(len(n_list)):
            for j in range(len(d_list)):
                val = data[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="CKA score")

    _save_fig(output_path)
    print(f"Saved: {output_path}")


def plot_null_drift_lines(results: dict, output_path: Path) -> None:
    """Plot null drift as line plots for fixed n, varying d."""
    n_list = results["n_list"]
    d_list = results["d_list"]

    fig, axes = plt.subplots(
        1, len(n_list), figsize=(3.25 * len(n_list), 2.5), sharey=True
    )
    if len(n_list) == 1:
        axes = [axes]

    for ax, (i, n) in zip(axes, enumerate(n_list)):
        style_line_axes(ax)
        for est in ESTIMATORS:
            means = results[est][i, :]
            stds = results.get(f"{est}_std", np.zeros_like(means))[i, :]

            ax.plot(
                d_list,
                means,
                "o-",
                color=COLORS[est],
                label=LABELS[est],
                linewidth=1.5,
            )
            ax.fill_between(
                d_list,
                means - stds,
                means + stds,
                color=COLORS[est],
                alpha=0.25,
            )

        ax.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("$d$")
        ax.set_title(f"$n = {n}$")
        ax.set_xscale("log", base=2)

    axes[0].set_ylabel("CKA score")
    prh_legend(axes[0], loc="upper left", style="square")

    _save_fig(output_path)
    print(f"Saved: {output_path}")


def plot_null_drift_overlap(results: dict, output_path: Path) -> None:
    """Plot null drift overlap for debiased/depcols/gated (zoomed)."""
    n_list = results["n_list"]
    d_list = results["d_list"]

    fig, axes = plt.subplots(
        1, len(n_list), figsize=(3.25 * len(n_list), 2.5), sharey=True
    )
    if len(n_list) == 1:
        axes = [axes]

    all_vals = np.concatenate([results[est].ravel() for est in OVERLAP_ESTIMATORS])
    max_abs = float(np.max(np.abs(all_vals)))
    ylim = max(0.01, max_abs * 1.25)

    for ax, (i, n) in zip(axes, enumerate(n_list)):
        style_line_axes(ax)
        for est in OVERLAP_ESTIMATORS:
            means = results[est][i, :]
            markerface = "none" if est == "gated" else COLORS[est]
            ax.plot(
                d_list,
                means,
                marker=MARKERS[est],
                color=COLORS[est],
                label=LABELS[est],
                linewidth=1.5,
                markerfacecolor=markerface,
            )

        ax.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("$d$")
        ax.set_title(f"$n = {n}$")
        ax.set_xscale("log", base=2)
        ax.set_ylim(-ylim, ylim)

    axes[0].set_ylabel("CKA score (zoom)")
    prh_legend(axes[0], loc="upper left", style="square")

    _save_fig(output_path)
    print(f"Saved: {output_path}")


def plot_null_drift_deltas(results: dict, output_path: Path) -> None:
    """Plot gated minus analytic corrections as heatmaps."""
    n_list = results["n_list"]
    d_list = results["d_list"]

    diff_debiased = results["gated"] - results["debiased"]
    diff_depcols = results["gated"] - results["depcols"]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))

    for ax, diff, title in zip(
        axes,
        [diff_debiased, diff_depcols],
        ["Gated - Debiased", "Gated - Dep-cols"],
    ):
        style_heatmap_axes(ax)
        vlim = _diverging_vlim(diff, percentile=99.0)
        im = ax.imshow(
            diff,
            aspect="auto",
            cmap=DIVERGING,
            vmin=-vlim,
            vmax=vlim,
            origin="lower",
        )
        ax.set_xticks(range(len(d_list)))
        ax.set_xticklabels(d_list)
        ax.set_yticks(range(len(n_list)))
        ax.set_yticklabels(n_list)
        ax.set_xlabel("$d$")
        ax.set_ylabel("$n$")
        ax.set_title(title)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Difference")

    _save_fig(output_path)
    print(f"Saved: {output_path}")


# =============================================================================
# Plot 2: High d/n Ratio Sweep
# =============================================================================


def plot_high_dn_ratio(results: dict, output_path: Path) -> None:
    """Plot high d/n ratio experiment."""
    ratio_list = results["ratio_list"]
    _n = results["n"]  # noqa: F841 - kept for reference
    # high_dn uses num_trials * 2 = 200 trials
    num_trials = results.get("num_trials", 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.5))

    # Left: CKA scores
    style_line_axes(ax1)
    for est in ESTIMATORS:
        means = results[f"{est}_mean"]
        stds = results[f"{est}_std"]
        # Convert std to standard error
        se = stds / np.sqrt(num_trials)

        ax1.plot(
            ratio_list,
            means,
            "o-",
            color=COLORS[est],
            label=LABELS_CITE[est],
            linewidth=1.5,
        )
        ax1.fill_between(
            ratio_list,
            means - se,
            means + se,
            color=COLORS[est],
            alpha=0.25,
        )

    ax1.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax1.axhline(1, color="#888888", linestyle=":", linewidth=1, alpha=0.7)
    ax1.set_xlabel(r"$d/n$")
    ax1.set_ylabel("CKA score")
    ax1.set_xscale("log", base=2)
    prh_legend(ax1, style="square")
    ax1.set_ylim(-0.2, 1.1)

    # Right: Rejection rate
    style_line_axes(ax2)
    ax2.plot(
        ratio_list,
        results["rejection_rate"],
        "o-",
        color=COLORS["gated"],
        linewidth=1.5,
        label="Rejection rate",
    )
    ax2.axhline(
        0.05,
        color=LINE_COLORS[1],
        linestyle="--",
        linewidth=1.5,
        label=r"$\alpha$ = 0.05",
    )
    ax2.set_xlabel(r"$d/n$")
    ax2.set_ylabel("Rejection rate")
    ax2.set_xscale("log", base=2)
    prh_legend(ax2, style="square")
    ax2.set_ylim(0, 0.3)

    _save_fig(output_path)
    print(f"Saved: {output_path}")


def plot_high_dn_overlap(results: dict, output_path: Path) -> None:
    """Plot high d/n overlap for debiased/depcols/gated (zoomed)."""
    ratio_list = results["ratio_list"]

    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.5))
    style_line_axes(ax)

    vals = []
    for est in OVERLAP_ESTIMATORS:
        means = results[f"{est}_mean"]
        vals.append(means)
        markerface = "none" if est == "gated" else COLORS[est]
        ax.plot(
            ratio_list,
            means,
            marker=MARKERS[est],
            color=COLORS[est],
            label=LABELS[est],
            linewidth=1.5,
            markerfacecolor=markerface,
        )

    all_vals = np.concatenate(vals)
    max_abs = float(np.max(np.abs(all_vals)))
    ylim = max(0.01, max_abs * 1.25)

    ax.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"$d/n$")
    ax.set_ylabel("CKA score (zoom)")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-ylim, ylim)
    prh_legend(ax, style="square")

    _save_fig(output_path)
    print(f"Saved: {output_path}")


def plot_high_dn_deltas(results: dict, output_path: Path) -> None:
    """Plot gated minus analytic corrections for high d/n sweep."""
    ratio_list = results["ratio_list"]
    diff_debiased = results["gated_mean"] - results["debiased_mean"]
    diff_depcols = results["gated_mean"] - results["depcols_mean"]

    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.5))
    style_line_axes(ax)

    ax.plot(
        ratio_list,
        diff_debiased,
        marker="o",
        color=COLORS["debiased"],
        label="Gated - Debiased",
        linewidth=1.5,
    )
    ax.plot(
        ratio_list,
        diff_depcols,
        marker="s",
        color=COLORS["depcols"],
        label="Gated - Dep-cols",
        linewidth=1.5,
    )

    max_abs = float(np.max(np.abs(np.concatenate([diff_debiased, diff_depcols]))))
    ylim = max(0.01, max_abs * 1.25)

    ax.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"$d/n$")
    ax.set_ylabel("Difference")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-ylim, ylim)
    prh_legend(ax, style="square")

    _save_fig(output_path)
    print(f"Saved: {output_path}")


# =============================================================================
# Plot 3: Dimension Padding with Signal
# =============================================================================


def plot_dim_padding(results: dict, output_path: Path) -> None:
    """Plot dimension padding experiment."""
    d_list = results["d_list"]
    snr_list = results["snr_list"]

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.0))

    for ax, (i, snr) in zip(axes.flat, enumerate(snr_list)):
        style_line_axes(ax)
        for est in ESTIMATORS:
            vals = results[est][i, :]
            ax.plot(
                d_list,
                vals,
                "o-",
                color=COLORS[est],
                label=LABELS[est],
                linewidth=1.5,
            )

        ax.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("$d$")
        ax.set_ylabel("CKA score")
        ax.set_title(f"SNR = {snr}" + " (null)" if snr == 0 else f"SNR = {snr}")
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.1, 1.05)

    prh_legend(axes[0, 0], loc="upper left", style="square")

    _save_fig(output_path)
    print(f"Saved: {output_path}")


def plot_dim_padding_rejection(results: dict, output_path: Path) -> None:
    """Plot rejection rates for dimension padding experiment."""
    d_list = results["d_list"]
    snr_list = results["snr_list"]

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    style_line_axes(ax)

    for i, snr in enumerate(snr_list):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ax.plot(
            d_list,
            results["rejection_rate"][i, :],
            "o-",
            color=color,
            label=f"SNR={snr}",
            linewidth=1.5,
        )

    ax.axhline(
        0.05, color="#888888", linestyle="--", linewidth=1.5, label=r"$\alpha$ = 0.05"
    )
    ax.set_xlabel("$d$")
    ax.set_ylabel("Rejection rate")
    ax.set_title("Type I error by SNR")
    ax.set_xscale("log", base=2)
    prh_legend(ax, style="square")
    ax.set_ylim(0, 1.1)

    _save_fig(output_path)
    print(f"Saved: {output_path}")


def plot_dim_padding_overlap(results: dict, output_path: Path) -> None:
    """Plot overlap for low-SNR settings (zoomed)."""
    d_list = results["d_list"]
    snr_list = results["snr_list"]
    snr_indices = [0, 1] if len(snr_list) > 1 else [0]

    fig, axes = plt.subplots(
        1, len(snr_indices), figsize=(3.25 * len(snr_indices), 2.5)
    )
    if len(snr_indices) == 1:
        axes = [axes]

    all_vals = []
    for idx in snr_indices:
        for est in OVERLAP_ESTIMATORS:
            all_vals.append(results[est][idx, :])
    all_vals = np.concatenate(all_vals)
    max_abs = float(np.max(np.abs(all_vals)))
    ylim = max(0.01, max_abs * 1.25)

    for ax, idx in zip(axes, snr_indices):
        style_line_axes(ax)
        snr = snr_list[idx]
        for est in OVERLAP_ESTIMATORS:
            vals = results[est][idx, :]
            markerface = "none" if est == "gated" else COLORS[est]
            ax.plot(
                d_list,
                vals,
                marker=MARKERS[est],
                color=COLORS[est],
                label=LABELS[est],
                linewidth=1.5,
                markerfacecolor=markerface,
            )
        ax.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("$d$")
        ax.set_title(f"SNR = {snr}")
        ax.set_xscale("log", base=2)
        ax.set_ylim(-ylim, ylim)

    axes[0].set_ylabel("CKA score (zoom)")
    prh_legend(axes[0], loc="upper left", style="square")

    _save_fig(output_path)
    print(f"Saved: {output_path}")


def plot_dim_padding_deltas(results: dict, output_path: Path) -> None:
    """Plot gated minus analytic corrections across SNR and d."""
    d_list = results["d_list"]
    snr_list = results["snr_list"]

    diff_debiased = results["gated"] - results["debiased"]
    diff_depcols = results["gated"] - results["depcols"]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))

    vlim_deb = _diverging_vlim(diff_debiased, percentile=99.0)
    vlim_dep = _diverging_vlim(diff_depcols, percentile=95.0)

    for ax, diff, vlim, title in zip(
        axes,
        [diff_debiased, diff_depcols],
        [vlim_deb, vlim_dep],
        ["Gated - Debiased", "Gated - Dep-cols (clipped)"],
    ):
        style_heatmap_axes(ax)
        im = ax.imshow(
            diff,
            aspect="auto",
            cmap=DIVERGING,
            vmin=-vlim,
            vmax=vlim,
            origin="lower",
        )
        ax.set_xticks(range(len(d_list)))
        ax.set_xticklabels(d_list)
        ax.set_yticks(range(len(snr_list)))
        ax.set_yticklabels(snr_list)
        ax.set_xlabel("$d$")
        ax.set_ylabel("SNR")
        ax.set_title(title)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Difference")

    _save_fig(output_path)
    print(f"Saved: {output_path}")


# =============================================================================
# Combined Summary Plot
# =============================================================================


def plot_summary(
    null_drift: dict | None,
    high_dn: dict | None,
    dim_padding: dict | None,
    output_path: Path,
) -> None:
    """Create a summary figure combining key results from all experiments."""
    num_plots = sum(
        [null_drift is not None, high_dn is not None, dim_padding is not None]
    )
    if num_plots == 0:
        print("No results to plot for summary")
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(3.25 * num_plots, 2.5))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # High d/n ratio (most important result)
    if high_dn is not None:
        ax = axes[plot_idx]
        style_line_axes(ax)
        ratio_list = high_dn["ratio_list"]
        # Get num_trials for SE calculation (default 200 if not stored)
        num_trials = high_dn.get("num_trials", 200)

        for est in ESTIMATORS:
            means = high_dn[f"{est}_mean"]
            stds = high_dn.get(f"{est}_std", np.zeros_like(means))
            # Convert std to standard error
            se = stds / np.sqrt(num_trials)

            ax.plot(
                ratio_list,
                means,
                "o-",
                color=COLORS[est],
                label=LABELS[est],
                linewidth=1.5,
                markersize=4,
            )
            ax.fill_between(
                ratio_list,
                means - se,
                means + se,
                color=COLORS[est],
                alpha=0.2,
            )

        ax.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(r"$d/n$")
        ax.set_ylabel("CKA score")
        ax.set_title("(a) Null hypothesis")
        ax.set_xscale("log", base=2)
        prh_legend(ax, style="square", fontsize=6)
        ax.set_ylim(-0.2, 1.1)
        plot_idx += 1

    # Null drift (pick highest d/n slice)
    if null_drift is not None:
        ax = axes[plot_idx]
        style_line_axes(ax)
        n_list = null_drift["n_list"]
        d_list = null_drift["d_list"]

        j = len(d_list) - 1
        for est in ESTIMATORS:
            vals = null_drift[est][:, j]
            stds = null_drift.get(f"{est}_std", np.zeros_like(vals))[:, j]

            ax.plot(
                n_list,
                vals,
                "o-",
                color=COLORS[est],
                label=LABELS[est],
                linewidth=1.5,
                markersize=4,
            )
            ax.fill_between(
                n_list,
                vals - stds,
                vals + stds,
                color=COLORS[est],
                alpha=0.2,
            )

        ax.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("$n$")
        ax.set_ylabel("CKA score")
        ax.set_title(f"(b) Null drift ($d$={d_list[j]})")
        prh_legend(ax, style="square", fontsize=6)
        ax.set_ylim(-0.2, 1.1)
        plot_idx += 1

    # Dim padding (SNR=0 vs SNR>0)
    if dim_padding is not None:
        ax = axes[plot_idx]
        style_line_axes(ax)
        d_list = dim_padding["d_list"]
        snr_list = dim_padding["snr_list"]

        snr_0_idx = 0
        snr_high_idx = len(snr_list) - 1

        for est in ESTIMATORS:
            vals_0 = dim_padding[est][snr_0_idx, :]
            vals_high = dim_padding[est][snr_high_idx, :]

            ax.plot(d_list, vals_0, "--", color=COLORS[est], alpha=0.5, linewidth=1.5)
            ax.plot(
                d_list,
                vals_high,
                "-",
                color=COLORS[est],
                label=LABELS[est],
                linewidth=1.5,
            )

        ax.axhline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("$d$")
        ax.set_ylabel("CKA score")
        ax.set_title(
            f"(c) Signal detection\n(dashed: null, solid: SNR={snr_list[snr_high_idx]})"
        )
        ax.set_xscale("log", base=2)
        prh_legend(ax, style="square", fontsize=6)
        ax.set_ylim(-0.2, 1.1)

    _save_fig(output_path)
    print(f"Saved: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Plot CKA comparison results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/cka_comparison",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results-dir)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()

    # Load results
    null_drift = None
    high_dn = None
    dim_padding = None

    null_drift_path = results_dir / "null_drift_cka_comparison.npy"
    if null_drift_path.exists():
        null_drift = np.load(null_drift_path, allow_pickle=True).item()
        plot_null_drift(null_drift, output_dir / "null_drift_heatmaps.pdf")
        plot_null_drift_lines(null_drift, output_dir / "null_drift_lines.pdf")
        plot_null_drift_overlap(null_drift, output_dir / "null_drift_overlap.pdf")
        plot_null_drift_deltas(null_drift, output_dir / "null_drift_deltas.pdf")

    high_dn_path = results_dir / "high_dn_cka_comparison.npy"
    if high_dn_path.exists():
        high_dn = np.load(high_dn_path, allow_pickle=True).item()
        plot_high_dn_ratio(high_dn, output_dir / "high_dn_ratio.pdf")
        plot_high_dn_overlap(high_dn, output_dir / "high_dn_overlap.pdf")
        plot_high_dn_deltas(high_dn, output_dir / "high_dn_deltas.pdf")

    dim_padding_path = results_dir / "dim_padding_cka_comparison.npy"
    if dim_padding_path.exists():
        dim_padding = np.load(dim_padding_path, allow_pickle=True).item()
        plot_dim_padding(dim_padding, output_dir / "dim_padding.pdf")
        plot_dim_padding_rejection(
            dim_padding, output_dir / "dim_padding_rejection.pdf"
        )
        plot_dim_padding_overlap(dim_padding, output_dir / "dim_padding_overlap.pdf")
        plot_dim_padding_deltas(dim_padding, output_dir / "dim_padding_deltas.pdf")

    # Summary plot
    if any([null_drift, high_dn, dim_padding]):
        plot_summary(null_drift, high_dn, dim_padding, output_dir / "summary.pdf")

    # NOTE: Fig. 12 (agreement_summary.pdf, the calibration-vs-analytical-debiasing
    # comparison) is now generated by scripts/analyses/cka_debiasing_comparison.py,
    # which uses the corrected Song-denominator debiased-CKA estimator
    # (aristotelian.metrics.estimators.cka_debiased). The earlier plot_agreement_summary
    # path (which reused high_dn/dim_padding data) is retired and no longer run.

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
