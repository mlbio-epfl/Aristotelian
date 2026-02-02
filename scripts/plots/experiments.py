#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Import PRH style system
from aristotelian.style import (
    DIVERGING,
    LINE_COLORS,
    SEQUENTIAL,
    get_fig3_colors,
    prh_colorbar,
    prh_legend,
    style_heatmap_axes,
    style_line_axes,
    use_prh_base,
)


def _apply_style() -> None:
    """Apply PRH (Platonic Representation Hypothesis) plotting style for ICML paper.

    This applies the sophisticated PRH aesthetic adapted for ICML format:
    - Cool-gray background (#F8F9FA) for professional appearance
    - Subtle major grid to aid readability
    - Blue-gray accent colors for spines, ticks, and labels
    - Thicker lines and larger markers for clarity
    - Professional legends with shadows

    Figure size guidelines for ICML:
    - Single column width: 3.25 inches
    - Double column width: 6.75 inches
    - Use figsize=(3.25, 2.5) for single column
    - Use figsize=(6.75, 2.5) for double column
    """
    # Apply PRH base style with ICML adaptations
    use_prh_base(Path(__file__).parent.parent.parent / "styles" / "paper.mplstyle")


def _should_skip(outputs: Sequence[Path], force: bool) -> bool:
    if force:
        return False
    normalized = []
    for p in outputs:
        if p.suffix.lower() != ".pdf":
            normalized.append(p.with_suffix(".pdf"))
        normalized.append(p)
    return all(p.exists() for p in normalized)


NOISE_TYPES = ("gaussian", "student_t", "laplace", "mixture")
DEFAULT_NOISE_TYPE = "gaussian"

# Metrics to exclude from main paper plots (keep in appendix)
APPENDIX_ONLY_METRICS = {
    "CKA (RBF)",
    "RSA",
    "RV",
    "Procrustes",
    "rsa",
    "rv",
    "procrustes",
}

# Metrics to include in main paper null drift plots (in display order)
MAIN_PAPER_NULL_DRIFT_METRICS = ["CKA (lin)", "kNN", "CCA"]

# Label overrides for main paper null drift plots
MAIN_PAPER_NULL_DRIFT_LABELS = {
    "CKA (lin)": "CKA",
    "kNN": "mKNN",
}

# Vision model families for main paper PRH plots (others go to appendix)
MAIN_VISION_FAMILIES = {"DINOv2", "CLIP"}

# Metric label prettifier for publication-quality plots
METRIC_LABELS = {
    # Code-style -> Pretty label
    "sgcka_lin": "CKA (linear)",
    "sgcka_rbf": "CKA (RBF)",
    "sgknn": "mKNN",
    "sgrsa": "RSA",
    "sgcca": "CCA",
    "sgsvcca": "SVCCA",
    "sgpwcca": "PWCCA",
    "sgrv": "RV",
    "sgprocrustes": "Procrustes",
    # Also handle variants without sg prefix
    "cka_lin": "CKA (linear)",
    "cka_rbf": "CKA (RBF)",
    "cka": "CKA",
    "knn": "mKNN",
    "rsa": "RSA",
    "cca": "CCA",
    "svcca": "SVCCA",
    "pwcca": "PWCCA",
    "rv": "RV",
    "procrustes": "Procrustes",
    "mutual_knn": "mKNN",
    "cycle_knn": "Cycle kNN",
    "cknna": "CKNNA",
    "unbiased_cka": "Unbiased CKA",
}


def _prettify_metric(name: str) -> str:
    """Convert code-style metric name to publication-quality label."""
    # Try exact match first
    if name in METRIC_LABELS:
        return METRIC_LABELS[name]
    # Try lowercase match
    lower = name.lower().replace("-", "_").replace(" ", "_")
    if lower in METRIC_LABELS:
        return METRIC_LABELS[lower]
    # Return original if no match
    return name


def _noise_variants(assets_dir: Path, base_name: str) -> list[tuple[str, Path]]:
    base = assets_dir / base_name
    variants: list[tuple[str, Path]] = []
    if base.exists():
        variants.append((DEFAULT_NOISE_TYPE, base))
    for noise_type in NOISE_TYPES:
        if noise_type == DEFAULT_NOISE_TYPE:
            continue
        candidate = assets_dir / f"{base.stem}_{noise_type}{base.suffix}"
        if candidate.exists():
            variants.append((noise_type, candidate))
    return variants


def _noise_suffix(noise_type: str) -> str:
    return "" if noise_type == DEFAULT_NOISE_TYPE else f"_{noise_type}"


def _slug(text: str) -> str:
    return (
        text.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def _order_metrics(metrics: Sequence[str]) -> list[str]:
    preferred = [
        "CKA (lin)",
        "CKA (rbf)",
        "kNN",
        "RSA",
        "CCA",
        "SVCCA",
        "PWCCA",
        "RV",
        "Procrustes",
    ]
    order = {name: idx for idx, name in enumerate(preferred)}
    return sorted(metrics, key=lambda m: (order.get(m, len(order)), m))


def _collect_metric_variants(
    vals: Dict[tuple[str, str], np.ndarray],
) -> Dict[str, set[str]]:
    variants: Dict[str, set[str]] = {}
    for metric, variant in vals.keys():
        variants.setdefault(metric, set()).add(variant)
    return variants


def _quantile_keys(variants: Sequence[str]) -> list[str]:
    q_vals = []
    for v in variants:
        if v.startswith("q") and v[1:].isdigit():
            q_vals.append((int(v[1:]), v))
    return [v for _, v in sorted(q_vals)]


def _default_quantile_key(variants: Sequence[str]) -> str | None:
    q_keys = _quantile_keys(variants)
    return q_keys[-1] if q_keys else None


def _variant_title(variant: str, q_keys: Sequence[str]) -> str:
    if variant == "raw":
        return "raw"
    if variant == "null_centered":
        return "null-centered"
    if variant == "z":
        return "z-score"
    if variant == "ari":
        return "ARI-adjusted"
    if variant.startswith("q") and variant[1:].isdigit():
        if len(q_keys) == 1 and variant == "q95":
            return "gated"
        return f"gated {variant}"
    return variant


def _save_fig(path: Path) -> None:
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _render_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    fig: plt.Figure | None = None,
    cmap: str = SEQUENTIAL,
    vmin: float | None = None,
    vmax: float | None = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    xticks: Sequence | None = None,
    xticklabels: Sequence | None = None,
    yticks: Sequence | None = None,
    yticklabels: Sequence | None = None,
    xrotation: int = 0,
    colorbar: bool = True,
    colorbar_label: str = "",
    interpolation: str = "nearest",
    extent: Sequence[float] | None = None,
) -> plt.cm.ScalarMappable:
    """Render heatmap with consistent PRH styling.

    Args:
        ax: Matplotlib axes to render on
        data: 2D array of values
        fig: Figure for colorbar (required if colorbar=True)
        cmap: Colormap name
        vmin, vmax: Color scale limits
        xlabel, ylabel, title: Axis labels and title
        xticks, xticklabels: X-axis tick positions and labels
        yticks, yticklabels: Y-axis tick positions and labels
        xrotation: Rotation angle for x-tick labels
        colorbar: Whether to add colorbar
        colorbar_label: Label for colorbar
        interpolation: Image interpolation method
        extent: [xmin, xmax, ymin, ymax] for axis limits

    Returns:
        The AxesImage for further customization
    """
    style_heatmap_axes(ax)
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        extent=extent,
    )

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=xrotation)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if colorbar and fig is not None:
        prh_colorbar(fig, ax, im, label=colorbar_label)

    return im


def _plot_line_with_error(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    std: np.ndarray | None = None,
    num_trials: int = 50,
    label: str = "",
    color: str | None = None,
    marker: str = "o",
    linestyle: str = "-",
    markersize: int = 4,
    linewidth: float = 1.5,
    alpha_fill: float = 0.25,
    clip_lower: float | None = 0.0,
) -> None:
    """Plot line with standard error band.

    Args:
        ax: Matplotlib axes to plot on
        x: X values
        y: Y values (means)
        std: Standard deviation (converted to SE using num_trials)
        num_trials: Number of trials for SE calculation
        label: Line label for legend
        color: Line color (auto if None)
        marker: Marker style
        linestyle: Line style
        markersize: Marker size
        linewidth: Line width
        alpha_fill: Alpha for error band fill
        clip_lower: Lower bound for error band (None to disable)
    """
    plot_kwargs = {
        "label": label,
        "marker": marker,
        "linestyle": linestyle,
        "markersize": markersize,
        "linewidth": linewidth,
    }
    if color is not None:
        plot_kwargs["color"] = color

    line = ax.plot(x, y, **plot_kwargs)

    if std is not None and np.any(std > 0):
        se = std / np.sqrt(num_trials)
        lower = y - se
        upper = y + se
        if clip_lower is not None:
            lower = np.maximum(lower, clip_lower)
        line_color = line[0].get_color()
        ax.fill_between(x, lower, upper, color=line_color, alpha=alpha_fill)


def _shorten_prh_label(name: str) -> str:
    label = name.split("/")[-1]
    label = label.replace("open_llama_", "openllama-")
    label = label.replace("bloomz-", "bloom-")
    label = label.replace("llama-", "llama-")
    label = label.replace("vit_", "ViT-")
    label = label.replace(".augreg_in21k", "")
    label = label.replace(".lvd142m", "")
    label = label.replace(".laion2b_ft_in12k", "-clip-ft12k")
    label = label.replace(".laion2b", "-clip")
    label = label.replace(".mae", "-mae")
    label = label.replace(".dinov2", "-d2")
    return label


def _get_prh_model_labels(
    shape: tuple[int, int],
    *,
    shorten_lang: bool = True,
    shorten_vis: bool = True,
) -> tuple[list[str], list[str]]:
    n_lang, n_vis = shape
    llm_models = [f"L{i+1}" for i in range(n_lang)]
    lvm_models = [f"V{i+1}" for i in range(n_vis)]
    try:
        from aristotelian.prh.prh_models import get_models

        llm_models, lvm_models = get_models("val", modality="all")
    except Exception:
        pass
    if len(llm_models) != n_lang:
        llm_models = [f"L{i+1}" for i in range(n_lang)]
    if len(lvm_models) != n_vis:
        lvm_models = [f"V{i+1}" for i in range(n_vis)]
    if shorten_lang:
        llm_labels = [_shorten_prh_label(m) for m in llm_models]
    else:
        llm_labels = list(llm_models)
    if shorten_vis:
        lvm_labels = [_shorten_prh_label(m) for m in lvm_models]
    else:
        lvm_labels = list(lvm_models)
    return (llm_labels, lvm_labels)


def _parse_llm_size_b(name: str) -> float | None:
    lower = name.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*b", lower)
    if match:
        return float(match.group(1)) * float(match.group(2))
    match = re.search(r"(\d+)\s*b\s*(\d+)", lower)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", lower)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*m", lower)
    if match:
        return float(match.group(1)) / 1000.0
    return None


def _infer_llm_family(name: str) -> str:
    lower = name.lower()
    if "bloom" in lower:
        return "bloomz"
    if "open_llama" in lower or "open-llama" in lower or "openllama" in lower:
        return "open-llama"
    if "llama" in lower:
        return "llama"
    if "olmo" in lower:
        return "olmo"
    if "gemma" in lower:
        return "gemma"
    if "mistral" in lower or "mixtral" in lower:
        return "mistral"
    return "other"


def _group_llm_indices_by_family(
    llm_names: Sequence[str],
) -> list[tuple[str, list[int]]]:
    family_order = [
        "bloomz",
        "open-llama",
        "llama",
        "olmo",
        "gemma",
        "mistral",
        "other",
    ]
    family_map: dict[str, list[int]] = {}
    for idx, name in enumerate(llm_names):
        family_map.setdefault(_infer_llm_family(name), []).append(idx)

    groups: list[tuple[str, list[int]]] = []
    for family in family_order:
        indices = family_map.pop(family, [])
        if indices:
            indices = sorted(
                indices,
                key=lambda i: (
                    _parse_llm_size_b(llm_names[i]) is None,
                    _parse_llm_size_b(llm_names[i]) or 0.0,
                    i,
                ),
            )
            groups.append((family, indices))

    for family in sorted(family_map):
        indices = family_map[family]
        indices = sorted(
            indices,
            key=lambda i: (
                _parse_llm_size_b(llm_names[i]) is None,
                _parse_llm_size_b(llm_names[i]) or 0.0,
                i,
            ),
        )
        groups.append((family, indices))
    return groups


def _hide_shared_ylabel(axes: Sequence[plt.Axes]) -> None:
    """Hide shared y-labels and y-ticks on all but the first axis.

    Use for multi-column plots where columns may share y-axis properties.
    - If y-labels match: hide y-label on all but first axis
    - If y-tick labels also match: hide y-tick labels on all but first axis
    """
    axes_list = list(axes)
    if len(axes_list) < 2:
        return

    # Check if y-labels match
    labels = [ax.get_ylabel() for ax in axes_list]
    first_label = labels[0]
    labels_match = first_label and all(label == first_label for label in labels)

    # Check if y-tick labels match (compare the actual tick label strings)
    # Only hide ticks if they are explicitly set (not auto-generated empty strings)
    def get_yticklabels(ax):
        return [t.get_text() for t in ax.get_yticklabels()]

    first_ticks = get_yticklabels(axes_list[0])
    # Only consider ticks as matching if they're actually set (not all empty)
    has_real_ticks = first_ticks and any(t.strip() for t in first_ticks)
    ticks_match = has_real_ticks and all(
        get_yticklabels(ax) == first_ticks for ax in axes_list[1:]
    )

    for ax in axes_list[1:]:
        if labels_match:
            ax.set_ylabel("")
        if ticks_match:
            ax.set_yticklabels([])


def _require_asset(path: Path) -> bool:
    if not path.exists():
        print(f"skip missing {path}")
        return False
    return True


def _extract_metric_and_k(filename: str) -> tuple[str, int | None, float | None]:
    """Extract metric name, k value, and sigma from PRH alignment filename.

    Examples:
        prh_alignment.npy -> ("mutual_knn", 10, None)  # default
        prh_alignment_mutual_knn_k20.npy -> ("mutual_knn", 20, None)
        prh_alignment_cka_lin.npy -> ("cka_lin", None, None)
        prh_alignment_cycle_knn_k50.npy -> ("cycle_knn", 50, None)
        prh_alignment_cka_rbf.npy -> ("cka_rbf", None, 1.0)  # default sigma
        prh_alignment_cka_rbf_sigma0.5.npy -> ("cka_rbf", None, 0.5)
    """
    stem = filename.replace(".npy", "").replace("prh_alignment", "")
    if not stem or stem == "":
        return "mutual_knn", 10, None  # default file

    stem = stem.lstrip("_")

    # Check for sigma value pattern (for RBF kernel)
    sigma_match = re.search(r"_sigma([\d.]+)$", stem)
    sigma_val = None
    if sigma_match:
        sigma_val = float(sigma_match.group(1))
        stem = stem[: sigma_match.start()]

    # Check for k value pattern at the end
    k_match = re.search(r"_k(\d+)$", stem)
    if k_match:
        k_val = int(k_match.group(1))
        metric = stem[: k_match.start()]
        return metric, k_val, sigma_val

    # No k value in filename - check if it's a kNN-based metric
    knn_metrics = ("mutual_knn", "cycle_knn", "cknna", "knn")
    if stem in knn_metrics:
        return stem, 10, sigma_val  # default k for kNN metrics without explicit k

    # Check if it's an RBF metric without explicit sigma (default to 1.0)
    if stem == "cka_rbf" and sigma_val is None:
        sigma_val = 1.0

    return stem, None, sigma_val  # non-kNN metric


def _iter_prh_alignment_payloads(
    assets_dir: Path,
) -> list[tuple[Path, dict, str, int | None, float | None]]:
    """Iterate over PRH alignment payloads.

    Returns list of (path, payload, metric, k, sigma) tuples.
    k is None for non-kNN metrics.
    sigma is None for non-RBF metrics, otherwise the RBF kernel bandwidth.
    """
    candidates = sorted(assets_dir.glob("prh_alignment*.npy"))
    if not candidates:
        print(f"skip missing {assets_dir / 'prh_alignment.npy'}")
        return []
    payloads = []
    for path in candidates:
        if not path.is_file():
            continue
        payload = np.load(path, allow_pickle=True).item()
        metric_from_payload = payload.get("metric")
        metric_from_file, k_val, sigma_val = _extract_metric_and_k(path.name)

        # Prefer payload metric if available, but use file-derived k and sigma
        metric = metric_from_payload if metric_from_payload else metric_from_file
        payloads.append((path, payload, metric, k_val, sigma_val))
    return payloads


def _render_null_drift_aggregate(
    metrics_list: list,
    output_path: Path,
    vals_dict: dict,
    *,
    variants_by_metric: dict[str, set[str]],
    n_list: list,
    d_list: list,
    label_overrides: dict[str, str] | None = None,
) -> None:
    """Render aggregate raw vs calibrated heatmaps across metrics.

    This is a shared helper for both gaussian and heavy-tailed null drift plots.

    Args:
        label_overrides: Optional dict mapping metric names to custom display labels.
            If provided, these take precedence over _prettify_metric().
    """
    if not metrics_list:
        return
    fig_width = max(6.75, 1.8 * len(metrics_list))
    fig, axes = plt.subplots(2, len(metrics_list), figsize=(fig_width, 3.6))
    if len(metrics_list) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    cmap = plt.colormaps.get_cmap(SEQUENTIAL).copy()
    cmap.set_bad("#DDDDDD")
    for col_idx, metric in enumerate(metrics_list):
        variants = variants_by_metric.get(metric, set())
        q_key = _default_quantile_key(variants)
        raw_local = vals_dict[(metric, "raw")]
        if q_key is None:
            q_local = np.full_like(raw_local, np.nan)
            calibrated_title = "calibrated (n/a)"
        else:
            q_local = vals_dict[(metric, q_key)]
            calibrated_title = "calibrated"
        vmin_r = float(np.nanmin([raw_local, q_local]))
        vmax_r = float(np.nanmax([raw_local, q_local]))
        # Use label override if provided, otherwise use _prettify_metric
        if label_overrides and metric in label_overrides:
            pretty_metric = label_overrides[metric]
        else:
            pretty_metric = _prettify_metric(metric)
        for row_idx, (data, title) in enumerate(
            [(raw_local, "uncalibrated"), (q_local, calibrated_title)]
        ):
            ax = axes[row_idx, col_idx]
            style_heatmap_axes(ax)
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmin_r,
                vmax=vmax_r,
            )
            # Title only on top row with just metric name
            if row_idx == 0:
                ax.set_title(pretty_metric)
            # X-axis label and ticks only on bottom row
            if row_idx == 1:
                ax.set_xlabel("$d$")
                ax.set_xticks(range(len(d_list)))
                ax.set_xticklabels(d_list, rotation=45, fontsize=10)
            else:
                ax.set_xlabel("")
                ax.set_xticks(range(len(d_list)))
                ax.set_xticklabels([])
            # Y-axis label and ticks only on first column
            if col_idx == 0:
                # {title}
                ax.set_ylabel("$n$")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            ax.set_yticks(range(len(n_list)))
            if col_idx == 0:
                ax.set_yticklabels(n_list, fontsize=10)
            prh_colorbar(fig, ax, im)
    _save_fig(output_path)


def _plot_null_drift(assets_dir: Path, *, force: bool, heavy: bool = False) -> None:
    """Plot null drift calibration heatmaps.

    Unified implementation for both gaussian and heavy-tailed noise experiments.

    Args:
        assets_dir: Directory containing experiment data and for plot outputs
        force: If True, regenerate plots even if they exist
        heavy: If True, use heavy-tailed noise data (default: gaussian)
    """
    suffix = "heavy" if heavy else "gaussian"
    title_suffix = " (heavy)" if heavy else ""
    src = assets_dir / f"null_drift_{suffix}.npy"
    if not _require_asset(src):
        return

    arr = np.load(src, allow_pickle=True)
    n_list, d_list, vals = arr.tolist()
    variants_by_metric = _collect_metric_variants(vals)
    metrics_to_compare = _order_metrics(variants_by_metric.keys())

    if "CKA (lin)" in variants_by_metric:
        metric = "CKA (lin)"
        default_q = _default_quantile_key(variants_by_metric[metric])
        if default_q is not None:
            output = assets_dir / f"null_drift_{suffix}_cka_lin.png"
            if not _should_skip([output], force):
                raw = vals[(metric, "raw")]
                q_vals = vals[(metric, default_q)]
                vmin = min(np.min(raw), np.min(q_vals))
                vmax = max(np.max(raw), np.max(q_vals))
                fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))

                for ax in axes:
                    style_heatmap_axes(ax)

                im0 = axes[0].imshow(
                    raw,
                    origin="lower",
                    aspect="auto",
                    cmap=SEQUENTIAL,
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[0].set_title(f"{metric}{title_suffix}")
                axes[0].set_xlabel("$d$")
                axes[0].set_ylabel("$n$ (uncalibrated)")
                axes[0].set_xticks(range(len(d_list)))
                axes[0].set_xticklabels(d_list, rotation=45)
                axes[0].set_yticks(range(len(n_list)))
                axes[0].set_yticklabels(n_list)
                prh_colorbar(fig, axes[0], im0)

                im1 = axes[1].imshow(
                    q_vals,
                    origin="lower",
                    aspect="auto",
                    cmap=SEQUENTIAL,
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[1].set_title("")
                axes[1].set_xlabel("$d$")
                axes[1].set_ylabel("(calibrated)")
                axes[1].set_xticks(range(len(d_list)))
                axes[1].set_xticklabels(d_list, rotation=45)
                axes[1].set_yticks(range(len(n_list)))
                axes[1].set_yticklabels([])
                prh_colorbar(fig, axes[1], im1)
                _save_fig(output)

    for metric in metrics_to_compare:
        variants = variants_by_metric.get(metric, set())
        q_keys = _quantile_keys(variants)
        variant_keys = (
            ["raw"]
            + q_keys
            + [key for key in ("null_centered", "z", "ari") if key in variants]
        )
        data_map = {key: vals[(metric, key)] for key in variant_keys}
        raw_like_keys = ["raw"] + q_keys
        raw_like = [data_map[key] for key in raw_like_keys if key in data_map]
        vmin_raw = float(np.min(raw_like)) if raw_like else 0.0
        vmax_raw = float(np.max(raw_like)) if raw_like else 1.0
        diverging_keys = [
            key for key in ("null_centered", "z", "ari") if key in data_map
        ]
        diverging = [data_map[key] for key in diverging_keys]
        vmax_div = float(np.max(np.abs(diverging))) if diverging else 1.0

        fig, axes = plt.subplots(
            1, len(variant_keys), figsize=(1.8 * len(variant_keys), 2.0)
        )
        if len(variant_keys) == 1:
            axes = [axes]
        for col_idx, (ax, key) in enumerate(zip(axes, variant_keys)):
            style_heatmap_axes(ax)
            data = data_map[key]
            if key in raw_like_keys:
                im = ax.imshow(
                    data,
                    origin="lower",
                    aspect="auto",
                    cmap=SEQUENTIAL,
                    vmin=vmin_raw,
                    vmax=vmax_raw,
                )
            else:
                im = ax.imshow(
                    data,
                    origin="lower",
                    aspect="auto",
                    cmap=DIVERGING,
                    vmin=-vmax_div,
                    vmax=vmax_div,
                )
            # Title only on first panel with metric name
            if col_idx == 0:
                ax.set_title(f"{_prettify_metric(metric)}{title_suffix}")
            else:
                ax.set_title(_variant_title(key, q_keys))
            ax.set_xlabel("$d$")
            ax.set_xticks(range(len(d_list)))
            ax.set_xticklabels(d_list, rotation=45)
            ax.set_yticks(range(len(n_list)))
            if col_idx == 0:
                ax.set_ylabel("$n$")
                ax.set_yticklabels(n_list)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            prh_colorbar(fig, ax, im)
        out_path = assets_dir / f"null_drift_{suffix}_{_slug(metric)}.png"
        _save_fig(out_path)

        # Extra: raw vs gated (two-panel) - styled like aggregate plots
        default_q = _default_quantile_key(variants)
        if default_q is not None:
            raw_local = data_map["raw"]
            q_local = data_map[default_q]
            vmin_r = float(np.min([raw_local, q_local]))
            vmax_r = float(np.max([raw_local, q_local]))
            pretty_metric = _prettify_metric(metric)
            fig, axes = plt.subplots(1, 2, figsize=(3.25, 2.0))
            for col_idx, (ax, data, title) in enumerate(
                zip(axes, [raw_local, q_local], ["uncalibrated", "calibrated"])
            ):
                style_heatmap_axes(ax)
                im = ax.imshow(
                    data,
                    origin="lower",
                    aspect="auto",
                    cmap=SEQUENTIAL,
                    vmin=vmin_r,
                    vmax=vmax_r,
                )
                ax.set_title(pretty_metric if col_idx == 0 else "")
                ax.set_xlabel("$d$")
                ax.set_xticks(range(len(d_list)))
                ax.set_xticklabels(d_list, rotation=45)
                ax.set_yticks(range(len(n_list)))
                if col_idx == 0:
                    ax.set_ylabel(f"$n$ ({title})")
                    ax.set_yticklabels(n_list)
                else:
                    ax.set_ylabel(f"({title})")
                    ax.set_yticklabels([])
                prh_colorbar(fig, ax, im)
            out_path = assets_dir / f"null_drift_{suffix}_reduction_{_slug(metric)}.png"
            _save_fig(out_path)

            # Figure 1 version: borderless two-panel heatmap
            fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.8))
            for col_idx, (ax, data, title) in enumerate(
                zip(axes, [raw_local, q_local], ["uncalibrated", "calibrated"])
            ):
                im = ax.imshow(
                    data,
                    origin="lower",
                    aspect="auto",
                    cmap=SEQUENTIAL,
                    vmin=vmin_r,
                    vmax=vmax_r,
                )
                # Remove all spines (borderless)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                # Remove all ticks and labels
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                # Subtle title only
                ax.set_title(title, fontsize=9)
            plt.tight_layout(pad=0.5)
            out_path = (
                assets_dir / f"null_drift_{suffix}_reduction_{_slug(metric)}_fig1.png"
            )
            _save_fig(out_path)

    # Aggregate: raw vs gated across metrics
    # Generate both main (filtered) and appendix (full) versions
    # Full version (appendix)
    agg_out = assets_dir / f"null_drift_{suffix}_reduction_aggregate.png"
    if metrics_to_compare and not _should_skip([agg_out], force):
        _render_null_drift_aggregate(
            metrics_to_compare,
            agg_out,
            vals,
            variants_by_metric=variants_by_metric,
            n_list=n_list,
            d_list=d_list,
        )

    # Main paper version (only CKA, mknn, CCA with custom labels)
    main_metrics = [m for m in MAIN_PAPER_NULL_DRIFT_METRICS if m in metrics_to_compare]
    agg_out_main = assets_dir / f"null_drift_{suffix}_reduction_aggregate_main.png"
    if main_metrics and not _should_skip([agg_out_main], force):
        _render_null_drift_aggregate(
            main_metrics,
            agg_out_main,
            vals,
            variants_by_metric=variants_by_metric,
            n_list=n_list,
            d_list=d_list,
            label_overrides=MAIN_PAPER_NULL_DRIFT_LABELS,
        )


def _plot_null_drift_gaussian(assets_dir: Path, *, force: bool) -> None:
    """Plot null drift calibration heatmaps for gaussian noise."""
    return _plot_null_drift(assets_dir, force=force, heavy=False)


def _plot_null_drift_heavy(assets_dir: Path, *, force: bool) -> None:
    """Plot null drift calibration heatmaps for heavy-tailed noise."""
    return _plot_null_drift(assets_dir, force=force, heavy=True)


def _plot_perm_budget(assets_dir: Path, *, force: bool) -> None:
    src = assets_dir / "perm_budget.npy"
    if not _require_asset(src):
        return
    output = assets_dir / "perm_budget_tau.pdf"
    if _should_skip([output], force):
        print(f"skip {output}")
        return

    loaded = np.load(src, allow_pickle=True).item()
    # Support both old format (dict of metrics) and new format (dict with 'results' and 'num_trials')
    if "results" in loaded and "num_trials" in loaded:
        arr = loaded["results"]
        num_trials = loaded["num_trials"]
    else:
        arr = loaded
        num_trials = 50  # Fallback for legacy data without num_trials
    budgets = sorted(next(iter(arr.values())).keys())
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))

    for ax in axes:
        style_line_axes(ax)

    for metric, out in arr.items():
        tau = np.array([out[b].tau for b in budgets])
        mean_g = np.array([out[b].mean for b in budgets])
        std_g = np.array([getattr(out[b], "std", 0.0) for b in budgets])
        budgets_arr = np.array(budgets)

        axes[0].plot(
            budgets_arr,
            tau,
            "o-",
            label=_prettify_metric(metric),
            markersize=4,
            linewidth=1.5,
        )
        _plot_line_with_error(
            axes[1],
            budgets_arr,
            mean_g,
            std=std_g,
            num_trials=num_trials,
            label=_prettify_metric(metric),
            alpha_fill=0.2,
            clip_lower=None,
        )

    axes[0].set_xlabel(r"Number of Permutations $K$")
    axes[0].set_ylabel(r"Threshold $\tau$")
    axes[0].set_title("Permutation budget")
    axes[1].set_xlabel(r"Number of Permutations $K$")
    axes[1].set_ylabel("Calibrated score")
    axes[1].set_title("Permutation budget")
    prh_legend(axes[0], style="square")
    prh_legend(axes[1], style="square")
    _save_fig(output)


def _plot_type1_calibration(assets_dir: Path, *, force: bool) -> None:
    src = assets_dir / "type1_calibration.npy"
    if not _require_asset(src):
        return
    output = assets_dir / "type1_calibration.png"
    if _should_skip([output], force):
        print(f"skip {output}")
        return

    arr = np.load(src, allow_pickle=True)
    payload = arr.item() if arr.shape == () else arr
    if isinstance(payload, dict):
        ds = payload["ds"]
        rates = payload["rates"]
        positives = payload["positives"]
        num_trials = payload.get("num_trials", 50)
        alpha = payload.get("alpha", 0.05)
        default_n = payload.get("default_n")
        if "rates_by_null" in payload and default_n is not None:
            n_idx = payload["ns"].index(default_n)
            rates = {
                metric: payload["rates_by_null"]["gaussian"][metric][n_idx]
                for metric in rates.keys()
            }
            positives = {
                metric: payload["positives_by_null"]["gaussian"][metric][n_idx]
                for metric in positives.keys()
            }
    else:
        ds, rates = payload.tolist()
        positives = None
        num_trials = 50
        alpha = 0.05
        default_n = 256

    fig, ax = plt.subplots(figsize=(3.375, 2.5))
    style_line_axes(ax)

    n_denom = default_n if default_n is not None else 256
    for metric, vals in rates.items():
        xs = [d / n_denom for d in ds]
        ax.plot(
            xs, vals, "o-", label=_prettify_metric(metric), markersize=4, linewidth=1.5
        )
        if positives is not None:
            counts = np.array(positives[metric], dtype=float)
            p = counts / float(num_trials)
            z = 1.96
            wilson_denom = 1.0 + z**2 / num_trials
            center = (p + z**2 / (2 * num_trials)) / wilson_denom
            half = (
                z
                * np.sqrt((p * (1 - p) / num_trials) + (z**2 / (4 * num_trials**2)))
                / wilson_denom
            )
            lower = np.clip(center - half, 0.0, 1.0)
            upper = np.clip(center + half, 0.0, 1.0)
            ax.fill_between(xs, lower, upper, alpha=0.25)
    ax.axhline(
        alpha, color="black", linewidth=1.5, linestyle="--", label=r"$\alpha$ = 0.05"
    )
    ax.set_xlabel(r"$d/n$")
    ax.set_ylabel("Type I error rate")
    prh_legend(ax, style="square")
    _save_fig(output)

    if isinstance(payload, dict) and "rates_by_null" in payload:
        null_types = payload["null_types"]
        ns = payload["ns"]
        default_n = payload.get("default_n", ns[len(ns) // 2])
        n_idx = ns.index(default_n)
        n_panels = len(null_types)
        fig, axes = plt.subplots(1, n_panels, figsize=(min(6.75, 3.5 * n_panels), 2.5))
        if len(null_types) == 1:
            axes = [axes]
        for ax, null_type in zip(axes, null_types):
            style_line_axes(ax)
            rates_null = payload["rates_by_null"][null_type]
            for metric, grid in rates_null.items():
                vals = grid[n_idx]
                ax.plot(
                    [d / default_n for d in ds], vals, label=_prettify_metric(metric)
                )
            ax.axhline(alpha, color="black", linestyle="--", label=r"$\alpha$ = 0.05")
            # Prettify null type names
            null_title = null_type.replace("_", " ").title()
            ax.set_title(f"{null_title} ($n$={default_n})")
            ax.set_xlabel(r"$d/n$")
            ax.set_ylabel("Type I error rate")
            prh_legend(ax, style="square", fontsize=7)
        _save_fig(assets_dir / "type1_calibration_nulls.png")


def _plot_snr_sweep(assets_dir: Path, *, force: bool) -> None:
    variants = _noise_variants(assets_dir, "snr_sweep.npy")
    if not variants:
        src = assets_dir / "snr_sweep.npy"
        if not _require_asset(src):
            return
        variants = [(DEFAULT_NOISE_TYPE, src)]

    for noise_type, src in variants:
        if not _require_asset(src):
            continue
        suffix = _noise_suffix(noise_type)
        output = assets_dir / f"snr_sweep{suffix}.png"
        if _should_skip([output], force):
            print(f"skip {output}")
            continue

        arr = np.load(src, allow_pickle=True)
        payload = arr.item() if arr.shape == () else arr
        if isinstance(payload, dict):
            noise_levels = np.array(payload["noise_levels"], dtype=float)
            strengths = list(payload["strengths"])
            ranks = list(payload["ranks"])
            metrics = payload.get("metrics")
            if metrics:
                for metric in metrics:
                    means = np.asarray(payload["mean_by_metric"][metric], dtype=float)
                    raw_means = np.asarray(
                        payload["raw_mean_by_metric"][metric], dtype=float
                    )
                    tau_means = np.asarray(
                        payload["tau_mean_by_metric"][metric], dtype=float
                    )
                    stds = np.asarray(payload["std_by_metric"][metric], dtype=float)
                    metric_out = assets_dir / f"snr_sweep_{metric}{suffix}.png"
                    if _should_skip([metric_out], force):
                        continue
                    strength_idx = strengths.index(max(strengths)) if strengths else 0
                    fig, ax = plt.subplots(figsize=(3.25, 2.5))
                    style_line_axes(ax)

                    num_trials = payload.get("num_trials", 50)
                    for r_idx, r in enumerate(ranks):
                        vals = means[strength_idx, r_idx, :]
                        std_vals = stds[strength_idx, r_idx, :]
                        _plot_line_with_error(
                            ax,
                            noise_levels,
                            vals,
                            std=std_vals,
                            num_trials=num_trials,
                            label=f"$r={r}$",
                            marker="",
                        )

                    ax.set_xlabel(r"Noise level $\sigma$")
                    ax.set_ylabel(_prettify_metric(metric))
                    ax.set_title(f"$s={strengths[strength_idx]:g}$")
                    prh_legend(ax, style="square")
                    _save_fig(metric_out)
            means = np.asarray(payload["mean"], dtype=float)
            raw_means = np.asarray(payload.get("raw_mean", means * np.nan), dtype=float)
            tau_means = np.asarray(payload.get("tau_mean", means * np.nan), dtype=float)

            strength_idx = strengths.index(max(strengths)) if strengths else 0
            num_trials = payload.get("num_trials", 50)
            fig, ax = plt.subplots(figsize=(3.25, 2.5))
            style_line_axes(ax)

            for r_idx, r in enumerate(ranks):
                vals = means[strength_idx, r_idx, :]
                stds = np.asarray(payload["std"], dtype=float)[strength_idx, r_idx, :]
                _plot_line_with_error(
                    ax,
                    noise_levels,
                    vals,
                    std=stds,
                    num_trials=num_trials,
                    label=f"$r={r}$",
                    marker="",
                )

            ax.set_xlabel(r"Noise level $\sigma$")
            ax.set_ylabel("Calibrated score")
            ax.set_title(f"$s={strengths[strength_idx]:g}$")
            prh_legend(ax, style="square")
            _save_fig(output)

            multi_output = assets_dir / f"snr_sweep_strength{suffix}.png"
            if not _should_skip([multi_output], force):
                stds_arr = np.asarray(payload["std"], dtype=float)
                num_trials = payload.get("num_trials", 50)
                fig, axes = plt.subplots(
                    1, len(ranks), figsize=(min(6.75, 3.25 * len(ranks)), 2.5)
                )
                if len(ranks) == 1:
                    axes = [axes]
                colors = plt.cm.viridis(np.linspace(0, 1, len(strengths)))
                for r_idx, r in enumerate(ranks):
                    ax = axes[r_idx]
                    style_line_axes(ax)
                    for s_idx, strength in enumerate(strengths):
                        vals = means[s_idx, r_idx, :]
                        std_vals = stds_arr[s_idx, r_idx, :]
                        _plot_line_with_error(
                            ax,
                            noise_levels,
                            vals,
                            std=std_vals,
                            num_trials=num_trials,
                            label=f"$s={strength:g}$",
                            color=colors[s_idx],
                            alpha_fill=0.2,
                        )
                    ax.set_title(f"$r={r}$")
                    ax.set_xlabel(r"Noise level $\sigma$")
                    ax.set_ylabel("Calibrated score")
                    prh_legend(ax, style="square", fontsize=8)
                _hide_shared_ylabel(axes)
                _save_fig(multi_output)

            heat_extent = [
                float(noise_levels[0]),
                float(noise_levels[-1]),
                0,
                len(strengths) - 1,
            ]
            s_labels = [f"{s:g}" for s in strengths]
            for r_idx, r in enumerate(ranks):
                heat_out = assets_dir / f"snr_sweep_heatmap_r{r}{suffix}.png"
                if not _should_skip([heat_out], force):
                    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5))
                    _render_heatmap(
                        ax,
                        means[:, r_idx, :],
                        fig=fig,
                        vmin=0.0,
                        vmax=1.0,
                        extent=heat_extent,
                        xlabel=r"Noise level $\sigma$",
                        ylabel="Signal strength",
                        title="Calibrated score",
                        yticks=list(range(len(strengths))),
                        yticklabels=s_labels,
                        # colorbar_label="Calibrated score",
                        interpolation="bilinear",
                    )
                    _save_fig(heat_out)

                if np.isfinite(raw_means).any():
                    raw_out = assets_dir / f"snr_sweep_raw_heatmap_r{r}{suffix}.png"
                    if not _should_skip([raw_out], force):
                        fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5))
                        _render_heatmap(
                            ax,
                            raw_means[:, r_idx, :],
                            fig=fig,
                            vmin=0.0,
                            vmax=1.0,
                            extent=heat_extent,
                            xlabel=r"Noise level $\sigma$",
                            ylabel="Signal strength",
                            title="Raw score",
                            yticks=list(range(len(strengths))),
                            yticklabels=s_labels,
                            # colorbar_label="Raw score",
                            interpolation="bilinear",
                        )
                        _save_fig(raw_out)

                if np.isfinite(tau_means).any():
                    tau_out = assets_dir / f"snr_sweep_tau_heatmap_r{r}{suffix}.png"
                    if not _should_skip([tau_out], force):
                        fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5))
                        _render_heatmap(
                            ax,
                            tau_means[:, r_idx, :],
                            fig=fig,
                            vmin=0.0,
                            vmax=1.0,
                            extent=heat_extent,
                            xlabel=r"Noise level $\sigma$",
                            ylabel="Signal strength",
                            title=f"Threshold ($r={r}$)",
                            yticks=list(range(len(strengths))),
                            yticklabels=s_labels,
                            colorbar_label=r"$\tau_\alpha$",
                            interpolation="bilinear",
                        )
                        _save_fig(tau_out)

                if np.isfinite(raw_means).any() and np.isfinite(tau_means).any():
                    delta_out = (
                        assets_dir / f"snr_sweep_raw_minus_tau_heatmap_r{r}{suffix}.png"
                    )
                    if not _should_skip([delta_out], force):
                        fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5))
                        delta = raw_means[:, r_idx, :] - tau_means[:, r_idx, :]
                        vmax = float(np.max(np.abs(delta))) if delta.size else 1.0
                        _render_heatmap(
                            ax,
                            delta,
                            fig=fig,
                            cmap=DIVERGING,
                            vmin=-vmax,
                            vmax=vmax,
                            extent=heat_extent,
                            xlabel=r"Noise level $\sigma$",
                            ylabel="Signal strength",
                            title=f"$r={r}$",
                            yticks=list(range(len(strengths))),
                            yticklabels=s_labels,
                            colorbar_label=r"Raw $-$ $\tau_\alpha$",
                        )
                        _save_fig(delta_out)
        else:
            noise_levels, snr_out = payload
            fig, ax = plt.subplots(figsize=(3.25, 2.5))
            style_line_axes(ax)
            for r, vals in snr_out.items():
                ax.plot(noise_levels, vals, label=f"$r={r}$")
            ax.set_xlabel(r"Noise level $\sigma$")
            ax.set_ylabel("Calibrated score")
            prh_legend(ax, style="square")
            _save_fig(output)


def _plot_type1_and_power_combined(assets_dir: Path, *, force: bool) -> None:
    """Combined 2-panel plot: Type I error (left) and Power (right) with shared legend below."""
    src_type1 = assets_dir / "type1_calibration.npy"
    src_signal = assets_dir / "signal_fn_rate.npy"

    if not _require_asset(src_type1) or not _require_asset(src_signal):
        return

    output = assets_dir / "type1_and_power_combined.pdf"
    if _should_skip([output], force):
        print(f"skip {output}")
        return

    # Load type1 calibration data
    arr_type1 = np.load(src_type1, allow_pickle=True)
    payload_type1 = arr_type1.item() if arr_type1.shape == () else arr_type1
    if isinstance(payload_type1, dict):
        ds = payload_type1["ds"]
        rates = payload_type1["rates"]
        positives = payload_type1["positives"]
        num_trials_type1 = payload_type1.get("num_trials", 50)
        alpha = payload_type1.get("alpha", 0.05)
        default_n = payload_type1.get("default_n")
        if "rates_by_null" in payload_type1 and default_n is not None:
            n_idx = payload_type1["ns"].index(default_n)
            rates = {
                metric: payload_type1["rates_by_null"]["gaussian"][metric][n_idx]
                for metric in rates.keys()
            }
            positives = {
                metric: payload_type1["positives_by_null"]["gaussian"][metric][n_idx]
                for metric in positives.keys()
            }
    else:
        ds, rates = payload_type1.tolist()
        positives = None
        num_trials_type1 = 50
        alpha = 0.05
        default_n = 256

    # Load signal FN rate data
    payload_signal = np.load(src_signal, allow_pickle=True).item()
    strengths = np.asarray(payload_signal["strengths"], dtype=float)
    metrics_signal = list(payload_signal["metrics"])
    fn_rate = payload_signal["fn_rate"]
    num_trials_signal = payload_signal.get("num_trials", 50)

    # Filter to signal >= 1.0 if older data has lower values
    mask = strengths >= 1.0
    if mask.sum() > 0:
        strengths = strengths[mask]

    # Create 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))
    for ax in axes:
        style_line_axes(ax)

    # Get consistent colors for metrics (use same order for both panels)
    all_metrics = list(rates.keys())
    metric_colors = {
        m: LINE_COLORS[i % len(LINE_COLORS)] for i, m in enumerate(all_metrics)
    }

    # LEFT PANEL: Type I error rate
    ax_left = axes[0]
    n_denom = default_n if default_n is not None else 256
    for metric, vals in rates.items():
        xs = [d / n_denom for d in ds]
        color = metric_colors.get(metric, None)
        ax_left.plot(
            xs,
            vals,
            "o-",
            label=_prettify_metric(metric),
            markersize=4,
            linewidth=1.5,
            color=color,
        )
        if positives is not None:
            counts = np.array(positives[metric], dtype=float)
            p = counts / float(num_trials_type1)
            z = 1.96
            wilson_denom = 1.0 + z**2 / num_trials_type1
            center = (p + z**2 / (2 * num_trials_type1)) / wilson_denom
            half = (
                z
                * np.sqrt(
                    (p * (1 - p) / num_trials_type1)
                    + (z**2 / (4 * num_trials_type1**2))
                )
                / wilson_denom
            )
            lower = np.clip(center - half, 0.0, 1.0)
            upper = np.clip(center + half, 0.0, 1.0)
            ax_left.fill_between(xs, lower, upper, alpha=0.25, color=color)

    # Add alpha line (this gets its own separate legend inside the left plot)
    alpha_line = ax_left.axhline(
        alpha, color="black", linewidth=1.5, linestyle="--", label=r"$\alpha$ = 0.05"
    )
    ax_left.set_xlabel(r"$d/n$")
    ax_left.set_ylabel("Type I error rate")

    # Create separate small legend for alpha line only (inside left plot)
    ax_left.legend(
        [alpha_line],
        [r"$\alpha$ = 0.05"],
        loc="upper right",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="#C8CDD4",
    )

    # RIGHT PANEL: Power (1 - FN rate)
    ax_right = axes[1]
    for metric in metrics_signal:
        fn_vals = np.asarray(fn_rate[metric], dtype=float)
        if mask.sum() > 0 and len(fn_vals) > mask.sum():
            fn_vals = fn_vals[mask]
        power_vals = 1.0 - fn_vals
        color = metric_colors.get(metric, None)
        ax_right.plot(
            strengths,
            power_vals,
            marker="o",
            linewidth=1.5,
            color=color,
            label=_prettify_metric(metric),
        )
        # Wilson confidence interval bands
        p = fn_vals
        z = 1.96
        wilson_denom = 1.0 + z**2 / num_trials_signal
        center = (p + z**2 / (2 * num_trials_signal)) / wilson_denom
        half = (
            z
            * np.sqrt(
                (p * (1 - p) / num_trials_signal) + (z**2 / (4 * num_trials_signal**2))
            )
            / wilson_denom
        )
        lower = np.clip(1.0 - (center + half), 0.0, 1.0)
        upper = np.clip(1.0 - (center - half), 0.0, 1.0)
        ax_right.fill_between(strengths, lower, upper, color=color, alpha=0.2)

    ax_right.set_xlabel("Signal strength")
    ax_right.set_ylabel("Power (detection rate)")

    # Adjust y-axis for power panel
    all_vals = [1.0 - np.asarray(fn_rate[m], dtype=float) for m in metrics_signal]
    if all_vals:
        vals_arr = np.concatenate(all_vals)
        vals_arr = vals_arr[np.isfinite(vals_arr)]
    else:
        vals_arr = np.array([])
    if vals_arr.size:
        std = float(vals_arr.std())
        y_min = max(0.0, float(vals_arr.min() - std))
        y_max = min(1.05, float(vals_arr.max() + std))
        if y_max - y_min < 1e-6:
            y_min = max(0.0, float(vals_arr.min() - 0.05))
            y_max = min(1.05, float(vals_arr.max() + 0.05))
        ax_right.set_ylim(y_min, y_max)
    else:
        ax_right.set_ylim(0.0, 1.05)

    # Collect handles for shared legend (metrics only, not alpha line)
    # Get handles from left panel but exclude the alpha line
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()

    # Filter out alpha line from shared legend
    shared_handles = []
    shared_labels = []
    for h, lbl in zip(handles_left, labels_left):
        if lbl != r"$\alpha$ = 0.05":
            shared_handles.append(h)
            shared_labels.append(lbl)
    # Add any from right panel that aren't duplicates
    for h, lbl in zip(handles_right, labels_right):
        if lbl not in shared_labels:
            shared_handles.append(h)
            shared_labels.append(lbl)

    # Create shared legend below both panels
    fig.legend(
        shared_handles,
        shared_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(shared_labels), 6),
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="#C8CDD4",
        shadow=True,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    _save_fig(output)


def _plot_phase_diagram(assets_dir: Path, *, force: bool) -> None:
    variants = _noise_variants(assets_dir, "phase_diagram.npy")
    if not variants:
        src = assets_dir / "phase_diagram.npy"
        if not _require_asset(src):
            return
        variants = [(DEFAULT_NOISE_TYPE, src)]

    for noise_type, src in variants:
        if not _require_asset(src):
            continue
        suffix = _noise_suffix(noise_type)
        output = assets_dir / f"phase_diagram{suffix}.png"
        if _should_skip([output], force):
            print(f"skip {output}")
            continue

        payload = np.load(src, allow_pickle=True).item()
        sigmas = np.asarray(payload["sigmas"], dtype=float)
        aspect = np.asarray(payload["aspect"], dtype=float)
        raw_grid = np.asarray(payload["raw_grid"], dtype=float)
        gated_grid = np.asarray(payload["gated_grid"], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25))
        for col_idx, (ax, data, title) in enumerate(
            zip(
                axes,
                [raw_grid, gated_grid],
                ["uncalibrated", "calibrated"],
            )
        ):
            style_heatmap_axes(ax)
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap=SEQUENTIAL,
                extent=[aspect.min(), aspect.max(), sigmas.min(), sigmas.max()],
                interpolation="bilinear",
            )
            ax.set_xlabel("$d/n$")
            ax.set_xticks(aspect)
            if col_idx == 0:
                ax.set_ylabel(r"Noise level $\sigma$")
                ax.set_title("Raw score")
            else:
                ax.set_ylabel(r"Noise level $\sigma$")
                ax.set_yticklabels([])
                ax.set_title("Calibrated score")
            prh_colorbar(fig, ax, im)
        _save_fig(output)


def _plot_exp_b_aggregator_calibration(assets_dir: Path, *, force: bool) -> None:
    variants = _noise_variants(assets_dir, "exp_b_aggregator_calibration.npy")
    if not variants:
        src = assets_dir / "exp_b_aggregator_calibration.npy"
        if not _require_asset(src):
            return
        variants = [(DEFAULT_NOISE_TYPE, src)]

    for noise_type, src in variants:
        if not _require_asset(src):
            continue
        payload = np.load(src, allow_pickle=True).item()
        layers = payload["layers"]
        results = payload["results"]

        suffix = _noise_suffix(noise_type)
        output = assets_dir / f"exp_b_aggregator_calibration{suffix}.png"
        signal_output = assets_dir / f"exp_b_aggregator_calibration_signal{suffix}.png"
        aligned_output = (
            assets_dir / f"exp_b_aggregator_calibration_aligned{suffix}.png"
        )

        if not _should_skip([output], force):
            num_trials = payload.get("num_trials", 4)
            fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))
            for ax in axes:
                style_line_axes(ax)

            # Use consistent colors for aggregators
            agg_colors = {
                "max": LINE_COLORS[0],
                "rowmax_mean": LINE_COLORS[1],
                "colmax_mean": LINE_COLORS[2],
                "topk_5": LINE_COLORS[3],
                "topk_10": LINE_COLORS[4],
            }
            # Prettier labels for legend
            agg_labels = {
                "max": "max",
                "rowmax_mean": "row-max mean",
                "colmax_mean": "col-max mean",
                "topk_5": "top-5 mean",
                "topk_10": "top-10 mean",
            }

            layers_arr = np.asarray(layers)
            naive_line = None  # Will be set if naive/wrong calibration line is plotted
            for name, vals in results.items():
                color = agg_colors.get(name, None)
                display_name = agg_labels.get(name, name)
                raw_mean = np.asarray(vals["raw_mean"])
                gated_mean = np.asarray(vals["gated_mean"])
                raw_std = vals.get("raw_std")
                gated_std = vals.get("gated_std")

                _plot_line_with_error(
                    axes[0],
                    layers_arr,
                    raw_mean,
                    std=np.asarray(raw_std) if raw_std is not None else None,
                    num_trials=num_trials,
                    label=display_name,
                    color=color,
                    markersize=5,
                    linewidth=1.8,
                    alpha_fill=0.2,
                    clip_lower=None,
                )
                _plot_line_with_error(
                    axes[1],
                    layers_arr,
                    gated_mean,
                    std=np.asarray(gated_std) if gated_std is not None else None,
                    num_trials=num_trials,
                    label=display_name,
                    color=color,
                    markersize=5,
                    linewidth=1.8,
                    alpha_fill=0.2,
                    clip_lower=None,
                )

                # Add naive/wrong calibration for max aggregator only
                # This gets its own separate legend inside the right plot
                if name == "max":
                    naive_mean = vals.get("naive_mean")
                    naive_std = vals.get("naive_std")
                    if naive_mean is not None:
                        naive_arr = np.asarray(naive_mean)
                        # Only plot if we have valid (non-NaN) data
                        if not np.all(np.isnan(naive_arr)):
                            _plot_line_with_error(
                                axes[1],
                                layers_arr,
                                naive_arr,
                                std=(
                                    np.asarray(naive_std)
                                    if naive_std is not None
                                    else None
                                ),
                                num_trials=num_trials,
                                label="_nolegend_",  # Exclude from automatic legend
                                color="#888888",  # Grey color
                                markersize=5,
                                linewidth=1.8,
                                alpha_fill=0.2,
                                clip_lower=None,
                                linestyle="--",
                            )
                            # Store reference for separate legend
                            naive_line = axes[1].get_lines()[-1]

            # Adjust y-axis for calibrated panel to show both gated (near 0) and naive
            # Check if naive data exists and has valid values
            max_naive = results.get("max", {}).get("naive_mean")
            has_naive = max_naive is not None and not np.all(np.isnan(max_naive))
            if has_naive:
                # Use wider limits to show naive values
                naive_max = np.nanmax(max_naive)
                axes[1].set_ylim(-0.02, max(0.02, naive_max * 1.2))
            else:
                # Tighter y-axis for calibrated panel (all values near 0)
                axes[1].set_ylim(-0.02, 0.02)

            # Better titles
            # axes[0].set_title("Uncalibrated", fontsize=11, fontweight="medium")
            # axes[1].set_title("Calibrated", fontsize=11, fontweight="medium")

            # Formal axis labels
            axes[0].set_xlabel("Number of layers", fontsize=10)
            axes[1].set_xlabel("Number of layers", fontsize=10)
            axes[0].set_ylabel("Uncalibrated score", fontsize=10)
            axes[1].set_ylabel("Calibrated score", fontsize=10)
            _hide_shared_ylabel(axes)

            # Add separate legend for naive/wrong calibration line inside right plot
            if has_naive and naive_line is not None:
                axes[1].legend(
                    [naive_line],
                    ["entry-wise calibration"],  # (wrong)
                    loc="upper right",
                    fontsize=7,
                    frameon=True,
                    facecolor="white",
                    edgecolor="#C8CDD4",
                )

            # Combine handles from both axes for shared legend below
            # (naive line excluded via _nolegend_ label)
            handles0, labels0 = axes[0].get_legend_handles_labels()
            handles1, labels1 = axes[1].get_legend_handles_labels()
            # Add handles from axes[1] that aren't duplicates
            all_handles = list(handles0)
            all_labels = list(labels0)
            for h, lbl in zip(handles1, labels1):
                if lbl not in all_labels:
                    all_handles.append(h)
                    all_labels.append(lbl)
            fig.legend(
                all_handles,
                all_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(all_labels), 6),
                fontsize=8,
                frameon=True,
                facecolor="white",
                edgecolor="#C8CDD4",
                shadow=True,
            )
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.22)
            _save_fig(output)
        else:
            print(f"skip {output}")

        has_signal = any(
            "raw_mean_signal" in vals and "gated_mean_signal" in vals
            for vals in results.values()
        )
        if has_signal and not _should_skip([signal_output], force):
            fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))
            for ax in axes:
                style_line_axes(ax)

            layers_arr = np.asarray(layers)
            for name, vals in results.items():
                if "raw_mean_signal" not in vals or "gated_mean_signal" not in vals:
                    continue
                raw_signal_std = vals.get("raw_std_signal")
                gated_signal_std = vals.get("gated_std_signal")
                raw_mean = np.asarray(vals["raw_mean_signal"])
                gated_mean = np.asarray(vals["gated_mean_signal"])
                _plot_line_with_error(
                    axes[0],
                    layers_arr,
                    raw_mean,
                    std=(
                        np.asarray(raw_signal_std)
                        if raw_signal_std is not None
                        else None
                    ),
                    num_trials=num_trials,
                    label=name,
                    clip_lower=None,
                )
                _plot_line_with_error(
                    axes[1],
                    layers_arr,
                    gated_mean,
                    std=(
                        np.asarray(gated_signal_std)
                        if gated_signal_std is not None
                        else None
                    ),
                    num_trials=num_trials,
                    label=name,
                    clip_lower=None,
                )

                # Add naive/wrong calibration for max aggregator only (signal)
                if name == "max":
                    naive_mean_signal = vals.get("naive_mean_signal")
                    naive_std_signal = vals.get("naive_std_signal")
                    if naive_mean_signal is not None:
                        naive_arr = np.asarray(naive_mean_signal)
                        if not np.all(np.isnan(naive_arr)):
                            _plot_line_with_error(
                                axes[1],
                                layers_arr,
                                naive_arr,
                                std=(
                                    np.asarray(naive_std_signal)
                                    if naive_std_signal is not None
                                    else None
                                ),
                                num_trials=num_trials,
                                label="max (wrong calib.)",
                                linestyle="--",
                                clip_lower=None,
                            )
            axes[0].set_title("raw (signal)")
            axes[1].set_title("gated-rescaled (signal)")
            for ax in axes:
                ax.set_xlabel("num layers")
                ax.set_ylabel("score")
            _hide_shared_ylabel(axes)
            # Combine handles from both axes (naive line is only on axes[1])
            handles0, labels0 = axes[0].get_legend_handles_labels()
            handles1, labels1 = axes[1].get_legend_handles_labels()
            all_handles = list(handles0)
            all_labels = list(labels0)
            for h, lbl in zip(handles1, labels1):
                if lbl not in all_labels:
                    all_handles.append(h)
                    all_labels.append(lbl)
            fig.legend(
                all_handles,
                all_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=min(len(all_labels), 4),
                fontsize=8,
                frameon=True,
                facecolor="white",
                edgecolor="#C8CDD4",
                shadow=True,
            )
            _save_fig(signal_output)
        elif has_signal:
            print(f"skip {signal_output}")

        if all(
            "raw_mean_aligned" in vals and "gated_mean_aligned" in vals
            for vals in results.values()
        ):
            if _should_skip([aligned_output], force):
                print(f"skip {aligned_output}")
                continue
            fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))
            for ax in axes:
                style_line_axes(ax)

            layers_arr = np.asarray(layers)
            for name, vals in results.items():
                raw_aligned_std = vals.get("raw_std_aligned")
                gated_aligned_std = vals.get("gated_std_aligned")
                raw_mean = np.asarray(vals["raw_mean_aligned"])
                gated_mean = np.asarray(vals["gated_mean_aligned"])
                _plot_line_with_error(
                    axes[0],
                    layers_arr,
                    raw_mean,
                    std=(
                        np.asarray(raw_aligned_std)
                        if raw_aligned_std is not None
                        else None
                    ),
                    num_trials=num_trials,
                    label=f"{name} (aligned)",
                    linestyle="--",
                    clip_lower=None,
                )
                _plot_line_with_error(
                    axes[1],
                    layers_arr,
                    gated_mean,
                    std=(
                        np.asarray(gated_aligned_std)
                        if gated_aligned_std is not None
                        else None
                    ),
                    num_trials=num_trials,
                    label=f"{name} (aligned)",
                    linestyle="--",
                    clip_lower=None,
                )

                # Add naive/wrong calibration for max aggregator only (aligned)
                if name == "max":
                    naive_mean_aligned = vals.get("naive_mean_aligned")
                    naive_std_aligned = vals.get("naive_std_aligned")
                    if naive_mean_aligned is not None:
                        naive_arr = np.asarray(naive_mean_aligned)
                        if not np.all(np.isnan(naive_arr)):
                            _plot_line_with_error(
                                axes[1],
                                layers_arr,
                                naive_arr,
                                std=(
                                    np.asarray(naive_std_aligned)
                                    if naive_std_aligned is not None
                                    else None
                                ),
                                num_trials=num_trials,
                                label="max (wrong calib.)",
                                linestyle=":",
                                clip_lower=None,
                            )
            axes[0].set_title("raw (aligned)")
            axes[1].set_title("gated-rescaled (aligned)")
            for ax in axes:
                ax.set_xlabel("num layers")
                ax.set_ylabel("score")
            _hide_shared_ylabel(axes)
            # Combine handles from both axes (naive line is only on axes[1])
            handles0, labels0 = axes[0].get_legend_handles_labels()
            handles1, labels1 = axes[1].get_legend_handles_labels()
            all_handles = list(handles0)
            all_labels = list(labels0)
            for h, lbl in zip(handles1, labels1):
                if lbl not in all_labels:
                    all_handles.append(h)
                    all_labels.append(lbl)
            fig.legend(
                all_handles,
                all_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=min(len(all_labels), 4),
                fontsize=8,
                frameon=True,
                facecolor="white",
                edgecolor="#C8CDD4",
                shadow=True,
            )
            _save_fig(aligned_output)


def _plot_prh_alignment(assets_dir: Path, *, force: bool) -> None:
    """Plot PRH alignment in the style of PRH paper Figure 13.

    Creates a multi-column figure where each column shows a vision model family
    (ImageNet21K, MAE, DINOv2, CLIP, CLIP ft). Within each column, different
    model sizes are shown with viridis gradient colors. Solid lines show raw
    alignment scores, dashed lines show gated scores.
    """
    payloads = _iter_prh_alignment_payloads(assets_dir)
    if not payloads:
        return
    multi_metric = len(payloads) > 1

    for _, payload, metric_name, k_value, sigma_value in payloads:
        raw_base = np.asarray(payload["scores"], dtype=float)
        gated_base = np.asarray(payload["gated"], dtype=float)
        fdr_mask = payload.get("fdr_mask")
        fdr_mask_arr = (
            np.asarray(fdr_mask, dtype=float) if fdr_mask is not None else None
        )
        pvalues_base = None
        pvalues_payload = payload.get("pvalues")
        if pvalues_payload is not None:
            pvalues_base = np.asarray(pvalues_payload, dtype=float)

        lang_labels, vis_labels = _get_prh_model_labels(
            raw_base.shape,
            shorten_lang=True,
        )
        lang_labels_base = list(lang_labels)

        raw = raw_base.copy()
        gated = gated_base.copy()
        llm_names: list[str] | None = None
        order: list[int] | None = None

        # Sort language models by size (small -> large) when names are available.
        try:
            from aristotelian.prh.prh_models import get_models

            llm_raw, _ = get_models("val", modality="all")
            if len(llm_raw) == raw.shape[0]:
                llm_names = llm_raw
            if len(llm_raw) == raw.shape[0]:

                def parse_size_b(name: str) -> float | None:
                    lower = name.lower()
                    match = re.search(
                        r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*b", lower
                    )
                    if match:
                        return float(match.group(1)) * float(match.group(2))
                    match = re.search(r"(\d+)\s*b\s*(\d+)", lower)
                    if match:
                        return float(f"{match.group(1)}.{match.group(2)}")
                    match = re.search(r"(\d+(?:\.\d+)?)\s*b", lower)
                    if match:
                        return float(match.group(1))
                    match = re.search(r"(\d+(?:\.\d+)?)\s*m", lower)
                    if match:
                        return float(match.group(1)) / 1000.0
                    return None

                order = list(range(len(llm_raw)))
                sizes = [parse_size_b(name) for name in llm_raw]
                if any(size is not None for size in sizes):
                    order = sorted(
                        order,
                        key=lambda i: (
                            float("inf") if sizes[i] is None else sizes[i],
                            i,
                        ),
                    )
                    raw = raw[order, :]
                    gated = gated[order, :]
                    lang_labels = [lang_labels[i] for i in order]
        except Exception:
            pass

        _fdr_mask_sorted = fdr_mask_arr  # noqa: F841
        if fdr_mask_arr is not None and order is not None:
            _fdr_mask_sorted = fdr_mask_arr[order, :]  # noqa: F841

        x = np.arange(len(lang_labels))

        # Define vision model groups matching PRH paper Figure 13 column order
        # Each group: (title, name_matcher, size_order for sorting small→large)
        vision_groups = [
            ("INet21K", lambda n: "augreg" in n, ["tiny", "small", "base", "large"]),
            ("MAE", lambda n: ".mae" in n, ["base", "large", "huge"]),
            ("DINOv2", lambda n: "dinov2" in n, ["small", "base", "large", "giant"]),
            (
                "CLIP",
                lambda n: "clip" in n and "ft_in12k" not in n,
                ["base", "large", "huge"],
            ),
            ("CLIP (INet ft)", lambda n: "ft_in12k" in n, ["base", "large", "huge"]),
        ]

        # Build groups with indices into the data arrays
        groups = []
        try:
            from aristotelian.prh.prh_models import get_models

            _, vision_raw = get_models("val", modality="all")
            if len(vision_raw) == raw.shape[1]:
                for title, matcher, size_order in vision_groups:
                    indices = [i for i, name in enumerate(vision_raw) if matcher(name)]
                    if indices:
                        # Sort indices by model size (small to large)
                        def size_key(idx):
                            name = vision_raw[idx].lower()
                            for rank, size in enumerate(size_order):
                                if size in name:
                                    return rank
                            return len(size_order)

                        indices = sorted(indices, key=size_key)
                        groups.append((title, indices))
        except Exception:
            pass

        if not groups:
            # Fallback: single group with all vision models
            groups = [("Vision models", list(range(raw.shape[1])))]

        llm_family_names = llm_names or lang_labels_base
        llm_family_groups = _group_llm_indices_by_family(llm_family_names)
        if not llm_family_groups:
            llm_family_groups = [("all", list(range(raw_base.shape[0])))]

        order_family: list[int] = []
        family_ranges: list[tuple[int, int, np.ndarray]] = []
        x_positions: list[int] = []
        pos = 0
        gap = 1
        for _, indices in llm_family_groups:
            if not indices:
                continue
            start = len(order_family)
            order_family.extend(indices)
            group_len = len(indices)
            x_group = np.arange(pos, pos + group_len)
            x_positions.extend(x_group.tolist())
            family_ranges.append((start, start + group_len, x_group))
            pos = int(x_group[-1]) + 1 + gap

        if not order_family:
            order_family = list(range(raw_base.shape[0]))
            family_ranges = [(0, len(order_family), np.arange(len(order_family)))]
            x_positions = list(range(len(order_family)))

        raw_family = raw_base[order_family, :]
        gated_family = gated_base[order_family, :]
        _lang_labels_family = [lang_labels_base[i] for i in order_family]  # noqa: F841
        x_family = np.asarray(x_positions)
        fdr_mask_family = None
        if fdr_mask_arr is not None:
            fdr_mask_family = fdr_mask_arr[order_family, :]
        pvalues_family = None
        if pvalues_base is not None:
            pvalues_family = pvalues_base[order_family, :]

        def _render_alignment(gated_scores: np.ndarray, output: Path) -> None:
            n_cols = len(groups)
            fig, axes = plt.subplots(
                1, n_cols, figsize=(2.2 * n_cols, 2.5), sharey=False
            )
            if n_cols == 1:
                axes = [axes]

            for ax, (title, indices) in zip(axes, groups):
                style_line_axes(ax)

                # Use viridis gradient for model sizes (PRH paper style: purple→yellow)
                n_models = len(indices)
                colors = get_fig3_colors(n_models)

                # Compute per-panel min/max across all series (raw + gated)
                series_raw = raw[:, indices]
                series_gated = gated_scores[:, indices]
                y_values = np.concatenate([series_raw.ravel(), series_gated.ravel()])
                y_values = y_values[np.isfinite(y_values)]
                if y_values.size:
                    ymin = float(np.min(y_values))
                    ymax = float(np.max(y_values))
                else:
                    ymin, ymax = 0.0, 1.0
                pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
                ax.set_ylim(ymin - pad, ymax + pad)
                ax.tick_params(axis="y", labelleft=True)

                for series_idx, idx in enumerate(indices):
                    color = colors[series_idx]
                    label = vis_labels[idx]
                    label = label.replace("ViT-", "").split("_")[0]
                    # Solid line for calibrated (gated) - our main contribution
                    ax.plot(
                        x,
                        gated_scores[:, idx],
                        "o",
                        color=color,
                        linewidth=1,
                        markersize=3,
                        label=label,
                        linestyle="-",
                    )
                    # Dotted line for uncalibrated (raw)
                    ax.plot(
                        x,
                        raw[:, idx],
                        "d",
                        color=color,
                        linewidth=1,
                        markersize=3,
                        linestyle=":",
                    )

                ax.set_ylabel(f"Alignment to {title}")
                ax.set_xticks(x)
                ax.set_xticklabels(lang_labels, rotation=60, ha="right", fontsize=6)
                prh_legend(ax, style="rounded", fontsize=6, loc="upper left")

            # Add shared legend below for line styles (solid=calibrated, dotted=uncalibrated)
            from matplotlib.lines import Line2D

            style_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=1,
                    linestyle="-",
                    marker="o",
                    markersize=3,
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=1,
                    linestyle=":",
                    marker="d",
                    markersize=3,
                ),
            ]
            style_labels = ["calibrated", "uncalibrated"]
            fig.legend(
                style_handles,
                style_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=2,
                fontsize=7,
                frameon=True,
                facecolor="white",
                edgecolor="#C8CDD4",
                shadow=True,
            )
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.18)
            _save_fig(output)

        def _render_alignment_by_family(
            raw_scores: np.ndarray,
            gated_scores: np.ndarray,
            output: Path,
            groups_to_plot: list | None = None,
            ylabel_as_title: bool = False,
        ) -> None:
            from matplotlib.patches import Patch

            plot_groups = groups_to_plot if groups_to_plot is not None else groups
            n_cols = len(plot_groups)
            fig, axes = plt.subplots(
                1,
                n_cols,
                figsize=(2.0 * n_cols, 2.0),
                sharey=False,
            )
            if n_cols == 1:
                axes = [axes]

            # Create individual model labels with sizes for x-axis
            individual_labels = []
            for i in order_family:
                label = lang_labels_base[i]
                # Shorten and prettify: "LLaMA-7B" -> "7B", "BLOOM-560M" -> "560M"
                parts = label.split("-")
                if len(parts) >= 2:
                    size_part = parts[-1]
                    individual_labels.append(size_part)
                else:
                    individual_labels.append(label)

            # Compute x-tick positions at family group centers for cleaner labels
            family_centers = []
            family_names = []
            for fam_name, fam_indices in llm_family_groups:
                if not fam_indices:
                    continue
                # Find where this family is in the x positions
                start_idx = order_family.index(fam_indices[0])
                end_idx = order_family.index(fam_indices[-1])
                center_x = (x_family[start_idx] + x_family[end_idx]) / 2
                family_centers.append(center_x)
                # Shorten family name with proper capitalization
                lower_name = fam_name.lower().replace("-", "").replace("_", "")
                if "openllama" in lower_name:
                    short_name = "OpenLLaMA"
                elif "llama" in lower_name:
                    short_name = "LLaMA"
                elif "bloom" in lower_name:
                    short_name = "BLOOM"
                elif "olmo" in lower_name:
                    short_name = "OLMo"
                elif "gemma" in lower_name:
                    short_name = "Gemma"
                elif "mistral" in lower_name:
                    short_name = "Mistral"
                else:
                    short_name = fam_name.capitalize()
                family_names.append(short_name)

            for col_idx, (ax, (title, indices)) in enumerate(zip(axes, plot_groups)):
                style_line_axes(ax)

                n_models = len(indices)
                colors = get_fig3_colors(n_models)

                series_raw = raw_scores[:, indices]
                series_gated = gated_scores[:, indices]
                y_values = np.concatenate([series_raw.ravel(), series_gated.ravel()])
                y_values = y_values[np.isfinite(y_values)]
                if y_values.size:
                    ymin = float(np.min(y_values))
                    ymax = float(np.max(y_values))
                else:
                    ymin, ymax = 0.0, 1.0
                pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.05
                ax.set_ylim(ymin - pad, ymax + pad)
                ax.tick_params(axis="y", labelleft=True)

                for series_idx, idx in enumerate(indices):
                    color = colors[series_idx]
                    label = vis_labels[idx]
                    label = label.replace("ViT-", "").split("_")[0]

                    y_raw = raw_scores[:, idx]
                    y_gated = gated_scores[:, idx]
                    for start, end, x_group in family_ranges:
                        # Solid line for calibrated (gated) - our main contribution
                        ax.plot(
                            x_group,
                            y_gated[start:end],
                            "o-",
                            color=color,
                            linewidth=1.5,
                            markersize=4,
                            label=label if start == family_ranges[0][0] else None,
                        )
                        # Dotted line for uncalibrated (raw)
                        ax.plot(
                            x_group,
                            y_raw[start:end],
                            "d:",
                            color=color,
                            linewidth=1.5,
                            markersize=4,
                        )
                    # Connect family groups with semi-transparent lines
                    for i in range(len(family_ranges) - 1):
                        _, end1, x_group1 = family_ranges[i]
                        start2, _, x_group2 = family_ranges[i + 1]
                        # Connect gated (solid)
                        ax.plot(
                            [x_group1[-1], x_group2[0]],
                            [y_gated[end1 - 1], y_gated[start2]],
                            "-",
                            color=color,
                            linewidth=1.5,
                            alpha=0.3,
                        )
                        # Connect raw (dotted)
                        ax.plot(
                            [x_group1[-1], x_group2[0]],
                            [y_raw[end1 - 1], y_raw[start2]],
                            ":",
                            color=color,
                            linewidth=1.5,
                            alpha=0.3,
                        )

                # Set title and y-axis label
                if ylabel_as_title:
                    # No title, use y-axis label for each panel
                    ax.set_ylabel(f"Alignment to {title}", fontsize=9)
                else:
                    ax.set_title(title, fontsize=9)
                    # Y-axis label only on first panel
                    if col_idx == 0:
                        ax.set_ylabel("Alignment score", fontsize=9)

                # X-axis: individual model sizes with family names as secondary labels
                ax.set_xticks(x_family)
                ax.set_xticklabels(
                    individual_labels, fontsize=5, rotation=45, ha="right"
                )

                # Add family name labels below
                for center, fam_name in zip(family_centers, family_names):
                    ax.annotate(
                        fam_name,
                        xy=(center, 0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -18),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=6,
                        fontweight="medium",
                        color="#324766",  # PRH_LABEL color
                    )

                # Add subtle vertical separators between families
                for i, (start, end, x_group) in enumerate(family_ranges[:-1]):
                    if i + 1 < len(family_ranges):
                        next_start = family_ranges[i + 1][2][0]
                        sep_x = (x_group[-1] + next_start) / 2
                        ax.axvline(sep_x, color="#E0E0E0", linewidth=0.8, zorder=0)

                # Create legend with colored rectangles instead of lines
                legend_handles = []
                legend_labels_list = []
                for series_idx, idx in enumerate(indices):
                    color = colors[series_idx]
                    label = vis_labels[idx]
                    label = label.replace("ViT-", "").split("_")[0]
                    legend_handles.append(Patch(facecolor=color, edgecolor="none"))
                    legend_labels_list.append(label)
                ax.legend(
                    legend_handles,
                    legend_labels_list,
                    loc="best",
                    fontsize=5,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    facecolor="white",
                    edgecolor="#C8CDD4",
                    framealpha=0.9,
                    handlelength=0.8,
                    handleheight=0.6,
                    labelspacing=0.3,
                    borderpad=0.3,
                )

            # Add shared legend below for line styles (solid=calibrated, dotted=uncalibrated)
            from matplotlib.lines import Line2D

            style_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=1.5,
                    linestyle="-",
                    marker="o",
                    markersize=4,
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=1.5,
                    linestyle=":",
                    marker="d",
                    markersize=4,
                ),
            ]
            style_labels = ["calibrated", "uncalibrated"]
            fig.legend(
                style_handles,
                style_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=2,
                fontsize=8,
                frameon=True,
                facecolor="white",
                edgecolor="#C8CDD4",
                shadow=True,
            )
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.26)
            _save_fig(output)

        def _render_pvalues_by_family(
            pvalues: np.ndarray,
            output: Path,
            groups_to_plot: list | None = None,
        ) -> None:
            """Render p-values by vision model family (calibrated only)."""
            from matplotlib.patches import Patch

            plot_groups = groups_to_plot if groups_to_plot is not None else groups
            n_cols = len(plot_groups)
            fig, axes = plt.subplots(
                1,
                n_cols,
                figsize=(2.0 * n_cols, 2.0),
                sharey=False,
            )
            if n_cols == 1:
                axes = [axes]

            # Create individual model labels with sizes for x-axis
            individual_labels = []
            for i in order_family:
                label = lang_labels_base[i]
                # Shorten and prettify: "LLaMA-7B" -> "7B", "BLOOM-560M" -> "560M"
                parts = label.split("-")
                if len(parts) >= 2:
                    size_part = parts[-1]
                    individual_labels.append(size_part)
                else:
                    individual_labels.append(label)

            # Compute x-tick positions at family group centers for cleaner labels
            family_centers = []
            family_names = []
            for fam_name, fam_indices in llm_family_groups:
                if not fam_indices:
                    continue
                # Find where this family is in the x positions
                start_idx = order_family.index(fam_indices[0])
                end_idx = order_family.index(fam_indices[-1])
                center_x = (x_family[start_idx] + x_family[end_idx]) / 2
                family_centers.append(center_x)
                # Shorten family name with proper capitalization
                lower_name = fam_name.lower().replace("-", "").replace("_", "")
                if "openllama" in lower_name:
                    short_name = "OpenLLaMA"
                elif "llama" in lower_name:
                    short_name = "LLaMA"
                elif "bloom" in lower_name:
                    short_name = "BLOOM"
                elif "olmo" in lower_name:
                    short_name = "OLMo"
                elif "gemma" in lower_name:
                    short_name = "Gemma"
                elif "mistral" in lower_name:
                    short_name = "Mistral"
                else:
                    short_name = fam_name.capitalize()
                family_names.append(short_name)

            for col_idx, (ax, (title, indices)) in enumerate(zip(axes, plot_groups)):
                style_line_axes(ax)

                n_models = len(indices)
                colors = get_fig3_colors(n_models)

                # Use log scale for p-values to see variability
                ax.set_yscale("log")
                ax.tick_params(axis="y", labelleft=True)

                for series_idx, idx in enumerate(indices):
                    color = colors[series_idx]
                    label = vis_labels[idx]
                    label = label.replace("ViT-", "").split("_")[0]

                    y_pval = pvalues[:, idx]
                    # Add small x-offset so overlapping lines are visible (p-values stay accurate)
                    x_offset = 0.15 * (series_idx - (n_models - 1) / 2)
                    for start, end, x_group in family_ranges:
                        x_jittered = x_group + x_offset
                        # Solid line for calibrated p-values
                        ax.plot(
                            x_jittered,
                            y_pval[start:end],
                            "o-",
                            color=color,
                            linewidth=1.5,
                            markersize=4,
                            label=label if start == family_ranges[0][0] else None,
                        )
                    # Connect family groups with semi-transparent lines
                    for i in range(len(family_ranges) - 1):
                        _, end1, x_group1 = family_ranges[i]
                        start2, _, x_group2 = family_ranges[i + 1]
                        ax.plot(
                            [x_group1[-1] + x_offset, x_group2[0] + x_offset],
                            [y_pval[end1 - 1], y_pval[start2]],
                            "-",
                            color=color,
                            linewidth=1.5,
                            alpha=0.3,
                        )

                # Set title and y-axis label
                ax.set_title(title, fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel("$p$-value (log scale)", fontsize=9)

                # Add significance threshold line at p=0.05
                ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, zorder=0)

                # X-axis: individual model sizes with family names as secondary labels
                ax.set_xticks(x_family)
                ax.set_xticklabels(
                    individual_labels, fontsize=5, rotation=45, ha="right"
                )

                # Add family name labels below
                for center, fam_name in zip(family_centers, family_names):
                    ax.annotate(
                        fam_name,
                        xy=(center, 0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -18),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=6,
                        fontweight="medium",
                        color="#324766",  # PRH_LABEL color
                    )

                # Add subtle vertical separators between families
                for i, (start, end, x_group) in enumerate(family_ranges[:-1]):
                    if i + 1 < len(family_ranges):
                        next_start = family_ranges[i + 1][2][0]
                        sep_x = (x_group[-1] + next_start) / 2
                        ax.axvline(sep_x, color="#E0E0E0", linewidth=0.8, zorder=0)

                # Create legend with colored rectangles and significance line
                from matplotlib.lines import Line2D

                legend_handles = []
                legend_labels_list = []
                for series_idx, idx in enumerate(indices):
                    color = colors[series_idx]
                    label = vis_labels[idx]
                    label = label.replace("ViT-", "").split("_")[0]
                    legend_handles.append(Patch(facecolor=color, edgecolor="none"))
                    legend_labels_list.append(label)
                # Add significance threshold to legend
                legend_handles.append(
                    Line2D([0], [0], color="black", linestyle="--", linewidth=1.0)
                )
                legend_labels_list.append(r"$\alpha = 0.05$")
                ax.legend(
                    legend_handles,
                    legend_labels_list,
                    loc="best",
                    fontsize=5,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    facecolor="white",
                    edgecolor="#C8CDD4",
                    framealpha=0.9,
                    handlelength=0.8,
                    handleheight=0.6,
                    labelspacing=0.3,
                    borderpad=0.3,
                )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.18)
            _save_fig(output)

        # Build suffix including k value for kNN metrics and sigma for RBF metrics
        if multi_metric:
            if k_value is not None:
                suffix = f"_{_slug(metric_name)}_k{k_value}"
            elif sigma_value is not None and sigma_value != 1.0:
                suffix = f"_{_slug(metric_name)}_sigma{sigma_value}"
            else:
                suffix = f"_{_slug(metric_name)}"
        else:
            suffix = ""
        # Full version (appendix) - all vision model families
        output_family = assets_dir / f"prh_alignment_lines_by_family{suffix}.pdf"
        if _should_skip([output_family], force):
            print(f"skip {output_family}")
        else:
            _render_alignment_by_family(raw_family, gated_family, output_family)

        # P-values by family plot (calibrated only)
        output_pvalues_family = assets_dir / f"prh_pvalues_lines_by_family{suffix}.pdf"
        if pvalues_family is not None and not _should_skip(
            [output_pvalues_family], force
        ):
            _render_pvalues_by_family(pvalues_family, output_pvalues_family)

        # Main paper version - only DINOv2, CLIP, INet21K (no p-values, ylabel as title)
        groups_main = [g for g in groups if g[0] in MAIN_VISION_FAMILIES]
        output_family_main = (
            assets_dir / f"prh_alignment_lines_by_family{suffix}_main.pdf"
        )
        if groups_main and not _should_skip([output_family_main], force):
            _render_alignment_by_family(
                raw_family,
                gated_family,
                output_family_main,
                groups_to_plot=groups_main,
                ylabel_as_title=True,
            )

        # Figure 1 version: simplified single-panel plot with small/medium/large vision models
        output_fig1 = assets_dir / f"prh_alignment_fig1{suffix}.pdf"
        if not _should_skip([output_fig1], force):
            from matplotlib.patches import Patch

            # Categorize vision models into small, medium, large based on their position
            # within each family (first = small, middle = medium, last = large)
            size_categories = {"small": [], "medium": [], "large": []}
            for _, indices in groups:
                n = len(indices)
                if n >= 1:
                    size_categories["small"].append(indices[0])
                if n >= 2:
                    size_categories["large"].append(indices[-1])
                if n >= 3:
                    mid_idx = n // 2
                    size_categories["medium"].append(indices[mid_idx])

            # Use continuous x-axis (no gaps) - just use order_family indices directly
            x_continuous = np.arange(len(order_family))

            # Less wide figure for better aspect ratio
            fig, ax = plt.subplots(figsize=(3.25, 2.5))
            style_line_axes(ax)

            # Use viridis gradient to match other PRH experiments
            fig3_colors = get_fig3_colors(3)
            size_colors = {
                "small": fig3_colors[0],
                "medium": fig3_colors[1],
                "large": fig3_colors[2],
            }
            size_labels = {
                "small": "Small vision model",
                "medium": "Medium vision model",
                "large": "Large vision model",
            }

            for size_name in ["small", "medium", "large"]:
                vis_indices = size_categories[size_name]
                if not vis_indices:
                    continue
                # Average alignment across vision models in this size category
                y_raw = raw_family[:, vis_indices].mean(axis=1)
                y_gated = gated_family[:, vis_indices].mean(axis=1)

                color = size_colors[size_name]
                # Solid line for calibrated (gated) - our main contribution
                ax.plot(
                    x_continuous,
                    y_gated,
                    "o-",
                    color=color,
                    linewidth=2.0,
                    markersize=5,
                )
                # Dotted line for uncalibrated (raw)
                ax.plot(
                    x_continuous,
                    y_raw,
                    "d:",
                    color=color,
                    linewidth=2.0,
                    markersize=5,
                )

            ax.set_xlabel("Language model capacity")
            ax.set_ylabel("Alignment score")
            # Use simple integer ticks (no individual labels)
            ax.set_xticks([])

            # Create legend with colored rectangles (not lines) for model sizes
            legend_handles = []
            legend_labels_list = []
            for size_name in ["small", "medium", "large"]:
                if size_categories[size_name]:
                    legend_handles.append(
                        Patch(facecolor=size_colors[size_name], edgecolor="none")
                    )
                    legend_labels_list.append(size_labels[size_name])
            ax.legend(
                legend_handles,
                legend_labels_list,
                loc="upper left",
                fontsize=6,
                frameon=True,
                fancybox=True,
                shadow=True,
                facecolor="white",
                edgecolor="#C8CDD4",
                framealpha=0.9,
                handlelength=0.8,
                handleheight=0.6,
                labelspacing=0.3,
                borderpad=0.3,
            )
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.20)
            _save_fig(output_fig1)

        output_tau = assets_dir / f"prh_alignment_tau{suffix}.pdf"
        if not _should_skip([output_tau], force):
            fig, ax = plt.subplots(figsize=(4.5, 2.4))
            style_line_axes(ax)
            tau_by_vis = np.asarray(payload["taus"], dtype=float).mean(axis=0)
            ax.plot(np.arange(len(tau_by_vis)), tau_by_vis, "o-", color=LINE_COLORS[2])
            ax.set_xlabel("VISION model index")
            ax.set_ylabel("tau_alpha (mean over language)")
            ax.set_title("PRH tau_alpha by vision model")
            _save_fig(output_tau)

        output_tau_family = assets_dir / f"prh_alignment_tau_by_family{suffix}.pdf"
        if not _should_skip([output_tau_family], force):
            tau_by_vis = np.asarray(payload["taus"], dtype=float).mean(axis=0)
            n_cols = len(groups)
            fig, axes = plt.subplots(
                1, n_cols, figsize=(2.2 * n_cols, 2.4), sharey=False
            )
            if n_cols == 1:
                axes = [axes]
            for ax, (title, indices) in zip(axes, groups):
                style_line_axes(ax)
                tau_vals = tau_by_vis[indices]
                ax.plot(np.arange(len(tau_vals)), tau_vals, "o-", color=LINE_COLORS[2])
                ax.set_title(title)
                ax.set_xlabel("model size order")
                ax.set_ylabel("tau_alpha")
            _save_fig(output_tau_family)

        if fdr_mask is None:
            continue

        output_fdr_family = (
            assets_dir / f"prh_alignment_lines_by_family_fdr{suffix}.pdf"
        )
        if _should_skip([output_fdr_family], force):
            print(f"skip {output_fdr_family}")
            continue
        gated_fdr_family = gated_family * fdr_mask_family
        _render_alignment_by_family(raw_family, gated_fdr_family, output_fdr_family)


# -----------------------------------------------------------------------------
# V2T (Video-to-Text) alignment plots
# -----------------------------------------------------------------------------


def _iter_v2t_alignment_payloads(
    assets_dir: Path, video_modelset: str = "videoprh", text_modelset: str = "videoprh"
) -> list[tuple[Path, dict, str]]:
    """Iterate over V2T alignment payloads across all metrics."""
    base_pattern = f"v2t_alignment_{video_modelset}_{text_modelset}*.npy"
    candidates = sorted(assets_dir.glob(base_pattern))
    if not candidates:
        print(
            f"skip missing {assets_dir / f'v2t_alignment_{video_modelset}_{text_modelset}.npy'}"
        )
        return []
    payloads = []
    base_name = f"v2t_alignment_{video_modelset}_{text_modelset}"
    for path in candidates:
        if not path.is_file():
            continue
        payload = np.load(path, allow_pickle=True).item()
        # Always derive metric from filename to capture k variants
        # e.g., "v2t_alignment_videoprh_videoprh_mutual_knn_k20" -> "mutual_knn_k20"
        if path.stem == base_name:
            metric = "mutual_knn"
        else:
            metric = path.stem.replace(f"{base_name}_", "")
        payloads.append((path, payload, metric))
    return payloads


def _shorten_v2t_video_label(name: str) -> str:
    """Shorten video model names for V2T plots.

    Returns just the size (small, base, large, huge, giant) since the
    model family is indicated by the y-axis label.
    """
    label = name.split("/")[-1].lower()

    # Extract size from the name
    for size in ["small", "base", "large", "huge", "giant"]:
        if size in label:
            return size

    return label


def _shorten_v2t_text_label(name: str) -> str:
    """Shorten text model names for V2T plots."""
    label = name.split("/")[-1]
    label = label.replace("bloomz-", "bloom-")
    label = label.replace("open_llama_", "ollama-")
    label = label.replace("Meta-Llama-3-", "llama3-")
    return label


def _infer_video_family(name: str) -> str:
    """Infer video model family from name."""
    lower = name.lower()
    if "videomae" in lower:
        return "videomae"
    if "dinov2" in lower or ".lvd142m" in lower:
        return "dinov2"
    if "clip" in lower or "laion" in lower:
        return "clip"
    return "other"


def _group_video_indices_by_family(
    model_names: list[str],
) -> list[tuple[str, list[int]]]:
    """Group video model indices by family, sorted by size within each."""
    family_order = ["videomae", "dinov2", "clip", "other"]
    size_order = ["small", "base", "large", "huge", "giant"]

    family_map: dict[str, list[int]] = {}
    for idx, name in enumerate(model_names):
        family = _infer_video_family(name)
        family_map.setdefault(family, []).append(idx)

    def size_key(idx: int) -> tuple[int, int]:
        name = model_names[idx].lower()
        for rank, size in enumerate(size_order):
            if size in name:
                return (rank, idx)
        return (len(size_order), idx)

    groups: list[tuple[str, list[int]]] = []
    for family in family_order:
        indices = family_map.pop(family, [])
        if indices:
            indices = sorted(indices, key=size_key)
            groups.append((family, indices))

    for family in sorted(family_map):
        indices = sorted(family_map[family], key=size_key)
        groups.append((family, indices))

    return groups


def _plot_v2t_alignment(assets_dir: Path, *, force: bool) -> None:
    """Plot V2T alignment in PRH paper style (line plots).

    Creates multi-column figure where each column shows a video model family.
    X-axis shows text models grouped by family. Lines show individual video models.
    Matches PRH Figure 13 style with proper legends, labels, and separators.
    """
    from matplotlib.patches import Patch

    payloads = _iter_v2t_alignment_payloads(assets_dir)
    if not payloads:
        return

    for _, payload, metric_name in payloads:
        output_name = (
            "v2t_alignment_videoprh.pdf"
            if metric_name == "mutual_knn"
            else f"v2t_alignment_videoprh_{metric_name}.pdf"
        )
        output = assets_dir / output_name
        output_main_name = (
            "v2t_alignment_videoprh_main.pdf"
            if metric_name == "mutual_knn"
            else f"v2t_alignment_videoprh_{metric_name}_main.pdf"
        )
        output_main = assets_dir / output_main_name

        # Check if both outputs already exist
        skip_main_plot = _should_skip([output], force)
        skip_main_version = _should_skip([output_main], force)
        if skip_main_plot and skip_main_version:
            print(f"skip {output} and {output_main}")
            continue
        if skip_main_plot:
            print(f"skip {output}")
        if skip_main_version:
            print(f"skip {output_main}")

        raw = np.asarray(payload["scores"], dtype=float)  # shape: (n_video, n_text)
        gated = np.asarray(payload["gated"], dtype=float)
        video_models = payload.get("video_models", [])
        text_models = payload.get("text_models", [])

        # Filter out Gemma models (out of place with the rest)
        non_gemma_indices = [
            i for i, m in enumerate(text_models) if "gemma" not in m.lower()
        ]
        if non_gemma_indices and len(non_gemma_indices) < len(text_models):
            raw = raw[:, non_gemma_indices]
            gated = gated[:, non_gemma_indices]
            text_models = [text_models[i] for i in non_gemma_indices]

        # Shorten labels
        video_labels = [_shorten_v2t_video_label(m) for m in video_models]
        text_labels = [_shorten_v2t_text_label(m) for m in text_models]

        # Group text models by LLM family for x-axis
        text_family_groups = _group_llm_indices_by_family(text_models)

        # Build x-axis order (text models grouped by family with gaps)
        order_x: list[int] = []
        x_positions: list[float] = []
        family_ranges: list[tuple[int, int, np.ndarray]] = []
        pos = 0
        gap = 1
        for _, indices in text_family_groups:
            if not indices:
                continue
            start = len(order_x)
            order_x.extend(indices)
            group_len = len(indices)
            x_group = np.arange(pos, pos + group_len, dtype=float)
            x_positions.extend(x_group.tolist())
            family_ranges.append((start, start + group_len, x_group))
            pos = int(x_group[-1]) + 1 + gap

        x_arr = np.asarray(x_positions)

        # Create individual model size labels for x-axis (e.g., "560M", "7B")
        individual_labels = []
        for i in order_x:
            label = text_labels[i]
            # Extract size part: "bloom-560m" -> "560M", "llama-7b" -> "7B"
            parts = label.replace("_", "-").split("-")
            size_part = parts[-1].upper() if len(parts) >= 2 else label
            individual_labels.append(size_part)

        # Compute family centers and names for secondary x-axis labels
        family_centers = []
        family_names = []
        for fam_name, fam_indices in text_family_groups:
            if not fam_indices:
                continue
            start_idx = order_x.index(fam_indices[0])
            end_idx = order_x.index(fam_indices[-1])
            center_x = (x_arr[start_idx] + x_arr[end_idx]) / 2
            family_centers.append(center_x)
            # Shorten family name with proper capitalization
            lower_name = fam_name.lower().replace("-", "").replace("_", "")
            if "openllama" in lower_name:
                short_name = "OpenLLaMA"
            elif "llama" in lower_name:
                short_name = "LLaMA"
            elif "bloom" in lower_name:
                short_name = "BLOOM"
            elif "olmo" in lower_name:
                short_name = "OLMo"
            elif "gemma" in lower_name:
                short_name = "Gemma"
            elif "mistral" in lower_name:
                short_name = "Mistral"
            else:
                short_name = fam_name.capitalize()
            family_names.append(short_name)

        # Reorder scores (columns) to match x-axis order
        raw_ordered = raw[:, order_x]
        gated_ordered = gated[:, order_x]

        # Group video models by family for columns
        video_family_groups = _group_video_indices_by_family(video_models)

        family_titles = {
            "videomae": "VideoMAE",
            "dinov2": "DINOv2",
            "clip": "CLIP",
            "other": "Other",
        }

        # Generate main plot (all video families)
        if not skip_main_plot:
            n_cols = len(video_family_groups)
            fig, axes = plt.subplots(
                1, n_cols, figsize=(2.0 * n_cols, 2.3), sharey=False
            )
            if n_cols == 1:
                axes = [axes]

            for col_idx, (family_name, video_indices) in enumerate(video_family_groups):
                ax = axes[col_idx]
                style_line_axes(ax)

                n_videos = len(video_indices)
                colors = get_fig3_colors(n_videos)

                # Compute y-limits
                y_vals = []
                for v_idx in video_indices:
                    y_vals.extend(raw_ordered[v_idx, :].tolist())
                    y_vals.extend(gated_ordered[v_idx, :].tolist())
                y_vals = [v for v in y_vals if np.isfinite(v)]
                if y_vals:
                    ymin, ymax = min(y_vals), max(y_vals)
                else:
                    ymin, ymax = 0.0, 1.0
                pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.05
                ax.set_ylim(ymin - pad, ymax + pad)
                ax.tick_params(axis="y", labelleft=True)

                for series_idx, v_idx in enumerate(video_indices):
                    color = colors[series_idx]
                    label = video_labels[v_idx]

                    # Plot each text family segment
                    for start, end, x_group in family_ranges:
                        y_raw = raw_ordered[v_idx, start:end]
                        y_gated = gated_ordered[v_idx, start:end]
                        # Calibrated (gated): solid line with circles
                        ax.plot(
                            x_group,
                            y_gated,
                            "o-",
                            color=color,
                            linewidth=1.5,
                            markersize=4,
                            label=label if start == family_ranges[0][0] else None,
                        )
                        # Uncalibrated (raw): dotted line with diamonds
                        ax.plot(
                            x_group,
                            y_raw,
                            "d:",
                            color=color,
                            linewidth=1.5,
                            markersize=4,
                        )
                    # Connect family groups with semi-transparent lines
                    for i in range(len(family_ranges) - 1):
                        _, end1, x_group1 = family_ranges[i]
                        start2, _, x_group2 = family_ranges[i + 1]
                        y_raw_end = raw_ordered[v_idx, end1 - 1]
                        y_raw_start = raw_ordered[v_idx, start2]
                        y_gated_end = gated_ordered[v_idx, end1 - 1]
                        y_gated_start = gated_ordered[v_idx, start2]
                        # Connect gated (solid)
                        ax.plot(
                            [x_group1[-1], x_group2[0]],
                            [y_gated_end, y_gated_start],
                            "-",
                            color=color,
                            linewidth=1.5,
                            alpha=0.3,
                        )
                        # Connect raw (dotted)
                        ax.plot(
                            [x_group1[-1], x_group2[0]],
                            [y_raw_end, y_raw_start],
                            ":",
                            color=color,
                            linewidth=1.5,
                            alpha=0.3,
                        )

                # Title with video family name
                title = family_titles.get(family_name, family_name)
                ax.set_title(title, fontsize=9)

                # Y-axis label only on first panel
                if col_idx == 0:
                    ax.set_ylabel("Alignment score", fontsize=9)

                # X-axis: individual model sizes
                ax.set_xticks(x_arr)
                ax.set_xticklabels(
                    individual_labels, fontsize=5, rotation=45, ha="right"
                )

                # Add family name labels below x-axis
                for center, fam_name in zip(family_centers, family_names):
                    ax.annotate(
                        fam_name,
                        xy=(center, 0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -18),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=6,
                        fontweight="medium",
                        color="#324766",
                    )

                # Add subtle vertical separators between text families
                for i, (start, end, x_group) in enumerate(family_ranges[:-1]):
                    if i + 1 < len(family_ranges):
                        next_start = family_ranges[i + 1][2][0]
                        sep_x = (x_group[-1] + next_start) / 2
                        ax.axvline(sep_x, color="#E0E0E0", linewidth=0.8, zorder=0)

                # Legend with colored patches (PRH style)
                legend_handles = []
                legend_labels_list = []
                for series_idx, v_idx in enumerate(video_indices):
                    color = colors[series_idx]
                    label = video_labels[v_idx]
                    legend_handles.append(Patch(facecolor=color, edgecolor="none"))
                    legend_labels_list.append(label)
                ax.legend(
                    legend_handles,
                    legend_labels_list,
                    loc="best",
                    fontsize=5,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    facecolor="white",
                    edgecolor="#C8CDD4",
                    framealpha=0.9,
                    handlelength=0.8,
                    handleheight=0.6,
                    labelspacing=0.3,
                    borderpad=0.3,
                )

            # Add shared legend below for line styles (solid=calibrated, dotted=uncalibrated)
            from matplotlib.lines import Line2D

            style_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=1.5,
                    linestyle="-",
                    marker="o",
                    markersize=4,
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=1.5,
                    linestyle=":",
                    marker="d",
                    markersize=4,
                ),
            ]
            style_labels = ["calibrated", "uncalibrated"]
            fig.legend(
                style_handles,
                style_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=2,
                fontsize=8,
                frameon=True,
                facecolor="white",
                edgecolor="#C8CDD4",
                shadow=True,
            )
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.26)
            _save_fig(output)

        # =====================================================================
        # Main paper version: 2 panels (VideoMAE, DINOv2) with y-axis as title
        # =====================================================================
        if not skip_main_version:
            MAIN_V2T_FAMILIES = {"videomae", "dinov2"}
            main_groups = [
                (fam, idx)
                for fam, idx in video_family_groups
                if fam in MAIN_V2T_FAMILIES
            ]
            if main_groups:
                n_cols_main = len(main_groups)
                fig_main, axes_main = plt.subplots(
                    1, n_cols_main, figsize=(2.2 * n_cols_main, 2.5), sharey=False
                )
                if n_cols_main == 1:
                    axes_main = [axes_main]

                for col_idx, (family_name, video_indices) in enumerate(main_groups):
                    ax = axes_main[col_idx]
                    style_line_axes(ax)

                    n_videos = len(video_indices)
                    colors = get_fig3_colors(n_videos)

                    # Compute y-limits
                    y_vals = []
                    for v_idx in video_indices:
                        y_vals.extend(raw_ordered[v_idx, :].tolist())
                        y_vals.extend(gated_ordered[v_idx, :].tolist())
                    y_vals = [v for v in y_vals if np.isfinite(v)]
                    if y_vals:
                        ymin, ymax = min(y_vals), max(y_vals)
                    else:
                        ymin, ymax = 0.0, 1.0
                    pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.05
                    ax.set_ylim(ymin - pad, ymax + pad)
                    ax.tick_params(axis="y", labelleft=True)

                    for series_idx, v_idx in enumerate(video_indices):
                        color = colors[series_idx]
                        label = video_labels[v_idx]

                        # Plot each text family segment
                        for start, end, x_group in family_ranges:
                            y_raw = raw_ordered[v_idx, start:end]
                            y_gated = gated_ordered[v_idx, start:end]
                            # Calibrated (gated): solid line with circles
                            ax.plot(
                                x_group,
                                y_gated,
                                "o-",
                                color=color,
                                linewidth=1.5,
                                markersize=4,
                                label=label if start == family_ranges[0][0] else None,
                            )
                            # Uncalibrated (raw): dashed line with diamonds
                            ax.plot(
                                x_group,
                                y_raw,
                                "d:",
                                color=color,
                                linewidth=1.5,
                                markersize=4,
                            )
                        # Connect family groups with semi-transparent lines
                        for i in range(len(family_ranges) - 1):
                            _, end1, x_group1 = family_ranges[i]
                            start2, _, x_group2 = family_ranges[i + 1]
                            y_raw_end = raw_ordered[v_idx, end1 - 1]
                            y_raw_start = raw_ordered[v_idx, start2]
                            y_gated_end = gated_ordered[v_idx, end1 - 1]
                            y_gated_start = gated_ordered[v_idx, start2]
                            ax.plot(
                                [x_group1[-1], x_group2[0]],
                                [y_gated_end, y_gated_start],
                                "-",
                                color=color,
                                linewidth=1.5,
                                alpha=0.3,
                            )
                            ax.plot(
                                [x_group1[-1], x_group2[0]],
                                [y_raw_end, y_raw_start],
                                ":",
                                color=color,
                                linewidth=1.5,
                                alpha=0.3,
                            )

                    # Y-axis label as title (main paper style)
                    title = family_titles.get(family_name, family_name)
                    ax.set_ylabel(f"Alignment to {title}", fontsize=9)

                    # X-axis: individual model sizes
                    ax.set_xticks(x_arr)
                    ax.set_xticklabels(
                        individual_labels, fontsize=5, rotation=45, ha="right"
                    )

                    # Add family name labels below x-axis
                    for center, fam_name in zip(family_centers, family_names):
                        ax.annotate(
                            fam_name,
                            xy=(center, 0),
                            xycoords=("data", "axes fraction"),
                            xytext=(0, -18),
                            textcoords="offset points",
                            ha="center",
                            va="top",
                            fontsize=6,
                            fontweight="medium",
                            color="#324766",
                        )

                    # Add subtle vertical separators between text families
                    for i, (start, end, x_group) in enumerate(family_ranges[:-1]):
                        if i + 1 < len(family_ranges):
                            next_start = family_ranges[i + 1][2][0]
                            sep_x = (x_group[-1] + next_start) / 2
                            ax.axvline(sep_x, color="#E0E0E0", linewidth=0.8, zorder=0)

                    # Legend with colored patches
                    legend_handles = []
                    legend_labels_list = []
                    for series_idx, v_idx in enumerate(video_indices):
                        color = colors[series_idx]
                        label = video_labels[v_idx]
                        legend_handles.append(Patch(facecolor=color, edgecolor="none"))
                        legend_labels_list.append(label)
                    ax.legend(
                        legend_handles,
                        legend_labels_list,
                        loc="upper left",
                        fontsize=6,
                        frameon=True,
                        fancybox=True,
                        shadow=True,
                        facecolor="white",
                        edgecolor="#C8CDD4",
                        framealpha=0.9,
                        handlelength=0.8,
                        handleheight=0.6,
                        labelspacing=0.3,
                        borderpad=0.3,
                    )

                # Add shared legend below for line styles
                from matplotlib.lines import Line2D

                style_handles = [
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=1.5,
                        linestyle="-",
                        marker="o",
                        markersize=4,
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=1.5,
                        linestyle=":",
                        marker="d",
                        markersize=4,
                    ),
                ]
                style_labels = ["calibrated", "uncalibrated"]
                fig_main.legend(
                    style_handles,
                    style_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=2,
                    fontsize=7,
                    frameon=True,
                    facecolor="white",
                    edgecolor="#C8CDD4",
                    shadow=True,
                )
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.18)
                _save_fig(output_main)

    # =========================================================================
    # Combined CKA + MkNN plot: VideoMAE with CKA RBF (left) and MkNN (right)
    # =========================================================================
    output_joint = assets_dir / "v2t_alignment_videoprh_cka_mknn_joint_main.pdf"
    if not _should_skip([output_joint], force):
        # Load CKA RBF sigma=0.5 data
        cka_path = assets_dir / "v2t_alignment_videoprh_videoprh_cka_rbf_sigma0.5.npy"
        # Load MkNN k=10 data (default metric)
        mknn_path = assets_dir / "v2t_alignment_videoprh_videoprh.npy"

        if cka_path.exists() and mknn_path.exists():
            cka_payload = np.load(cka_path, allow_pickle=True).item()
            mknn_payload = np.load(mknn_path, allow_pickle=True).item()

            # Process CKA data
            cka_raw = np.asarray(cka_payload["scores"], dtype=float)
            cka_gated = np.asarray(cka_payload["gated"], dtype=float)
            video_models = cka_payload.get("video_models", [])
            text_models = cka_payload.get("text_models", [])

            # Filter out Gemma
            non_gemma_idx = [
                i for i, m in enumerate(text_models) if "gemma" not in m.lower()
            ]
            if non_gemma_idx and len(non_gemma_idx) < len(text_models):
                cka_raw = cka_raw[:, non_gemma_idx]
                cka_gated = cka_gated[:, non_gemma_idx]
                text_models = [text_models[i] for i in non_gemma_idx]

            # Process MkNN data (same filtering)
            mknn_raw = np.asarray(mknn_payload["scores"], dtype=float)
            mknn_gated = np.asarray(mknn_payload["gated"], dtype=float)
            mknn_raw = mknn_raw[:, non_gemma_idx]
            mknn_gated = mknn_gated[:, non_gemma_idx]

            # Get labels
            video_labels = [_shorten_v2t_video_label(m) for m in video_models]
            text_labels = [_shorten_v2t_text_label(m) for m in text_models]

            # Group text models by family
            text_family_groups = _group_llm_indices_by_family(text_models)

            # Build x-axis
            order_x: list[int] = []
            x_positions: list[float] = []
            family_ranges: list[tuple[int, int, np.ndarray]] = []
            pos = 0
            gap = 1
            for _, indices in text_family_groups:
                if not indices:
                    continue
                start = len(order_x)
                order_x.extend(indices)
                group_len = len(indices)
                x_group = np.arange(pos, pos + group_len, dtype=float)
                x_positions.extend(x_group.tolist())
                family_ranges.append((start, start + group_len, x_group))
                pos = int(x_group[-1]) + 1 + gap

            x_arr = np.asarray(x_positions)

            # Individual labels for x-axis
            individual_labels = []
            for i in order_x:
                label = text_labels[i]
                parts = label.replace("_", "-").split("-")
                size_part = parts[-1].upper() if len(parts) >= 2 else label
                individual_labels.append(size_part)

            # Family centers and names
            family_centers = []
            family_names = []
            for fam_name, fam_indices in text_family_groups:
                if not fam_indices:
                    continue
                start_idx = order_x.index(fam_indices[0])
                end_idx = order_x.index(fam_indices[-1])
                center_x = (x_arr[start_idx] + x_arr[end_idx]) / 2
                family_centers.append(center_x)
                lower_name = fam_name.lower().replace("-", "").replace("_", "")
                if "openllama" in lower_name:
                    short_name = "OpenLLaMA"
                elif "llama" in lower_name:
                    short_name = "LLaMA"
                elif "bloom" in lower_name:
                    short_name = "BLOOM"
                else:
                    short_name = fam_name.capitalize()
                family_names.append(short_name)

            # Reorder scores
            cka_raw_ordered = cka_raw[:, order_x]
            cka_gated_ordered = cka_gated[:, order_x]
            mknn_raw_ordered = mknn_raw[:, order_x]
            mknn_gated_ordered = mknn_gated[:, order_x]

            # Get VideoMAE indices
            video_family_groups = _group_video_indices_by_family(video_models)
            videomae_indices = None
            for fam, indices in video_family_groups:
                if fam == "videomae":
                    videomae_indices = indices
                    break

            if videomae_indices:
                from matplotlib.lines import Line2D
                from matplotlib.patches import Patch

                # Create 2-panel figure
                fig_joint, axes_joint = plt.subplots(
                    1, 2, figsize=(4.0, 2.0), sharey=False
                )

                # Panel configs: (title, raw_data, gated_data, ylabel)
                panels = [
                    ("CKA", cka_raw_ordered, cka_gated_ordered, "CKA RBF"),
                    ("MkNN", mknn_raw_ordered, mknn_gated_ordered, "mKNN"),
                ]

                for col_idx, (panel_title, raw_ord, gated_ord, ylabel) in enumerate(
                    panels
                ):
                    ax = axes_joint[col_idx]
                    style_line_axes(ax)

                    n_videos = len(videomae_indices)
                    colors = get_fig3_colors(n_videos)

                    # Compute y-limits
                    y_vals = []
                    for v_idx in videomae_indices:
                        y_vals.extend(raw_ord[v_idx, :].tolist())
                        y_vals.extend(gated_ord[v_idx, :].tolist())
                    y_vals = [v for v in y_vals if np.isfinite(v)]
                    if y_vals:
                        ymin, ymax = min(y_vals), max(y_vals)
                    else:
                        ymin, ymax = 0.0, 1.0
                    pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.05
                    ax.set_ylim(ymin - pad, ymax + pad)
                    ax.tick_params(axis="y", labelleft=True)

                    for series_idx, v_idx in enumerate(videomae_indices):
                        color = colors[series_idx]
                        label = video_labels[v_idx]

                        for start, end, x_group in family_ranges:
                            y_raw = raw_ord[v_idx, start:end]
                            y_gated = gated_ord[v_idx, start:end]
                            # Calibrated: solid with circles
                            ax.plot(
                                x_group,
                                y_gated,
                                "o-",
                                color=color,
                                linewidth=1.5,
                                markersize=4,
                                label=label if start == family_ranges[0][0] else None,
                            )
                            # Uncalibrated: dashed with diamonds
                            ax.plot(
                                x_group,
                                y_raw,
                                "d:",
                                color=color,
                                linewidth=1.5,
                                markersize=4,
                            )
                        # Connect family groups with semi-transparent lines
                        for i in range(len(family_ranges) - 1):
                            _, end1, x_group1 = family_ranges[i]
                            start2, _, x_group2 = family_ranges[i + 1]
                            y_raw_end = raw_ord[v_idx, end1 - 1]
                            y_raw_start = raw_ord[v_idx, start2]
                            y_gated_end = gated_ord[v_idx, end1 - 1]
                            y_gated_start = gated_ord[v_idx, start2]
                            ax.plot(
                                [x_group1[-1], x_group2[0]],
                                [y_gated_end, y_gated_start],
                                "-",
                                color=color,
                                linewidth=1.5,
                                alpha=0.3,
                            )
                            ax.plot(
                                [x_group1[-1], x_group2[0]],
                                [y_raw_end, y_raw_start],
                                ":",
                                color=color,
                                linewidth=1.5,
                                alpha=0.3,
                            )

                    ax.set_ylabel(ylabel, fontsize=9)
                    ax.set_xticks(x_arr)
                    ax.set_xticklabels(
                        individual_labels, fontsize=5, rotation=45, ha="right"
                    )

                    # Family name labels below x-axis
                    for center, fam_name in zip(family_centers, family_names):
                        ax.annotate(
                            fam_name,
                            xy=(center, 0),
                            xycoords=("data", "axes fraction"),
                            xytext=(0, -18),
                            textcoords="offset points",
                            ha="center",
                            va="top",
                            fontsize=6,
                            fontweight="medium",
                            color="#324766",
                        )

                    # Vertical separators
                    for i, (start, end, x_group) in enumerate(family_ranges[:-1]):
                        if i + 1 < len(family_ranges):
                            next_start = family_ranges[i + 1][2][0]
                            sep_x = (x_group[-1] + next_start) / 2
                            ax.axvline(sep_x, color="#E0E0E0", linewidth=0.8, zorder=0)

                # Combined legend below: row 1 = model sizes (colored patches), row 2 = line styles
                # Build model size handles (colored patches)
                n_videos = len(videomae_indices)
                colors = get_fig3_colors(n_videos)
                size_handles = []
                size_labels = []
                for series_idx, v_idx in enumerate(videomae_indices):
                    color = colors[series_idx]
                    label = video_labels[v_idx]
                    size_handles.append(Patch(facecolor=color, edgecolor="none"))
                    size_labels.append(label)

                # Build line style handles
                style_handles = [
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=1.5,
                        linestyle="-",
                        marker="o",
                        markersize=4,
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=1.5,
                        linestyle=":",
                        marker="d",
                        markersize=4,
                    ),
                ]
                style_labels = ["calibrated", "uncalibrated"]

                # Single row legend: model sizes + line styles
                all_handles = size_handles + style_handles
                all_labels = size_labels + style_labels

                fig_joint.legend(
                    all_handles,
                    all_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=len(all_handles),
                    fontsize=7,
                    frameon=True,
                    facecolor="white",
                    edgecolor="#C8CDD4",
                    shadow=True,
                    handlelength=1.0,
                    handleheight=0.8,
                    columnspacing=1.0,
                )
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.20)
                _save_fig(output_joint)


SECTION_FUNCS: Dict[str, Callable[..., None]] = {
    "null_drift_gaussian": _plot_null_drift_gaussian,
    "null_drift_heavy": _plot_null_drift_heavy,
    "perm_budget": _plot_perm_budget,
    "type1_calibration": _plot_type1_calibration,
    "snr_sweep": _plot_snr_sweep,
    "type1_and_power_combined": _plot_type1_and_power_combined,
    "phase_diagram": _plot_phase_diagram,
    "exp_b_aggregator_calibration": _plot_exp_b_aggregator_calibration,
    "prh_alignment": _plot_prh_alignment,
    "v2t_alignment": _plot_v2t_alignment,
}


def _parse_sections(raw_sections: Sequence[str]) -> Sequence[str]:
    if not raw_sections:
        return list(SECTION_FUNCS.keys())
    parts = []
    for item in raw_sections:
        parts.extend(p.strip() for p in item.split(","))
    parts = [p for p in parts if p]
    if "all" in parts:
        return list(SECTION_FUNCS.keys())
    unknown = sorted(set(parts) - set(SECTION_FUNCS.keys()))
    if unknown:
        raise ValueError(f"Unknown sections: {', '.join(unknown)}")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sgCKA experiment outputs.")
    parser.add_argument(
        "--sections",
        nargs="*",
        default=["all"],
        help="Plot sections to run (comma-separated or space-separated).",
    )
    parser.add_argument(
        "--assets-dir",
        default="assets",
        help="Directory with experiment outputs and plot destinations.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite plots if they exist."
    )
    args = parser.parse_args()

    _apply_style()
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    sections = _parse_sections(args.sections)

    for section in sections:
        print(f"plot {section}")
        SECTION_FUNCS[section](assets_dir, force=args.force)


if __name__ == "__main__":
    main()
