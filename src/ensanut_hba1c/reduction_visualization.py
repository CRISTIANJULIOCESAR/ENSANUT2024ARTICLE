"""Plots for comparing PCA1D, PCA2D, UMAP2D and graph-derived PHATE2D."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _sorted_labels(labels: Iterable) -> list:
    values = pd.Series(labels).dropna().unique().tolist()
    try:
        return sorted(values, key=float)
    except Exception:
        return sorted(values, key=str)


def plot_reduction_comparison(
    table: pd.DataFrame,
    *,
    cluster_col: str = "Cluster_SHAP",
    figure_size: tuple[float, float] = (12.5, 9.5),
    point_size: float = 7.0,
    alpha: float = 0.75,
    pca1d_jitter: float = 0.06,
    random_state: int = 42,
):
    """Compare four views while clearly marking PCA1D display jitter."""
    required = {
        "PCA_1", "PCA_2", "UMAP_VISUAL_1", "UMAP_VISUAL_2",
        "UMAP_GRAPH_PHATE_1", "UMAP_GRAPH_PHATE_2", cluster_col,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Missing columns for reduction comparison: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=figure_size, constrained_layout=True)
    labels = table[cluster_col]
    clusters = _sorted_labels(labels)
    cmap = plt.get_cmap("tab20", max(1, len(clusters)))
    color_map = {cluster: cmap(index) for index, cluster in enumerate(clusters)}
    point_colors = [color_map.get(value, (0.5, 0.5, 0.5, 1.0)) for value in labels]

    rng = np.random.default_rng(random_state)
    jitter = rng.normal(0.0, pca1d_jitter, len(table))
    axes[0, 0].scatter(table["PCA_1"], jitter, c=point_colors, s=point_size, alpha=alpha, linewidths=0)
    axes[0, 0].axhline(0, linewidth=0.6)
    axes[0, 0].set_title("PCA 1D: PC1")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("Visualization jitter\n(not a dimension)")
    axes[0, 0].set_yticks([])

    axes[0, 1].scatter(table["PCA_1"], table["PCA_2"], c=point_colors, s=point_size, alpha=alpha, linewidths=0)
    axes[0, 1].set_title("PCA 2D: PC1–PC2")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")

    axes[1, 0].scatter(table["UMAP_VISUAL_1"], table["UMAP_VISUAL_2"], c=point_colors, s=point_size, alpha=alpha, linewidths=0)
    axes[1, 0].set_title("UMAP 2D ajustado sobre el PCA configurado")
    axes[1, 0].set_xlabel("UMAP 1")
    axes[1, 0].set_ylabel("UMAP 2")

    axes[1, 1].scatter(table["UMAP_GRAPH_PHATE_1"], table["UMAP_GRAPH_PHATE_2"], c=point_colors, s=point_size, alpha=alpha, linewidths=0)
    axes[1, 1].set_title("PHATE 2D desde graph_ interno de UMAP")
    axes[1, 1].set_xlabel("PHATE 1")
    axes[1, 1].set_ylabel("PHATE 2")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=color_map[cluster], label=f"C{cluster}")
        for cluster in clusters
    ]
    if handles:
        fig.legend(handles=handles, title="Leiden", loc="outside lower center", ncol=min(10, len(handles)), frameon=False)
    return fig


def save_figure_bundle(fig, base_path: str | Path, *, dpi: int = 600) -> list[Path]:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = base.with_suffix(f".{suffix}")
        kwargs = {"dpi": dpi} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    return paths
