"""Mellon density estimation in a fixed PCA50 representation.

Density is estimated in the first 20 PCA components supplied to the function and is
displayed on the hybrid UMAP-graph/PHATE projection built from PCA50. The plotted coordinates do not change Leiden.
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _effective_landmarks(requested: int | None, n_samples: int) -> int | None:
    if requested is None:
        return None
    requested = int(requested)
    if requested < 0:
        raise ValueError("n_landmarks must be None or an integer >= 0.")
    if requested == 0:
        return 0
    return min(requested, max(1, n_samples - 1))


def _density_summary(data: pd.DataFrame, *, cluster_col: str, density_col: str) -> pd.DataFrame:
    grouped = data.groupby(cluster_col, dropna=False)[density_col]
    summary = grouped.agg(
        n="size", mean="mean", standard_deviation="std", median="median",
        minimum="min", maximum="max",
    ).reset_index()
    q25 = grouped.quantile(0.25).rename("q25").reset_index()
    q75 = grouped.quantile(0.75).rename("q75").reset_index()
    return summary.merge(q25, on=cluster_col).merge(q75, on=cluster_col)


def plot_mellon_density_pca(
    data: pd.DataFrame, *, x_col: str, y_col: str, density_col: str,
    input_dimension: int, figure_size: tuple[float, float] = (10, 8),
    point_size: float = 12.0, alpha: float = 0.9, cmap: str = "magma",
    title: str | None = None,
):
    required = [x_col, y_col, density_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Missing Mellon plotting columns: {missing}")
    values = pd.to_numeric(data[density_col], errors="coerce").to_numpy(float)
    x = pd.to_numeric(data[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(data[y_col], errors="coerce").to_numpy(float)
    finite = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError("No finite rows are available for the Mellon plot.")
    order = np.argsort(values[finite])
    fig, ax = plt.subplots(figsize=figure_size)
    scatter = ax.scatter(
        x[finite][order], y[finite][order], c=values[finite][order],
        s=float(point_size), alpha=float(alpha), cmap=str(cmap),
        linewidths=0, rasterized=True,
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mellon normalized log density (lower tail clipped)")
    ax.set_title(title or f"Mellon density in PCA-{input_dimension}D projected on UMAP-graph → PHATE2D")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    return fig


def run_mellon_density_analysis(
    *, umap_df: pd.DataFrame, representation: np.ndarray, output_dir: str | Path,
    random_state: int, input_space_name: str = "PCA50", x_col: str = "UMAP_GRAPH_PHATE_1",
    y_col: str = "UMAP_GRAPH_PHATE_2", cluster_col: str = "Cluster_SHAP",
    id_columns: Sequence[str] = ("FOLIO_I", "FOLIO_INT"),
    n_landmarks: int | None = 2000, gp_type: str = "sparse_cholesky",
    d_method: str = "embedding", optimizer: str = "L-BFGS-B", n_iter: int = 100,
    init_learn_rate: float = 0.1, jitter: float = 1e-6, jit: bool = False,
    clip_lower_quantile: float = 0.05, figure_size: tuple[float, float] = (10, 8),
    point_size: float = 12.0, alpha: float = 0.9, cmap: str = "magma",
) -> dict[str, Any]:
    missing = [column for column in (x_col, y_col, cluster_col) if column not in umap_df.columns]
    if missing:
        raise KeyError(f"Missing plotting columns for Mellon density: {missing}")
    matrix = np.asarray(representation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(umap_df):
        raise ValueError("Mellon representation and participant dataframe are not aligned.")
    if matrix.shape[0] < 4 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("Mellon requires a finite matrix with at least four rows and two columns.")
    if not 0.0 <= float(clip_lower_quantile) < 1.0:
        raise ValueError("clip_lower_quantile must be in [0, 1).")
    try:
        import mellon
    except ImportError as error:
        raise ImportError(
            "Mellon is required. Install the project requirements or run "
            "`python -m pip install mellon==1.7.1`."
        ) from error

    n_samples, n_dimensions = matrix.shape
    effective_landmarks = _effective_landmarks(n_landmarks, n_samples)
    model = mellon.DensityEstimator(
        n_landmarks=effective_landmarks, gp_type=str(gp_type), d_method=str(d_method),
        optimizer=str(optimizer), n_iter=int(n_iter), init_learn_rate=float(init_learn_rate),
        jitter=float(jitter), jit=bool(jit), random_state=int(random_state),
    )
    log_density = np.asarray(model.fit_predict(matrix, build_predict=False), dtype=float).reshape(-1)
    if log_density.shape[0] != n_samples or not np.isfinite(log_density).all():
        raise RuntimeError("Mellon returned invalid log-density values.")

    normalized = log_density - np.log(float(n_samples))
    lower = float(np.quantile(normalized, float(clip_lower_quantile)))
    clipped = np.clip(normalized, lower, None)
    percentiles = pd.Series(normalized).rank(method="average", pct=True).to_numpy()

    out = umap_df.copy().reset_index(drop=True)
    out["Mellon_Log_Density"] = log_density
    out["Mellon_Log_Density_Normalized"] = normalized
    out["Mellon_Log_Density_Clipped"] = clipped
    out["Mellon_Density_Percentile"] = percentiles
    out["Mellon_Density_Input_Dimensions"] = int(n_dimensions)
    out["Mellon_Density_Input_Space"] = str(input_space_name)

    density_dir = Path(output_dir) / "mellon_density_high_dimensional"
    density_dir.mkdir(parents=True, exist_ok=True)
    export_columns = [column for column in id_columns if column in out.columns] + [
        cluster_col, x_col, y_col, "Mellon_Log_Density",
        "Mellon_Log_Density_Normalized", "Mellon_Log_Density_Clipped",
        "Mellon_Density_Percentile", "Mellon_Density_Input_Dimensions",
        "Mellon_Density_Input_Space",
    ]
    density_table = out[export_columns].copy()
    density_table.to_csv(density_dir / "mellon_density_by_participant.csv", index=False)
    summary = _density_summary(out, cluster_col=cluster_col, density_col="Mellon_Log_Density_Normalized")
    summary.to_csv(density_dir / "mellon_density_summary_by_microcluster.csv", index=False)

    configuration: Mapping[str, Any] = {
        "algorithm": "Mellon DensityEstimator",
        "density_input_space": str(input_space_name),
        "density_input_dimensions": int(n_dimensions),
        "n_samples": int(n_samples),
        "n_landmarks_requested": None if n_landmarks is None else int(n_landmarks),
        "n_landmarks_effective": None if effective_landmarks is None else int(effective_landmarks),
        "gp_type": str(gp_type), "d_method": str(d_method), "optimizer": str(optimizer),
        "n_iter": int(n_iter), "init_learn_rate": float(init_learn_rate),
        "jitter": float(jitter), "jit": bool(jit), "random_state": int(random_state),
        "clip_lower_quantile": float(clip_lower_quantile), "clip_lower_value": lower,
        "visual_coordinates": [x_col, y_col],
        "visual_coordinates_are_density_input": False,
        "projection_note": (
            "Density uses PCA50; the figure uses PHATE2D fitted to the UMAP affinity graph constructed from PCA50."
        ),
        "mellon_version": _safe_version("mellon"), "jax_version": _safe_version("jax"),
        "jaxlib_version": _safe_version("jaxlib"),
    }
    (density_dir / "mellon_density_configuration.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fig = plot_mellon_density_pca(
        out, x_col=x_col, y_col=y_col, density_col="Mellon_Log_Density_Clipped",
        input_dimension=n_dimensions, figure_size=figure_size, point_size=point_size,
        alpha=alpha, cmap=cmap,
    )
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 400} if suffix == "png" else {}
        fig.savefig(density_dir / f"pca20_umapgraph_phate2d_colored_by_mellon_density.{suffix}", bbox_inches="tight", **kwargs)
    return {
        "umap_df": out, "density_table": density_table, "cluster_summary": summary,
        "configuration": dict(configuration), "model": model, "figure": fig,
        "density_output_dir": density_dir,
    }


__all__ = ["plot_mellon_density_pca", "run_mellon_density_analysis"]
