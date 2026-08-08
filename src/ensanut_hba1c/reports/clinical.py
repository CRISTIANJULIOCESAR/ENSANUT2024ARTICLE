"""Publication-ready clinical tables and robust-Z heatmaps by cluster."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import chi2_contingency, kruskal


def median_iqr_text(series: pd.Series, decimals: int = 2) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "NA"
    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return f"{median:.{decimals}f} [{q1:.{decimals}f}–{q3:.{decimals}f}]"


def build_cluster_heatmap_matrix(
    df: pd.DataFrame,
    variables: Sequence[str],
    cluster_col: str = "Cluster_SHAP",
):
    available = [variable for variable in variables if variable in df.columns]
    if not available:
        raise ValueError("None of the requested clinical variables is available.")
    numeric = df[available].apply(pd.to_numeric, errors="coerce")
    medians = numeric.groupby(df[cluster_col], observed=True).median().T
    row_center = medians.median(axis=1)
    row_iqr = medians.quantile(0.75, axis=1) - medians.quantile(0.25, axis=1)
    row_scale = (row_iqr / 1.349).replace(0, np.nan)
    z = (
        medians.sub(row_center, axis=0)
        .div(row_scale, axis=0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    if z.shape[0] > 1:
        variable_order = z.index[
            leaves_list(
                linkage(
                    z.values,
                    method="ward",
                    metric="euclidean",
                    optimal_ordering=True,
                )
            )
        ].tolist()
    else:
        variable_order = z.index.tolist()
    if z.shape[1] > 1:
        cluster_order = z.columns[
            leaves_list(
                linkage(
                    z.T.values,
                    method="ward",
                    metric="euclidean",
                    optimal_ordering=True,
                )
            )
        ].tolist()
    else:
        cluster_order = z.columns.tolist()
    return z.loc[variable_order, cluster_order], variable_order, cluster_order


def build_cluster_table(
    df: pd.DataFrame,
    variables: Sequence[str],
    *,
    cluster_col: str = "Cluster_SHAP",
    label_map: Mapping[str, str] | None = None,
    unit_map: Mapping[str, str] | None = None,
    sex_col: str | None = None,
    female_value=2,
    cluster_order: Sequence | None = None,
) -> pd.DataFrame:
    label_map = dict(label_map or {})
    unit_map = dict(unit_map or {})
    clusters = (
        sorted(df[cluster_col].dropna().unique(), key=lambda value: str(value))
        if cluster_order is None
        else list(cluster_order)
    )
    rows: list[dict] = []
    for variable in variables:
        if variable not in df.columns:
            continue
        groups = [
            pd.to_numeric(
                df.loc[df[cluster_col] == cluster, variable], errors="coerce"
            ).dropna()
            for cluster in clusters
        ]
        valid_groups = [group for group in groups if len(group) > 0]
        try:
            p_value = kruskal(*valid_groups).pvalue if len(valid_groups) >= 2 else np.nan
        except ValueError:
            p_value = np.nan
        row = {
            "Variable": label_map.get(variable, variable),
            "Unit": unit_map.get(variable, ""),
            "Overall": median_iqr_text(df[variable]),
            "p_value": p_value,
        }
        for cluster in clusters:
            row[f"Cluster {cluster}"] = median_iqr_text(
                df.loc[df[cluster_col] == cluster, variable]
            )
        rows.append(row)

    if sex_col and sex_col in df.columns:
        contingency = pd.crosstab(df[cluster_col], df[sex_col])
        try:
            p_value = (
                chi2_contingency(contingency).pvalue
                if contingency.shape[0] > 1 and contingency.shape[1] > 1
                else np.nan
            )
        except ValueError:
            p_value = np.nan

        def female_text(frame: pd.DataFrame) -> str:
            sex = pd.to_numeric(frame[sex_col], errors="coerce")
            valid = sex.notna()
            denominator = int(valid.sum())
            numerator = int((sex.loc[valid] == female_value).sum())
            return (
                "NA"
                if denominator == 0
                else f"{numerator} ({100 * numerator / denominator:.1f}%)"
            )

        row = {
            "Variable": "Female, n (%)",
            "Unit": "",
            "Overall": female_text(df),
            "p_value": p_value,
        }
        for cluster in clusters:
            row[f"Cluster {cluster}"] = female_text(
                df.loc[df[cluster_col] == cluster]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_cluster_heatmap(
    z: pd.DataFrame,
    label_map: Mapping[str, str] | None = None,
    vmin: float = -2.5,
    vmax: float = 2.5,
):
    label_map = dict(label_map or {})
    fig, ax = plt.subplots(figsize=(7, max(5, 0.34 * len(z))))
    image = ax.imshow(z.values, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(z.shape[1]))
    ax.set_xticklabels([f"Cluster {cluster}" for cluster in z.columns], rotation=45, ha="right")
    ax.set_yticks(range(z.shape[0]))
    ax.set_yticklabels([label_map.get(variable, variable) for variable in z.index])
    ax.tick_params(axis="both", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(
        image,
        ax=ax,
        label="Robust Z-score of cluster median",
        fraction=0.035,
        pad=0.03,
    )
    fig.tight_layout()
    return fig


def generate_clinical_cluster_report(
    df: pd.DataFrame,
    *,
    variables: Sequence[str],
    cluster_col: str,
    label_map: Mapping[str, str] | None = None,
    unit_map: Mapping[str, str] | None = None,
    sex_col: str | None = None,
    female_value=2,
    output_dir: str | Path = ".",
    heatmap_vmin: float = -2.5,
    heatmap_vmax: float = 2.5,
):
    """Build, save and return the publication table and heatmap."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    z, variable_order, cluster_order = build_cluster_heatmap_matrix(
        df, variables, cluster_col
    )
    table = build_cluster_table(
        df,
        variable_order,
        cluster_col=cluster_col,
        label_map=label_map,
        unit_map=unit_map,
        sex_col=sex_col,
        female_value=female_value,
        cluster_order=cluster_order,
    )
    table.to_csv(output_dir / "publication_cluster_table.csv", index=False)
    table.to_excel(output_dir / "publication_cluster_table.xlsx", index=False)
    z.to_csv(output_dir / "cluster_robust_z_matrix.csv")
    fig = plot_cluster_heatmap(z, label_map, heatmap_vmin, heatmap_vmax)
    fig.savefig(output_dir / "heatmap_clusters.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_dir / "heatmap_clusters.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "heatmap_clusters.svg", bbox_inches="tight")
    return {
        "heatmap_values": z,
        "variable_order": variable_order,
        "cluster_order": cluster_order,
        "table": table,
        "figure": fig,
    }


__all__ = [
    "median_iqr_text",
    "build_cluster_heatmap_matrix",
    "build_cluster_table",
    "plot_cluster_heatmap",
    "generate_clinical_cluster_report",
]
