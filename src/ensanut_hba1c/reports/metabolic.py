"""Metabolic-syndrome summaries and figures for micro- and macroclusters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COMPONENT_COLUMNS = [
    "MetS_Abdominal_Obesity",
    "MetS_Hypertriglyceridemia",
    "MetS_Low_HDL",
    "MetS_Elevated_BP",
    "MetS_Elevated_Glucose",
]
DEFAULT_COMPONENT_LABELS = {
    "MetS_Abdominal_Obesity": "Abdominal obesity",
    "MetS_Hypertriglyceridemia": "Hypertriglyceridemia",
    "MetS_Low_HDL": "Low HDL cholesterol",
    "MetS_Elevated_BP": "Elevated blood pressure",
    "MetS_Elevated_Glucose": "Elevated glucose",
}


def _cluster_sort(values):
    try:
        return sorted(values, key=lambda value: int(value))
    except (TypeError, ValueError):
        return sorted(values, key=lambda value: str(value))


def _save_figure(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


def _plot_average_criteria(
    summary: pd.DataFrame,
    *,
    cluster_col: str,
    cluster_label: str,
    prefix: str,
):
    plot_df = summary.copy()
    plot_df = plot_df.sort_values("average_number_of_criteria", ascending=False)
    x = np.arange(len(plot_df))
    values = plot_df["average_number_of_criteria"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, 0.72 * len(plot_df) + 4), 6.2))
    bars = ax.bar(x, values, edgecolor="white", linewidth=0.8)
    ax.axhline(3.0, linestyle="--", linewidth=1.8, color="#B23A2B")
    ax.text(
        len(plot_df) - 0.35,
        3.05,
        "Metabolic-syndrome threshold (3+)",
        ha="right",
        va="bottom",
        color="#9D2F24",
        fontweight="bold",
    )

    for position, (bar, value) in enumerate(zip(bars, values)):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.05,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    labels = [
        f"{prefix}{cluster}\n(n={int(n_complete):,})"
        for cluster, n_complete in zip(
            plot_df[cluster_col], plot_df["n_complete_five_criteria"]
        )
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontweight="bold")
    ax.set_ylim(0, 5.0)
    ax.set_ylabel("Average number of criteria (0–5)")
    ax.set_xlabel(cluster_label)
    ax.set_title(
        f"Average metabolic-syndrome criteria by {cluster_label.lower()}",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_stacked_components(
    component: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    cluster_col: str,
    cluster_label: str,
    prefix: str,
):
    mean_order = (
        summary.set_index(cluster_col)["average_number_of_criteria"]
        .sort_values(ascending=False)
        .index
    )
    contribution = component.reindex(mean_order).div(100).astype(float)
    x = np.arange(len(contribution))
    bottom = np.zeros(len(contribution), dtype=float)

    fig, ax = plt.subplots(figsize=(max(11, 0.78 * len(contribution) + 4), 6.8))
    for column in contribution.columns:
        values = contribution[column].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=column,
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += values

    summary_index = summary.set_index(cluster_col)
    for position, cluster in enumerate(contribution.index):
        ax.text(
            position,
            bottom[position] + 0.05,
            f"{bottom[position]:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(3.0, linestyle="--", linewidth=1.8, color="#B23A2B")
    ax.text(
        len(contribution) - 0.35,
        3.05,
        "Metabolic-syndrome threshold (3+)",
        ha="right",
        va="bottom",
        color="#9D2F24",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{prefix}{cluster}\n(n={int(summary_index.loc[cluster, 'n_complete_five_criteria']):,})"
            for cluster in contribution.index
        ],
        fontweight="bold",
    )
    ax.set_ylim(0, 5.0)
    ax.set_ylabel("Average number of criteria (0–5)")
    ax.set_xlabel(cluster_label)
    ax.set_title(
        f"Metabolic risk-factor accumulation by {cluster_label.lower()}",
        loc="left",
        fontweight="bold",
    )
    ax.legend(
        title="Risk factors",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def generate_metabolic_cluster_report(
    df: pd.DataFrame,
    cluster_col: str,
    output_dir: str | Path,
    file_prefix: str,
    cluster_label: str,
    component_columns: list[str] | None = None,
    component_labels: dict[str, str] | None = None,
    syndrome_col: str = "MetS_Binary",
    criteria_count_col: str = "MetS_Criteria_Count",
    complete_count_col: str = "MetS_Available_Criteria",
) -> dict[str, Any]:
    """Export metabolic criteria, syndrome prevalence, and cluster figures.

    All prevalence and mean-criteria figures use participants with all five
    criteria evaluable. This keeps cluster comparisons on the same denominator.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    components = component_columns or DEFAULT_COMPONENT_COLUMNS
    labels = {**DEFAULT_COMPONENT_LABELS, **(component_labels or {})}

    if cluster_col not in df.columns:
        raise KeyError(f"Cluster column does not exist: {cluster_col}")
    missing = [column for column in components if column not in df.columns]
    if missing:
        raise KeyError(f"Missing metabolic component columns: {missing}")

    data = df.dropna(subset=[cluster_col]).copy()
    for column in components + [syndrome_col, criteria_count_col, complete_count_col]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    clusters = _cluster_sort(data[cluster_col].dropna().unique())
    complete = data.loc[data[components].notna().all(axis=1)].copy()
    if complete.empty:
        raise ValueError("No participant has all five metabolic-syndrome components available.")

    complete["MetS_Criteria_Count_Complete"] = complete[components].sum(axis=1).astype(int)
    complete["MetS_Binary_Complete"] = complete["MetS_Criteria_Count_Complete"].ge(3).astype(int)

    summary_rows = []
    for cluster in clusters:
        subset = data.loc[data[cluster_col].eq(cluster)]
        complete_subset = complete.loc[complete[cluster_col].eq(cluster)]
        count_values = complete_subset["MetS_Criteria_Count_Complete"]
        syndrome_values = complete_subset["MetS_Binary_Complete"]
        summary_rows.append(
            {
                cluster_col: cluster,
                "n_total": len(subset),
                "n_complete_five_criteria": len(complete_subset),
                "complete_case_percentage": (
                    100 * len(complete_subset) / len(subset) if len(subset) else np.nan
                ),
                "metabolic_syndrome_n": int(syndrome_values.sum()),
                "metabolic_syndrome_percentage_complete": (
                    100 * syndrome_values.mean() if len(syndrome_values) else np.nan
                ),
                "average_number_of_criteria": (
                    count_values.mean() if len(count_values) else np.nan
                ),
                "median_number_of_criteria": (
                    count_values.median() if len(count_values) else np.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    component = (
        complete.groupby(cluster_col, observed=True)[components]
        .mean()
        .mul(100)
        .reindex(clusters)
    )
    component.columns = [labels.get(column, column) for column in component.columns]
    component = component.astype(float)

    counts_long = (
        complete.groupby([cluster_col, "MetS_Criteria_Count_Complete"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={"MetS_Criteria_Count_Complete": "criteria_count"})
    )
    totals = counts_long.groupby(cluster_col)["n"].transform("sum")
    counts_long["percentage_within_complete_cases"] = 100 * counts_long["n"] / totals

    summary.to_csv(output_dir / f"{file_prefix}_summary.csv", index=False)
    component.to_csv(output_dir / f"{file_prefix}_component_prevalence.csv")
    counts_long.to_csv(output_dir / f"{file_prefix}_criteria_count_distribution.csv", index=False)
    complete.to_csv(output_dir / f"{file_prefix}_complete_cases.csv", index=False)

    excel_path = output_dir / f"{file_prefix}_metabolic_report.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        component.to_excel(writer, sheet_name="Component_prevalence")
        counts_long.to_excel(writer, sheet_name="Criteria_distribution", index=False)
        complete.to_excel(writer, sheet_name="Complete_cases", index=False)

    # Heatmap: components x clusters.
    fig_heat, ax_heat = plt.subplots(figsize=(max(8, 0.72 * len(clusters) + 4), 5.8))
    image = ax_heat.imshow(component.T.values, aspect="auto", vmin=0, vmax=100, cmap="viridis")
    ax_heat.set_xticks(range(len(component.index)))
    prefix = "M" if "macro" in cluster_label.lower() else "C"
    ax_heat.set_xticklabels([f"{prefix}{value}" for value in component.index], rotation=45, ha="right")
    ax_heat.set_yticks(range(len(component.columns)))
    ax_heat.set_yticklabels(component.columns)
    ax_heat.set_xlabel(cluster_label)
    ax_heat.set_title("Prevalence of metabolic-syndrome components", loc="left", fontweight="bold")
    colorbar = fig_heat.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.03)
    colorbar.set_label("Prevalence among complete cases (%)")
    for row in range(component.shape[1]):
        for column in range(component.shape[0]):
            value = component.iloc[column, row]
            ax_heat.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 55 else "black",
            )
    fig_heat.tight_layout()
    _save_figure(fig_heat, output_dir / f"{file_prefix}_metabolic_component_heatmap")

    fig_average = _plot_average_criteria(
        summary,
        cluster_col=cluster_col,
        cluster_label=cluster_label,
        prefix=prefix,
    )
    _save_figure(fig_average, output_dir / f"{file_prefix}_average_criteria_by_cluster")

    fig_stacked = _plot_stacked_components(
        component,
        summary,
        cluster_col=cluster_col,
        cluster_label=cluster_label,
        prefix=prefix,
    )
    _save_figure(fig_stacked, output_dir / f"{file_prefix}_metabolic_risk_stacked_bars")

    return {
        "summary": summary,
        "component_prevalence": component,
        "criteria_distribution": counts_long,
        "complete_cases": complete,
        "excel_path": excel_path,
        "output_dir": output_dir,
        "heatmap_figure": fig_heat,
        "average_figure": fig_average,
        "stacked_bar_figure": fig_stacked,
        # Backward-compatible name used by the previous notebook.
        "bar_figure": fig_stacked,
    }


def plot_micro_macro_average_criteria(
    micro_summary: pd.DataFrame,
    macro_summary: pd.DataFrame,
    *,
    micro_col: str,
    macro_col: str,
    output_dir: str | Path,
):
    """Create and save one side-by-side mean-criteria figure."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    for ax, summary, cluster_col, prefix, title in [
        (axes[0], micro_summary, micro_col, "C", "A. Leiden microclusters"),
        (axes[1], macro_summary, macro_col, "M", "B. SHAP macroclusters"),
    ]:
        plot_df = summary.sort_values("average_number_of_criteria", ascending=False)
        x = np.arange(len(plot_df))
        values = plot_df["average_number_of_criteria"].to_numpy(dtype=float)
        bars = ax.bar(x, values, edgecolor="white", linewidth=0.8)
        ax.axhline(3.0, linestyle="--", linewidth=1.6, color="#B23A2B")
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.05,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{prefix}{value}" for value in plot_df[cluster_col]],
            fontweight="bold",
        )
        ax.set_ylim(0, 5)
        ax.set_ylabel("Average number of criteria (0–5)")
        ax.set_xlabel("Cluster")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Average metabolic-syndrome criteria across micro- and macroclusters",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir / "micro_macro_average_metabolic_criteria")
    return fig


def generate_original_and_macro_reports(
    df: pd.DataFrame,
    original_cluster_col: str,
    macro_cluster_col: str,
    output_root: str | Path,
    **kwargs,
) -> dict[str, dict[str, Any]]:
    """Run the identical metabolic analysis for microclusters and macroclusters."""

    output_root = Path(output_root)
    original = generate_metabolic_cluster_report(
        df=df,
        cluster_col=original_cluster_col,
        output_dir=output_root / "original_leiden_clusters",
        file_prefix="original_leiden_clusters",
        cluster_label="Leiden microcluster",
        **kwargs,
    )
    macro = generate_metabolic_cluster_report(
        df=df,
        cluster_col=macro_cluster_col,
        output_dir=output_root / "shap_macroclusters",
        file_prefix="shap_macroclusters",
        cluster_label="SHAP macrocluster",
        **kwargs,
    )
    combined = plot_micro_macro_average_criteria(
        original["summary"],
        macro["summary"],
        micro_col=original_cluster_col,
        macro_col=macro_cluster_col,
        output_dir=output_root,
    )
    return {"original": original, "macro": macro, "combined_average_figure": combined}


__all__ = [
    "DEFAULT_COMPONENT_COLUMNS",
    "generate_metabolic_cluster_report",
    "plot_micro_macro_average_criteria",
    "generate_original_and_macro_reports",
]
