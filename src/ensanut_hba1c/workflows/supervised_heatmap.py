"""Faithful modular implementation of the original supervised NB heatmap.

This module intentionally preserves the plotting pipeline that was previously
implemented in the large notebook cell.  The refactor changes code location,
not scientific filtering, variable order, cluster order, annotation placement,
or visual design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd

from .nb_evidence import NBEvidenceBundle, clean_text


SUPERVISED_HEATMAP_MODULE_VERSION = "2026-07-27-numeric-clusters-compact-colorbar-v2"


@dataclass(frozen=True)
class SupervisedHeatmapConfig:
    """Editable parameters matching the original notebook heatmap."""

    figure_height: float = 14.5
    dpi: int = 350
    color_percentile: float = 99.0
    annotation_font_size: float = 4.5
    annotation_wrap_width: int = 45
    annotation_line_separation: float = 0.7
    annotation_outer_pad: float = 0.02
    annotation_border_pad: float = 0.04
    left_text_width: float = 4.5
    left_connector_width: float = 0.85
    heatmap_width: float = 3.35
    right_connector_width: float = 0.85
    right_text_width: float = 4.5

    # Cluster labels and legend. Cluster data remain named C0, C1, ...;
    # show_cluster_prefix only changes the visible labels.
    cluster_label_font_size: float = 6.8
    cluster_label_pad: float = 2.0
    show_cluster_prefix: bool = False
    legend_show_cluster_prefix: bool = False
    legend_font_size: float = 8.0

    # Compact Bayes-score colorbar.
    colorbar_height: float = 0.18
    colorbar_width_fraction: float = 0.58
    colorbar_label_font_size: float = 9.0
    colorbar_tick_font_size: float = 7.5

    cmap: str = "YlGnBu"
    show_filter_summary: bool = False
    show_in_notebook: bool = True
    notebook_preview_mode: str = "saved_png"
    notebook_preview_width: int = 1500
    close_after_display: bool = True
    verbose: bool = False

    # Kept for backward compatibility with the first modular draft.  They no
    # longer alter the original heatmap design.
    figure_width: float | None = None
    point_size_min: float = 5.0
    point_size_max: float = 60.0
    show_all_coverage_points: bool = False
    show_variable_count_in_title: bool = True


@dataclass
class SupervisedHeatmapResult:
    png_path: Path
    pdf_path: Path
    workbook_path: Path
    manifest_path: Path
    annotation_table: pd.DataFrame
    filter_summary: pd.DataFrame
    filtered_variables: list[str]
    cluster_order: list[str]
    n_variables: int
    n_clusters: int
    figure: object | None = None


def _translate_category_minimal(category: object) -> str:
    """Apply the same minimal category translation used in the original cell."""

    category = clean_text(category)
    if not category:
        return "Not specified"

    exact_translations = {
        "1: Sí": "Yes",
        "1: SI": "Yes",
        "1: Si": "Yes",
        "1: Sí.": "Yes",
        "1: SI.": "Yes",
        "1: Si.": "Yes",
        "2: No": "No",
        "2: NO": "No",
        "2: No.": "No",
        "2: NO.": "No",
        "2: Lactando": "Breastfeeding",
        "4: Le es indiferente": "Indifferent",
        "10: No responde": "No response",
        "9: No sabe": "Does not know",
        "7: No sabe": "Does not know",
        "2: cesárea por urgencias?": "Emergency cesarean section",
        "4: No utiliza el teléfono": "Does not use a telephone",
    }
    if category in exact_translations:
        return exact_translations[category]

    interval_match = re.search(r"\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]", category)
    if interval_match:
        return f"{interval_match.group(1).strip()}–{interval_match.group(2).strip()}"

    replacements = {
        "No responde": "No response",
        "No sabe": "Does not know",
        "Le es indiferente": "Indifferent",
        "Lactando": "Breastfeeding",
        "Sí": "Yes",
        "Si": "Yes",
        "SI": "Yes",
        "NO": "No",
    }
    translated = category
    for original_text, replacement_text in replacements.items():
        translated = translated.replace(original_text, replacement_text)
    return translated


def _normalize_cluster(value: object) -> str:
    text = clean_text(value)
    numeric = pd.to_numeric(re.sub(r"^[Cc]", "", text), errors="coerce")
    if pd.isna(numeric):
        return text
    return f"C{int(numeric)}"


def _cluster_numeric_key(value: object) -> tuple[int, str]:
    """Return a stable numeric key for labels such as C0, C1, ..., C10."""

    normalized = _normalize_cluster(value)
    numeric = pd.to_numeric(
        re.sub(r"^[Cc]", "", normalized),
        errors="coerce",
    )
    if pd.isna(numeric):
        return (10**9, normalized)
    return (int(numeric), normalized)


def _cluster_display_label(cluster: str, *, show_prefix: bool) -> str:
    """Format a cluster label without changing its internal identifier."""

    normalized = _normalize_cluster(cluster)
    if show_prefix:
        return normalized
    return re.sub(r"^[Cc]", "", normalized)


def _validate_manual_input(
    supervised_input_df: pd.DataFrame,
    evidence: NBEvidenceBundle,
) -> pd.DataFrame:
    required_columns = [
        "cluster",
        "var",
        "english_description",
        "category_en_override",
    ]
    missing_columns = [
        column for column in required_columns if column not in supervised_input_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "SUPERVISED_INPUT_DF is missing required columns: "
            + ", ".join(missing_columns)
        )

    manual_input_df = supervised_input_df[required_columns].copy()
    for column in required_columns:
        manual_input_df[column] = manual_input_df[column].map(clean_text)
    manual_input_df["cluster"] = manual_input_df["cluster"].map(_normalize_cluster)
    manual_input_df = manual_input_df.loc[
        manual_input_df["cluster"].ne("") | manual_input_df["var"].ne("")
    ].reset_index(drop=True)

    if manual_input_df.empty:
        raise ValueError("SUPERVISED_INPUT_DF is empty.")

    invalid_rows = manual_input_df.loc[
        manual_input_df["cluster"].eq("")
        | manual_input_df["var"].eq("")
        | manual_input_df["english_description"].eq("")
    ]
    if not invalid_rows.empty:
        raise ValueError(
            "Rows without cluster, var, or english_description were found:\n"
            + invalid_rows.to_string(index=False)
        )

    duplicated = manual_input_df.duplicated(
        subset=["cluster", "var"], keep=False
    )
    if duplicated.any():
        duplicate_rows = manual_input_df.loc[
            duplicated, ["cluster", "var"]
        ].drop_duplicates()
        raise ValueError(
            "Duplicated cluster-variable annotations were found:\n"
            + duplicate_rows.to_string(index=False)
        )

    unknown_clusters = [
        cluster
        for cluster in manual_input_df["cluster"].unique()
        if cluster not in evidence.cluster_order
    ]
    unknown_variables = [
        variable
        for variable in manual_input_df["var"].unique()
        if variable not in evidence.variable_order
    ]
    if unknown_clusters:
        raise ValueError(
            "Manual annotations contain clusters absent from the filtered matrix: "
            + ", ".join(unknown_clusters)
        )
    if unknown_variables:
        raise ValueError(
            "Manual annotations contain variables absent from the filtered matrix: "
            + ", ".join(unknown_variables)
        )
    return manual_input_df


def _build_filter_summary(evidence: NBEvidenceBundle) -> pd.DataFrame:
    summary = (
        evidence.filtered_evidence_df.groupby("cluster_plot", sort=False)
        .agg(
            surviving_rows=("score", "size"),
            eligible_variables=("var", "nunique"),
            cluster_size=("n_cluster", "first"),
            maximum_score=("score", "max"),
            maximum_coverage=("coverage_in_cluster", "max"),
        )
        .reindex(evidence.cluster_order)
    )
    summary["surviving_rows"] = summary["surviving_rows"].fillna(0).astype(int)
    summary["eligible_variables"] = summary["eligible_variables"].fillna(0).astype(int)
    return summary


def _build_annotation_table(
    manual_input_df: pd.DataFrame,
    evidence: NBEvidenceBundle,
    wrap_width: int,
) -> pd.DataFrame:
    row_position = {
        variable: position for position, variable in enumerate(evidence.variable_order)
    }
    selection_errors: list[str] = []
    records: list[dict[str, object]] = []

    for input_position, manual_row in manual_input_df.iterrows():
        cluster = manual_row["cluster"]
        variable = manual_row["var"]
        matches = evidence.filtered_evidence_df.loc[
            evidence.filtered_evidence_df["cluster_plot"].eq(cluster)
            & evidence.filtered_evidence_df["var"].eq(variable)
        ].copy()

        if matches.empty:
            unfiltered = evidence.evidence_df.loc[
                evidence.evidence_df["cluster_plot"].eq(cluster)
                & evidence.evidence_df["var"].eq(variable)
            ].copy()
            if unfiltered.empty:
                reason = "the variable is absent from this cluster"
            else:
                best_unfiltered = unfiltered.nlargest(
                    1, ["score", "coverage_in_cluster", "n_cluster_x"]
                ).iloc[0]
                reason = (
                    "the row did not pass the filter: "
                    f"score={best_unfiltered['score']:.3f}, "
                    f"coverage={best_unfiltered['coverage_in_cluster']:.1%}, "
                    f"n_cluster_x={best_unfiltered['n_cluster_x']:.0f}"
                )
            selection_errors.append(f"{cluster} / {variable}: {reason}.")
            continue

        best_row = matches.nlargest(
            1, ["score", "coverage_in_cluster", "n_cluster_x"]
        ).iloc[0]
        raw_category = clean_text(best_row.get(evidence.category_source_column, ""))
        category_to_display = (
            manual_row["category_en_override"]
            if manual_row["category_en_override"]
            else _translate_category_minimal(raw_category)
        )
        score_value = float(best_row["score"])
        coverage_value = float(np.clip(best_row["coverage_in_cluster"], 0.0, 1.0))
        continuous_label_text = (
            f"{cluster} | {variable}: "
            f"{manual_row['english_description']}; "
            f"{category_to_display}; "
            f"score {score_value:.2f}; "
            f"coverage {coverage_value:.1%}"
        )
        label_lines = textwrap.wrap(
            continuous_label_text,
            width=int(wrap_width),
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=True,
            drop_whitespace=True,
        )
        records.append(
            {
                "input_position": input_position,
                "annotation_id": f"{cluster}::{variable}",
                "cluster": cluster,
                "var": variable,
                "english_description": manual_row["english_description"],
                "category_en_override": category_to_display,
                "score": score_value,
                "coverage": coverage_value,
                "row_position": row_position[variable],
                "label_lines": label_lines,
                "label_text": "\n".join(label_lines),
            }
        )

    if selection_errors:
        raise ValueError(
            "INVALID MANUAL SELECTION\n"
            + "-" * 78
            + "\n"
            + "\n".join(f"- {error}" for error in selection_errors)
        )

    annotation_df = pd.DataFrame(records).reset_index(drop=True)
    if annotation_df.empty:
        raise ValueError("No supervised annotations were generated.")
    return annotation_df


def _calculate_label_positions(detail_df: pd.DataFrame, total_rows: int) -> dict[str, float]:
    if detail_df.empty:
        return {}
    annotation_ids = detail_df["annotation_id"].tolist()
    number_of_labels = len(annotation_ids)
    upper_limit = max(4.0, total_rows * 0.025)
    lower_limit = min(total_rows - 5.0, total_rows * 0.975)
    if number_of_labels == 1:
        positions = np.array([total_rows / 2.0])
    else:
        positions = np.linspace(upper_limit, lower_limit, number_of_labels)
    return dict(zip(annotation_ids, positions))


def _draw_bezier_connector(
    axis: plt.Axes,
    label_y: float,
    variable_y: float,
    color: object,
    side: str,
) -> None:
    if side == "left":
        vertices = [
            (0.0, label_y),
            (0.36, label_y),
            (0.64, variable_y),
            (1.0, variable_y),
        ]
    else:
        vertices = [
            (0.0, variable_y),
            (0.36, variable_y),
            (0.64, label_y),
            (1.0, label_y),
        ]
    connector_path = MplPath(
        vertices,
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    axis.add_patch(
        PathPatch(
            connector_path,
            transform=axis.transData,
            facecolor="none",
            edgecolor=color,
            linewidth=0.80,
            alpha=0.28,
            capstyle="round",
            joinstyle="round",
            clip_on=False,
            zorder=3,
        )
    )


def _draw_annotation_side(
    detail_df: pd.DataFrame,
    positions: dict[str, float],
    text_axis: plt.Axes,
    connector_axis: plt.Axes,
    heatmap_axis: plt.Axes,
    cluster_colors: dict[str, object],
    n_clusters: int,
    side: str,
    font_size: float,
    line_separation: float,
    outer_pad: float,
    border_pad: float,
) -> None:
    is_left = side == "left"
    if is_left:
        text_x = 0.985
        line_alignment = "right"
        box_alignment = (1.0, 0.5)
    else:
        text_x = 0.015
        line_alignment = "left"
        box_alignment = (0.0, 0.5)

    for _, row in detail_df.iterrows():
        annotation_id = row["annotation_id"]
        cluster = row["cluster"]
        variable_y = float(row["row_position"])
        label_y = float(positions[annotation_id])
        cluster_color = cluster_colors[cluster]

        # Every line is an independent TextArea.  This is the critical part of
        # the original true left/right line alignment.
        line_objects = [
            TextArea(
                line,
                textprops={
                    "fontsize": font_size,
                    "color": "0.12",
                    "ha": line_alignment,
                    "va": "center",
                },
            )
            for line in row["label_lines"]
        ]
        packed_text = VPacker(
            children=line_objects,
            align=line_alignment,
            pad=0,
            sep=line_separation,
        )
        annotation_box = AnnotationBbox(
            packed_text,
            (text_x, label_y),
            xycoords=text_axis.get_yaxis_transform(),
            box_alignment=box_alignment,
            frameon=True,
            pad=outer_pad,
            bboxprops={
                "boxstyle": f"round,pad={border_pad:g}",
                "facecolor": to_rgba(cluster_color, 0.08),
                "edgecolor": to_rgba(cluster_color, 0.68),
                "linewidth": 0.80,
            },
            zorder=5,
        )
        text_axis.add_artist(annotation_box)
        _draw_bezier_connector(
            axis=connector_axis,
            label_y=label_y,
            variable_y=variable_y,
            color=cluster_color,
            side=side,
        )

        heatmap_x = [-0.58, -0.50] if is_left else [n_clusters - 0.50, n_clusters - 0.42]
        heatmap_axis.plot(
            heatmap_x,
            [variable_y, variable_y],
            color=cluster_color,
            linewidth=1.05,
            alpha=0.78,
            clip_on=False,
            zorder=7,
        )


def create_supervised_nb_heatmap(
    evidence: NBEvidenceBundle,
    supervised_input_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: SupervisedHeatmapConfig | None = None,
) -> SupervisedHeatmapResult:
    """Create the original global NB heatmap without altering its pipeline."""

    config = config or SupervisedHeatmapConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manual_input_df = _validate_manual_input(supervised_input_df, evidence)
    annotation_df = _build_annotation_table(
        manual_input_df,
        evidence,
        wrap_width=config.annotation_wrap_width,
    )
    original_cluster_order = list(evidence.cluster_order)
    cluster_order = sorted(original_cluster_order, key=_cluster_numeric_key)
    variable_order = list(evidence.variable_order)
    filter_summary = _build_filter_summary(evidence).reindex(cluster_order)
    n_clusters = len(cluster_order)
    total_variables = len(variable_order)

    score_plot = evidence.score_matrix.reindex(
        index=variable_order, columns=cluster_order
    ).apply(pd.to_numeric, errors="coerce")
    coverage_plot = evidence.coverage_matrix.reindex(
        index=variable_order, columns=cluster_order
    ).apply(pd.to_numeric, errors="coerce")
    count_plot = evidence.count_matrix.reindex(
        index=variable_order, columns=cluster_order
    ).apply(pd.to_numeric, errors="coerce")
    category_plot = evidence.category_matrix.reindex(
        index=variable_order, columns=cluster_order
    ).fillna("")

    selected_clusters = set(annotation_df["cluster"])
    left_clusters: list[str] = []
    right_clusters: list[str] = []
    side_counter = 0
    for cluster in cluster_order:
        if cluster not in selected_clusters:
            continue
        if side_counter % 2 == 0:
            left_clusters.append(cluster)
        else:
            right_clusters.append(cluster)
        side_counter += 1

    left_draw_df = (
        annotation_df.loc[annotation_df["cluster"].isin(left_clusters)]
        .sort_values("row_position", kind="stable")
        .reset_index(drop=True)
    )
    right_draw_df = (
        annotation_df.loc[annotation_df["cluster"].isin(right_clusters)]
        .sort_values("row_position", kind="stable")
        .reset_index(drop=True)
    )
    left_label_positions = _calculate_label_positions(left_draw_df, total_variables)
    right_label_positions = _calculate_label_positions(right_draw_df, total_variables)

    all_values = score_plot.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(all_values)
    positive_values = all_values[np.isfinite(all_values) & (all_values > 0)]
    if positive_values.size == 0:
        raise ValueError("The score matrix contains no positive values.")
    vmax_value = float(np.nanpercentile(positive_values, config.color_percentile))
    if not np.isfinite(vmax_value) or vmax_value <= 0:
        vmax_value = float(positive_values.max())
    norm = Normalize(vmin=0.0, vmax=vmax_value, clip=True)

    heatmap_cmap = plt.get_cmap(config.cmap).copy()
    heatmap_cmap.set_bad("white")
    cluster_cmap = plt.get_cmap("tab10")
    original_cluster_colors = {
        cluster: cluster_cmap(position % 10)
        for position, cluster in enumerate(original_cluster_order)
    }
    cluster_colors = {
        cluster: original_cluster_colors[cluster]
        for cluster in cluster_order
    }
    cluster_axis_labels = [
        _cluster_display_label(
            cluster,
            show_prefix=config.show_cluster_prefix,
        )
        for cluster in cluster_order
    ]
    cluster_legend_labels = [
        _cluster_display_label(
            cluster,
            show_prefix=config.legend_show_cluster_prefix,
        )
        for cluster in cluster_order
    ]

    figure_width = (
        config.left_text_width
        + config.left_connector_width
        + config.heatmap_width
        + config.right_connector_width
        + config.right_text_width
    )
    fig = plt.figure(figsize=(figure_width, config.figure_height))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=5,
        width_ratios=[
            config.left_text_width,
            config.left_connector_width,
            config.heatmap_width,
            config.right_connector_width,
            config.right_text_width,
        ],
        height_ratios=[12.8, config.colorbar_height],
        wspace=0.008,
        hspace=0.08,
    )
    left_text_axis = fig.add_subplot(grid[0, 0])
    left_connector_axis = fig.add_subplot(grid[0, 1])
    heatmap_axis = fig.add_subplot(grid[0, 2])
    right_connector_axis = fig.add_subplot(grid[0, 3])
    right_text_axis = fig.add_subplot(grid[0, 4])
    colorbar_axis = fig.add_subplot(grid[1, 2])

    image = heatmap_axis.imshow(
        masked_values,
        aspect="auto",
        interpolation="nearest",
        cmap=heatmap_cmap,
        norm=norm,
        origin="upper",
    )
    heatmap_axis.set_xlim(-0.5, n_clusters - 0.5)
    heatmap_axis.set_ylim(total_variables - 0.5, -0.5)
    heatmap_axis.set_xticks(np.arange(n_clusters))
    heatmap_axis.set_xticklabels(
        cluster_axis_labels,
        fontsize=config.cluster_label_font_size,
        fontweight="bold",
    )
    heatmap_axis.xaxis.tick_top()
    heatmap_axis.tick_params(
        axis="x",
        length=0,
        pad=config.cluster_label_pad,
    )
    heatmap_axis.set_yticks([])
    heatmap_axis.set_xticks(np.arange(-0.5, n_clusters, 1), minor=True)
    heatmap_axis.grid(
        which="minor", axis="x", linewidth=0.42, color="white", alpha=0.75
    )
    heatmap_axis.tick_params(which="minor", bottom=False, top=False)
    for spine in heatmap_axis.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("0.45")

    for axis in [
        left_text_axis,
        left_connector_axis,
        right_connector_axis,
        right_text_axis,
    ]:
        axis.set_ylim(total_variables - 0.5, -0.5)
        axis.set_xlim(0, 1)
        axis.axis("off")

    _draw_annotation_side(
        detail_df=left_draw_df,
        positions=left_label_positions,
        text_axis=left_text_axis,
        connector_axis=left_connector_axis,
        heatmap_axis=heatmap_axis,
        cluster_colors=cluster_colors,
        n_clusters=n_clusters,
        side="left",
        font_size=config.annotation_font_size,
        line_separation=config.annotation_line_separation,
        outer_pad=config.annotation_outer_pad,
        border_pad=config.annotation_border_pad,
    )
    _draw_annotation_side(
        detail_df=right_draw_df,
        positions=right_label_positions,
        text_axis=right_text_axis,
        connector_axis=right_connector_axis,
        heatmap_axis=heatmap_axis,
        cluster_colors=cluster_colors,
        n_clusters=n_clusters,
        side="right",
        font_size=config.annotation_font_size,
        line_separation=config.annotation_line_separation,
        outer_pad=config.annotation_outer_pad,
        border_pad=config.annotation_border_pad,
    )

    for _, row in annotation_df.iterrows():
        cluster = row["cluster"]
        column_position = cluster_order.index(cluster)
        variable_y = float(row["row_position"])
        coverage_value = float(row["coverage"])
        heatmap_axis.scatter(
            column_position,
            variable_y,
            s=22 + 100 * coverage_value,
            facecolors=cluster_colors[cluster],
            edgecolors="white",
            linewidths=1.0,
            alpha=0.95,
            zorder=10,
        )

    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(
        "Bayes score",
        fontsize=config.colorbar_label_font_size,
        labelpad=4,
    )
    colorbar.ax.tick_params(
        axis="x",
        labelsize=config.colorbar_tick_font_size,
        length=2.6,
    )
    colorbar.outline.set_linewidth(0.65)

    colorbar_width_fraction = float(
        np.clip(config.colorbar_width_fraction, 0.05, 1.0)
    )
    colorbar_position = colorbar_axis.get_position()
    reduced_width = colorbar_position.width * colorbar_width_fraction
    centered_x0 = (
        colorbar_position.x0
        + (colorbar_position.width - reduced_width) / 2.0
    )
    colorbar_axis.set_position(
        [
            centered_x0,
            colorbar_position.y0,
            reduced_width,
            colorbar_position.height,
        ]
    )

    cluster_handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="None",
            markersize=7,
            markerfacecolor=cluster_colors[cluster],
            markeredgecolor="none",
            label=cluster_legend_labels[position],
        )
        for position, cluster in enumerate(cluster_order)
    ]
    fig.legend(
        handles=cluster_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.938),
        ncol=min(n_clusters, 10),
        frameon=False,
        fontsize=config.legend_font_size,
        columnspacing=1.10,
        handletextpad=0.30,
    )
    fig.subplots_adjust(left=0.012, right=0.988, top=0.895, bottom=0.060)
    fig.suptitle(
        f"Bayes evidence map: {total_variables:,} variables × {n_clusters} clusters",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    png_path = output_dir / "heatmap_manual_dataframe_true_line_alignment.png"
    pdf_path = output_dir / "heatmap_manual_dataframe_true_line_alignment.pdf"
    workbook_path = output_dir / "heatmap_manual_dataframe_true_line_alignment.xlsx"
    manifest_path = output_dir / "heatmap_manual_dataframe_true_line_alignment_manifest.json"

    fig.savefig(
        png_path,
        dpi=int(config.dpi),
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    final_annotation_df = annotation_df[
        [
            "cluster",
            "var",
            "english_description",
            "category_en_override",
            "score",
            "coverage",
        ]
    ].copy().reset_index(drop=True)

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        manual_input_df.to_excel(writer, sheet_name="manual_input", index=False)
        final_annotation_df.to_excel(
            writer, sheet_name="selected_annotations", index=False
        )
        score_plot.to_excel(writer, sheet_name="scores")
        coverage_plot.to_excel(writer, sheet_name="coverage")
        count_plot.to_excel(writer, sheet_name="counts")
        category_plot.to_excel(writer, sheet_name="categories")
        evidence.filtered_evidence_df.to_excel(
            writer, sheet_name="eligible_evidence", index=False
        )
        filter_summary.to_excel(writer, sheet_name="filter_summary")

    manifest = {
        "analysis": "Supervised Naive Bayes evidence heatmap",
        "pipeline": "faithful modular port of the original notebook heatmap cell",
        "source": str(evidence.source_path),
        "thresholds": asdict(evidence.config),
        "n_variables": total_variables,
        "n_clusters": n_clusters,
        "cluster_order": cluster_order,
        "n_manual_annotations": int(len(final_annotation_df)),
        "left_clusters": left_clusters,
        "right_clusters": right_clusters,
        "configuration": asdict(config),
        "outputs": [png_path.name, pdf_path.name, workbook_path.name],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if config.show_filter_summary:
        try:
            from IPython.display import display

            display(filter_summary.style.format({
                "cluster_size": lambda value: "—" if pd.isna(value) else f"{value:.0f}",
                "maximum_score": lambda value: "—" if pd.isna(value) else f"{value:.3f}",
                "maximum_coverage": lambda value: "—" if pd.isna(value) else f"{value:.1%}",
            }))
        except Exception:
            pass

    if config.show_in_notebook:
        try:
            from IPython.display import Image, IFrame, display

            preview_mode = str(config.notebook_preview_mode).strip().lower()
            if preview_mode == "saved_png":
                display(
                    Image(
                        filename=str(png_path),
                        width=int(config.notebook_preview_width),
                    )
                )
            elif preview_mode == "saved_pdf":
                display(
                    IFrame(
                        src=pdf_path.as_uri(),
                        width="100%",
                        height=900,
                    )
                )
            elif preview_mode == "matplotlib":
                display(fig)
            else:
                raise ValueError(
                    "notebook_preview_mode must be one of: "
                    "'saved_png', 'saved_pdf', or 'matplotlib'."
                )
        except Exception:
            plt.show()

    if config.verbose:
        print(
            f"Supervised heatmap completed: {total_variables:,} variables, "
            f"{n_clusters} clusters, {len(final_annotation_df)} annotations."
        )
        print(f"PNG: {png_path}")
        print(f"PDF: {pdf_path}")
        print(f"Excel: {workbook_path}")

    returned_figure: object | None = fig
    if config.close_after_display:
        plt.close(fig)
        returned_figure = None

    return SupervisedHeatmapResult(
        png_path=png_path,
        pdf_path=pdf_path,
        workbook_path=workbook_path,
        manifest_path=manifest_path,
        annotation_table=final_annotation_df,
        filter_summary=filter_summary,
        filtered_variables=variable_order,
        cluster_order=cluster_order,
        n_variables=total_variables,
        n_clusters=n_clusters,
        figure=returned_figure,
    )
