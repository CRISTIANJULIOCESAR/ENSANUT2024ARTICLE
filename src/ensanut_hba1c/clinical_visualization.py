"""Hybrid PCA-to-UMAP-graph-to-PHATE visualization and clinical utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def derive_glycemic_status(
    df: pd.DataFrame,
    *,
    hba1c_col: str = "HB1AC",
    normal_upper: float = 5.7,
    diabetes_lower: float = 6.5,
    output_col: str = "HbA1c_Status",
) -> pd.DataFrame:
    """Add standard descriptive HbA1c categories."""

    out = df.copy()
    hba1c = _numeric(out, hba1c_col)
    status = pd.Series(pd.NA, index=out.index, dtype="string")
    status.loc[hba1c.lt(normal_upper)] = "Normal"
    status.loc[hba1c.ge(normal_upper) & hba1c.lt(diabetes_lower)] = "Prediabetes"
    status.loc[hba1c.ge(diabetes_lower)] = "Diabetes"
    out[output_col] = pd.Categorical(
        status,
        categories=["Normal", "Prediabetes", "Diabetes"],
        ordered=True,
    )
    return out


def _nullable_indicator(
    positive: pd.Series,
    known: pd.Series,
    index: pd.Index,
) -> pd.Series:
    result = pd.Series(pd.NA, index=index, dtype="Int64")
    result.loc[known & ~positive] = 0
    result.loc[known & positive] = 1
    return result


def derive_metabolic_syndrome(
    df: pd.DataFrame,
    *,
    sex_col: str,
    male_value,
    female_value,
    waist_col: str,
    triglycerides_col: str,
    hdl_col: str,
    systolic_bp_col: str,
    diastolic_bp_col: str,
    glucose_col: str,
    waist_male_cutoff: float = 90.0,
    waist_female_cutoff: float = 80.0,
    triglycerides_cutoff: float = 150.0,
    hdl_male_cutoff: float = 40.0,
    hdl_female_cutoff: float = 50.0,
    systolic_bp_cutoff: float = 130.0,
    diastolic_bp_cutoff: float = 85.0,
    glucose_cutoff: float = 100.0,
    minimum_criteria: int = 3,
    status_order: Sequence[str] = (
        "No metabolic syndrome",
        "Metabolic syndrome",
        "Not classifiable",
    ),
) -> pd.DataFrame:
    """Derive five harmonized metabolic-syndrome criteria and status.

    Medication information is not assumed. The criteria therefore represent
    available measured values only and should be described as an operational
    phenotype when treatment variables are unavailable.
    """

    out = df.copy()
    sex = _numeric(out, sex_col)
    waist = _numeric(out, waist_col)
    triglycerides = _numeric(out, triglycerides_col)
    hdl = _numeric(out, hdl_col)
    systolic_bp = _numeric(out, systolic_bp_col)
    diastolic_bp = _numeric(out, diastolic_bp_col)
    glucose = _numeric(out, glucose_col)

    waist_known = waist.notna() & sex.isin([male_value, female_value])
    abdominal_obesity = (
        (sex.eq(male_value) & waist.ge(waist_male_cutoff))
        | (sex.eq(female_value) & waist.ge(waist_female_cutoff))
    )
    out["MetS_Abdominal_Obesity"] = _nullable_indicator(
        abdominal_obesity, waist_known, out.index
    )

    out["MetS_Hypertriglyceridemia"] = _nullable_indicator(
        triglycerides.ge(triglycerides_cutoff), triglycerides.notna(), out.index
    )

    hdl_known = hdl.notna() & sex.isin([male_value, female_value])
    low_hdl = (
        (sex.eq(male_value) & hdl.lt(hdl_male_cutoff))
        | (sex.eq(female_value) & hdl.lt(hdl_female_cutoff))
    )
    out["MetS_Low_HDL"] = _nullable_indicator(low_hdl, hdl_known, out.index)

    # A positive BP criterion is known when either observed value is elevated.
    # A negative BP criterion is only known when BOTH SBP and DBP were measured
    # and both are below their cutoffs. This avoids treating a missing companion
    # measurement as evidence that the criterion is absent.
    elevated_bp = systolic_bp.ge(systolic_bp_cutoff) | diastolic_bp.ge(
        diastolic_bp_cutoff
    )
    bp_known = elevated_bp | (systolic_bp.notna() & diastolic_bp.notna())
    out["MetS_Elevated_BP"] = _nullable_indicator(
        elevated_bp, bp_known, out.index
    )

    out["MetS_Elevated_Glucose"] = _nullable_indicator(
        glucose.ge(glucose_cutoff), glucose.notna(), out.index
    )

    criteria_columns = [
        "MetS_Abdominal_Obesity",
        "MetS_Hypertriglyceridemia",
        "MetS_Low_HDL",
        "MetS_Elevated_BP",
        "MetS_Elevated_Glucose",
    ]
    criteria = out[criteria_columns]
    out["MetS_Available_Criteria"] = criteria.notna().sum(axis=1).astype("Int64")
    out["MetS_Criteria_Count"] = criteria.sum(axis=1, min_count=1).astype("Int64")

    positive = out["MetS_Criteria_Count"].fillna(-1).ge(minimum_criteria)
    complete = out["MetS_Available_Criteria"].eq(len(criteria_columns))
    status = pd.Series("Not classifiable", index=out.index, dtype="string")
    status.loc[complete & ~positive] = "No metabolic syndrome"
    status.loc[positive] = "Metabolic syndrome"
    out["Metabolic_Syndrome_Status"] = pd.Categorical(
        status, categories=list(status_order), ordered=True
    )
    out["MetS_Binary"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[status.eq("No metabolic syndrome"), "MetS_Binary"] = 0
    out.loc[status.eq("Metabolic syndrome"), "MetS_Binary"] = 1

    # Human-readable audit trail for every participant.
    out["MetS_Classification_Rule"] = pd.Series(
        "Fewer than five evaluable criteria and fewer than three positives",
        index=out.index,
        dtype="string",
    )
    out.loc[status.eq("Metabolic syndrome"), "MetS_Classification_Rule"] = (
        "At least three positive criteria"
    )
    out.loc[status.eq("No metabolic syndrome"), "MetS_Classification_Rule"] = (
        "All five criteria evaluable and zero to two positives"
    )
    return out


def _cluster_sort_key(value):
    """Sort numeric cluster identifiers numerically and other identifiers textually."""

    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _cluster_label_anchor_details(cluster_frame: pd.DataFrame) -> dict:
    """Return an observed member nearest the robust bivariate cluster center.

    The anchor is always one of the rows belonging to the cluster. Labels are
    never anchored to a centroid that could fall inside another cluster.
    """

    xy = cluster_frame[["UMAP_GRAPH_PHATE_1", "UMAP_GRAPH_PHATE_2"]].dropna()
    if xy.empty:
        return {
            "anchor_x": np.nan,
            "anchor_y": np.nan,
            "anchor_row_index": pd.NA,
            "anchor_is_observed_cluster_member": False,
        }
    center = xy.median(axis=0).to_numpy(dtype=float)
    scale = xy.quantile(0.75) - xy.quantile(0.25)
    scale = scale.replace(0, 1.0).fillna(1.0).to_numpy(dtype=float)
    standardized = (xy.to_numpy(dtype=float) - center) / scale
    nearest = int(np.argmin(np.sum(standardized**2, axis=1)))
    selected = xy.iloc[nearest]
    return {
        "anchor_x": float(selected["UMAP_GRAPH_PHATE_1"]),
        "anchor_y": float(selected["UMAP_GRAPH_PHATE_2"]),
        "anchor_row_index": str(xy.index[nearest]),
        "anchor_is_observed_cluster_member": True,
    }


def get_cluster_label_anchor(cluster_frame: pd.DataFrame) -> tuple[float, float]:
    """Return the coordinates of an observed member near the robust center."""

    details = _cluster_label_anchor_details(cluster_frame)
    return details["anchor_x"], details["anchor_y"]


def cluster_label_anchors(
    df: pd.DataFrame,
    cluster_col: str,
) -> pd.DataFrame:
    """Audit table for the exact observed point used to anchor every label."""

    rows = []
    for cluster in sorted(df[cluster_col].dropna().unique(), key=_cluster_sort_key):
        frame = df.loc[df[cluster_col].eq(cluster)]
        details = _cluster_label_anchor_details(frame)
        rows.append(
            {
                cluster_col: cluster,
                **details,
                "n": len(frame),
            }
        )
    return pd.DataFrame(rows)


def _plot_categorical(
    ax,
    df: pd.DataFrame,
    column: str,
    *,
    title: str,
    point_size: float,
    alpha: float,
    legend_title: str | None = None,
    color_map: Mapping | None = None,
    draw_order: Sequence | None = None,
    alpha_map: Mapping | None = None,
    edgecolor_map: Mapping | None = None,
    linewidth_map: Mapping | None = None,
):
    """Plot categorical UMAP-graph → PHATE2D coordinates without allowing one class to hide another.

    ``draw_order`` controls only the layer order. The legend retains the
    scientific/category order stored in the categorical dtype.
    """

    values = df[column]
    categories = (
        list(values.cat.categories)
        if isinstance(values.dtype, pd.CategoricalDtype)
        else sorted(values.dropna().unique(), key=lambda value: str(value))
    )
    categories = [category for category in categories if pd.notna(category)]

    requested_order = list(draw_order or categories)
    plot_categories = [category for category in requested_order if category in categories]
    plot_categories += [category for category in categories if category not in plot_categories]

    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    category_index = {category: index for index, category in enumerate(categories)}

    for layer, category in enumerate(plot_categories, start=2):
        mask = values.eq(category)
        category_color = (color_map or {}).get(
            category, cmap(category_index.get(category, 0))
        )
        category_alpha = (alpha_map or {}).get(category, alpha)
        category_edge = (edgecolor_map or {}).get(category, "none")
        category_linewidth = (linewidth_map or {}).get(category, 0.0)
        ax.scatter(
            df.loc[mask, "UMAP_GRAPH_PHATE_1"],
            df.loc[mask, "UMAP_GRAPH_PHATE_2"],
            s=point_size,
            alpha=category_alpha,
            linewidths=category_linewidth,
            edgecolors=category_edge,
            rasterized=False,
            color=category_color,
            zorder=layer,
        )

    missing = values.isna()
    if missing.any():
        ax.scatter(
            df.loc[missing, "UMAP_GRAPH_PHATE_1"],
            df.loc[missing, "UMAP_GRAPH_PHATE_2"],
            s=point_size,
            alpha=0.25,
            linewidths=0,
            rasterized=False,
            color="lightgray",
            zorder=1,
        )

    legend_handles = []
    for category in categories:
        category_color = (color_map or {}).get(
            category, cmap(category_index.get(category, 0))
        )
        category_alpha = (alpha_map or {}).get(category, alpha)
        category_edge = (edgecolor_map or {}).get(category, "none")
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=category_color,
                markeredgecolor=category_edge,
                alpha=category_alpha,
                markersize=5.5,
                label=f"{category} (n={int(values.eq(category).sum())})",
            )
        )
    if missing.any():
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor="lightgray", markeredgecolor="none",
                alpha=0.35, markersize=5.5,
                label=f"Missing (n={int(missing.sum())})",
            )
        )

    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("UMAP-graph PHATE 1")
    ax.set_ylabel("UMAP-graph PHATE 2")
    ax.legend(
        handles=legend_handles,
        title=legend_title,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="best",
    )



def _candidate_label_offsets_points() -> list[tuple[float, float]]:
    """Return deterministic screen-space offsets for collision-free labels."""

    offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    # Cardinal directions are tried before diagonals, then increasingly distant
    # rings. Distances are in typographic points and therefore do not depend on
    # the numerical scale of the selected PHATE2D projection of the UMAP affinity graph built from PCA50.
    directions = [
        (0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0),
        (0.7071, 0.7071), (0.7071, -0.7071),
        (-0.7071, -0.7071), (-0.7071, 0.7071),
    ]
    for radius in (12.0, 18.0, 26.0, 36.0, 48.0, 62.0, 78.0):
        offsets.extend((radius * dx, radius * dy) for dx, dy in directions)
    return offsets


def _bbox_overlap_area(first, second) -> float:
    """Return the intersection area of two display-coordinate bounding boxes."""

    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def _place_collision_aware_labels(ax, label_specs: Sequence[dict]) -> None:
    """Place cluster labels inside one axis and draw safe manual connectors.

    The placement is deterministic and does not use ``adjustText``. Candidate
    positions are evaluated in display coordinates, which prevents labels from
    escaping the panel when the PHATE2D projection of the UMAP affinity graph built from PCA50 has an elongated or fragmented shape.
    Connectors are drawn separately with ``Axes.annotate`` and positive
    ``shrinkA``/``shrinkB`` values, avoiding the FancyArrowPatch transform
    warning and preventing lines from running through the label boxes.
    """

    if not label_specs:
        return

    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis_box = ax.get_window_extent(renderer=renderer)
    points_to_pixels = figure.dpi / 72.0
    boundary_padding = 4.0 * points_to_pixels
    separation_padding = 2.5 * points_to_pixels
    offsets = _candidate_label_offsets_points()

    # Place labels in order of local crowding. Labels with the closest nearby
    # anchor are resolved first; isolated labels normally remain on their anchor.
    anchor_display = np.asarray([
        ax.transData.transform((spec["anchor_x"], spec["anchor_y"]))
        for spec in label_specs
    ], dtype=float)
    if len(anchor_display) > 1:
        pairwise = np.linalg.norm(
            anchor_display[:, None, :] - anchor_display[None, :, :], axis=2
        )
        np.fill_diagonal(pairwise, np.inf)
        nearest_distance = pairwise.min(axis=1)
    else:
        nearest_distance = np.array([np.inf])
    placement_order = np.argsort(nearest_distance, kind="stable")

    placed_boxes = []
    selected: dict[int, tuple[object, tuple[float, float], tuple[float, float]]] = {}

    for spec_index in placement_order:
        spec = label_specs[int(spec_index)]
        anchor_x = float(spec["anchor_x"])
        anchor_y = float(spec["anchor_y"])
        anchor_px = ax.transData.transform((anchor_x, anchor_y))

        text_artist = ax.text(
            anchor_x,
            anchor_y,
            spec["label"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": spec["color"],
                "linewidth": 1.0,
                "alpha": 0.94,
            },
            zorder=10,
        )

        best = None
        for dx_points, dy_points in offsets:
            candidate_px = anchor_px + points_to_pixels * np.array(
                [dx_points, dy_points], dtype=float
            )
            candidate_xy = tuple(ax.transData.inverted().transform(candidate_px))
            text_artist.set_position(candidate_xy)
            candidate_box = text_artist.get_window_extent(renderer=renderer)

            outside = (
                max(0.0, axis_box.x0 + boundary_padding - candidate_box.x0)
                + max(0.0, candidate_box.x1 - axis_box.x1 + boundary_padding)
                + max(0.0, axis_box.y0 + boundary_padding - candidate_box.y0)
                + max(0.0, candidate_box.y1 - axis_box.y1 + boundary_padding)
            )
            overlap = sum(
                _bbox_overlap_area(candidate_box, previous)
                for previous in placed_boxes
            )
            distance_sq = dx_points**2 + dy_points**2
            score = outside * 1.0e9 + overlap * 1.0e6 + distance_sq

            if best is None or score < best[0]:
                best = (
                    score,
                    candidate_xy,
                    candidate_box,
                    (dx_points, dy_points),
                )
            if outside == 0.0 and overlap == 0.0:
                break

        assert best is not None
        _, selected_xy, selected_box, selected_offset = best
        text_artist.set_position(selected_xy)
        placed_boxes.append(selected_box.expanded(
            1.0 + separation_padding / max(selected_box.width, 1.0),
            1.0 + separation_padding / max(selected_box.height, 1.0),
        ))
        selected[int(spec_index)] = (text_artist, selected_xy, selected_offset)

    # Draw connectors only after all labels are fixed. They remain below the
    # text boxes and above the UMAP-graph/PHATE points.
    for spec_index, spec in enumerate(label_specs):
        _, selected_xy, selected_offset = selected[spec_index]
        moved_points = float(np.hypot(*selected_offset))
        if moved_points < 4.0:
            continue
        ax.annotate(
            "",
            xy=(float(spec["anchor_x"]), float(spec["anchor_y"])),
            xytext=selected_xy,
            textcoords="data",
            arrowprops={
                "arrowstyle": "-",
                "color": "0.40",
                "linewidth": 0.65,
                "alpha": 0.78,
                "shrinkA": 9.0,
                "shrinkB": 3.0,
                "connectionstyle": "arc3,rad=0.0",
            },
            annotation_clip=True,
            zorder=7,
        )


def _plot_clusters(
    ax,
    df: pd.DataFrame,
    cluster_col: str,
    *,
    title: str,
    point_size: float,
    alpha: float,
    label_prefix: str,
    label_placement: str = "adjusted_with_connectors",
):
    """Plot clusters with labels tied to verified cluster-member anchors.

    ``adjusted_with_connectors`` is the publication default. Labels are moved
    only when needed to avoid collisions, are constrained to the panel, and
    remain connected to an observed participant from their own cluster.
    ``anchored`` places every label directly on its verified anchor. No mode
    uses ``adjustText``.
    """

    allowed = {"anchored", "adjusted_with_connectors", "none"}
    if label_placement not in allowed:
        raise ValueError(f"label_placement must be one of {sorted(allowed)}")

    clusters = sorted(df[cluster_col].dropna().unique(), key=_cluster_sort_key)
    cmap = plt.get_cmap("nipy_spectral", max(len(clusters), 1))
    label_specs: list[dict] = []

    for index, cluster in enumerate(clusters):
        mask = df[cluster_col].eq(cluster)
        cluster_color = cmap(index)
        ax.scatter(
            df.loc[mask, "UMAP_GRAPH_PHATE_1"],
            df.loc[mask, "UMAP_GRAPH_PHATE_2"],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=False,
            color=cluster_color,
            zorder=2,
        )
        if label_placement == "none":
            continue
        x, y = get_cluster_label_anchor(df.loc[mask])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        # Mark the exact observed anchor so the label-cluster relationship is
        # auditable even when the text is displaced.
        ax.scatter(
            [x],
            [y],
            s=max(point_size * 1.8, 24),
            facecolors="white",
            edgecolors=cluster_color,
            linewidths=1.1,
            zorder=8,
        )
        label_specs.append({
            "label": f"{label_prefix}{cluster}",
            "anchor_x": float(x),
            "anchor_y": float(y),
            "color": cluster_color,
        })

    if label_placement == "anchored":
        for spec in label_specs:
            ax.text(
                spec["anchor_x"],
                spec["anchor_y"],
                spec["label"],
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                clip_on=True,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": spec["color"],
                    "linewidth": 1.0,
                    "alpha": 0.94,
                },
                zorder=10,
            )
        label_specs = []

    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("UMAP-graph PHATE 1")
    ax.set_ylabel("UMAP-graph PHATE 2")
    ax.margins(0.055)
    return label_specs

def plot_umapgraph_phate_four_panels(
    df: pd.DataFrame,
    *,
    cluster_col: str = "Cluster_SHAP",
    hba1c_col: str = "HB1AC",
    glycemic_col: str = "HbA1c_Status",
    metabolic_col: str = "Metabolic_Syndrome_Status",
    glycemic_colors: Mapping | None = None,
    metabolic_colors: Mapping | None = None,
    figure_size: tuple[float, float] = (14, 10),
    point_size: float = 15,
    alpha: float = 0.72,
    cluster_label_placement: str = "adjusted_with_connectors",
):
    """Create the primary PHATE2D projection of the UMAP affinity graph built from PCA50 with verified anchored labels."""

    required = ["UMAP_GRAPH_PHATE_1", "UMAP_GRAPH_PHATE_2", cluster_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing UMAP-graph/PHATE plotting columns: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    cluster_label_specs = _plot_clusters(
        axes[0, 0],
        df,
        cluster_col,
        title="A. Leiden microclusters (shared UMAP graph)",
        point_size=point_size,
        alpha=alpha,
        label_prefix="C",
        label_placement=cluster_label_placement,
    )

    hba1c = _numeric(df, hba1c_col)
    scatter = axes[0, 1].scatter(
        df["UMAP_GRAPH_PHATE_1"],
        df["UMAP_GRAPH_PHATE_2"],
        c=hba1c,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        rasterized=False,
        cmap="viridis",
    )
    axes[0, 1].set_title("B. HbA1c", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("UMAP-graph PHATE 1")
    axes[0, 1].set_ylabel("UMAP-graph PHATE 2")
    fig.colorbar(scatter, ax=axes[0, 1], label="HbA1c (%)")

    _plot_categorical(
        axes[1, 0],
        df,
        glycemic_col,
        title="C. Glycemic status",
        point_size=point_size,
        alpha=alpha,
        legend_title="HbA1c category",
        color_map=glycemic_colors,
    )
    _plot_categorical(
        axes[1, 1],
        df,
        metabolic_col,
        title="D. Metabolic syndrome",
        point_size=point_size,
        alpha=alpha,
        legend_title="Metabolic status",
        color_map=metabolic_colors,
        # Draw gray first, blue second and red last. Incomplete observations
        # remain in the background while metabolic-syndrome cases stay visible.
        draw_order=[
            "Not classifiable",
            "No metabolic syndrome",
            "Metabolic syndrome",
        ],
        alpha_map={
            "Not classifiable": 0.24,
            "No metabolic syndrome": 0.82,
            "Metabolic syndrome": 0.92,
        },
        edgecolor_map={
            "No metabolic syndrome": "#0B3C5D",
            "Metabolic syndrome": "#8E1B12",
        },
        linewidth_map={
            "No metabolic syndrome": 0.08,
            "Metabolic syndrome": 0.08,
        },
    )

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()
    if cluster_label_specs:
        _place_collision_aware_labels(axes[0, 0], cluster_label_specs)
        fig.canvas.draw()
    return fig


def apply_shap_macrocluster_map(
    df: pd.DataFrame,
    merge_map: Mapping,
    *,
    micro_col: str = "Cluster_SHAP",
    macro_col: str = "Cluster_SHAP_macro",
) -> pd.DataFrame:
    """Validate and apply the single reviewed micro-to-macro map."""

    if micro_col not in df.columns:
        raise KeyError(f"Missing microcluster column: {micro_col}")
    observed = set(df[micro_col].dropna().unique().tolist())
    missing_keys = sorted(observed.difference(merge_map), key=lambda value: str(value))
    if missing_keys:
        raise ValueError(f"Microclusters absent from SHAP_CLUSTER_MERGE_MAP: {missing_keys}")
    null_assignments = sorted(
        [cluster for cluster in observed if pd.isna(merge_map.get(cluster))],
        key=lambda value: str(value),
    )
    if null_assignments:
        raise ValueError(
            "Define a macrocluster number for every microcluster before continuing: "
            f"{null_assignments}"
        )
    out = df.copy()
    out[macro_col] = out[micro_col].map(merge_map)
    if out[macro_col].isna().any():
        raise ValueError("The macrocluster map generated missing assignments.")
    try:
        out[macro_col] = out[macro_col].astype(int)
    except (TypeError, ValueError):
        pass
    return out


def macrocluster_tables(
    df: pd.DataFrame,
    *,
    micro_col: str = "Cluster_SHAP",
    macro_col: str = "Cluster_SHAP_macro",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mapping and size tables for manual review/export."""

    mapping = (
        df[[micro_col, macro_col]]
        .drop_duplicates()
        .sort_values([macro_col, micro_col])
        .reset_index(drop=True)
    )
    sizes = (
        df.groupby([macro_col, micro_col], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    sizes["percentage_total"] = 100 * sizes["n"] / sizes["n"].sum()
    return mapping, sizes


def plot_micro_macro_umapgraph_phate(
    df: pd.DataFrame,
    *,
    micro_col: str = "Cluster_SHAP",
    macro_col: str = "Cluster_SHAP_macro",
    figure_size: tuple[float, float] = (14, 6),
    point_size: float = 15,
    alpha: float = 0.72,
    cluster_label_placement: str = "adjusted_with_connectors",
):
    """Plot microclusters and macroclusters in PC1-PC2 with anchored labels."""

    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharex=True, sharey=True)
    micro_label_specs = _plot_clusters(
        axes[0],
        df,
        micro_col,
        title="A. Leiden microclusters (shared UMAP graph)",
        point_size=point_size,
        alpha=alpha,
        label_prefix="C",
        label_placement=cluster_label_placement,
    )
    macro_label_specs = _plot_clusters(
        axes[1],
        df,
        macro_col,
        title="B. Reviewed SHAP macroclusters",
        point_size=point_size,
        alpha=alpha,
        label_prefix="M",
        label_placement=cluster_label_placement,
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()
    if micro_label_specs:
        _place_collision_aware_labels(axes[0], micro_label_specs)
    if macro_label_specs:
        _place_collision_aware_labels(axes[1], macro_label_specs)
    fig.canvas.draw()
    return fig


def save_figure_bundle(
    fig,
    base_path: str | Path,
    *,
    dpi: int = 400,
    extensions: Iterable[str] = ("png", "pdf", "svg"),
) -> list[Path]:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for extension in extensions:
        path = base.with_suffix(f".{extension}")
        kwargs = {"dpi": dpi} if extension.lower() in {"png", "tif", "tiff"} else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    return paths


def build_cluster_summary_tables(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    glycemic_col: str,
    glycemic_order: Sequence[str],
    metabolic_col: str,
    metabolic_order: Sequence[str],
    metabolic_criteria_columns: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Create overall and cluster-level glycemic/metabolic summary tables."""

    glycemic_summary = (
        df[glycemic_col]
        .value_counts(dropna=False)
        .reindex(list(glycemic_order), fill_value=0)
        .rename_axis(glycemic_col)
        .reset_index(name="n")
    )
    glycemic_summary["percentage"] = 100 * glycemic_summary["n"] / len(df)

    metabolic_summary = (
        df[metabolic_col]
        .value_counts(dropna=False)
        .reindex(list(metabolic_order), fill_value=0)
        .rename_axis(metabolic_col)
        .reset_index(name="n")
    )
    metabolic_summary["percentage"] = 100 * metabolic_summary["n"] / len(df)

    glycemic_counts = pd.crosstab(
        df[cluster_col], df[glycemic_col], dropna=False
    ).reindex(columns=list(glycemic_order), fill_value=0)
    glycemic_percentages = glycemic_counts.div(
        glycemic_counts.sum(axis=1).replace(0, np.nan), axis=0
    ).mul(100)

    metabolic_counts = pd.crosstab(
        df[cluster_col], df[metabolic_col], dropna=False
    ).reindex(columns=list(metabolic_order), fill_value=0)
    metabolic_total_percentages = metabolic_counts.div(
        metabolic_counts.sum(axis=1).replace(0, np.nan), axis=0
    ).mul(100)

    classifiable_labels = [
        label for label in metabolic_order if label != "Not classifiable"
    ]
    metabolic_classifiable_counts = metabolic_counts[
        [label for label in classifiable_labels if label in metabolic_counts.columns]
    ].copy()
    metabolic_classifiable_percentages = metabolic_classifiable_counts.div(
        metabolic_classifiable_counts.sum(axis=1).replace(0, np.nan), axis=0
    ).mul(100)

    criteria_percentages = (
        df.groupby(cluster_col, observed=True)[list(metabolic_criteria_columns)]
        .mean()
        .mul(100)
    )
    readable = {
        "MetS_Abdominal_Obesity": "Abdominal obesity (%)",
        "MetS_Hypertriglyceridemia": "Hypertriglyceridemia (%)",
        "MetS_Low_HDL": "Low HDL (%)",
        "MetS_Elevated_BP": "Elevated blood pressure (%)",
        "MetS_Elevated_Glucose": "Elevated glucose (%)",
    }
    criteria_percentages = criteria_percentages.rename(columns=readable)

    return {
        "glycemic_summary": glycemic_summary,
        "metabolic_summary": metabolic_summary,
        "glycemic_counts": glycemic_counts,
        "glycemic_percentages": glycemic_percentages,
        "metabolic_counts": metabolic_counts,
        "metabolic_total_percentages": metabolic_total_percentages,
        "metabolic_classifiable_counts": metabolic_classifiable_counts,
        "metabolic_classifiable_percentages": metabolic_classifiable_percentages,
        "criteria_percentages": criteria_percentages,
    }


__all__ = [
    "derive_glycemic_status",
    "derive_metabolic_syndrome",
    "get_cluster_label_anchor",
    "cluster_label_anchors",
    "plot_umapgraph_phate_four_panels",
    "apply_shap_macrocluster_map",
    "macrocluster_tables",
    "plot_micro_macro_umapgraph_phate",
    "save_figure_bundle",
    "build_cluster_summary_tables",
]
