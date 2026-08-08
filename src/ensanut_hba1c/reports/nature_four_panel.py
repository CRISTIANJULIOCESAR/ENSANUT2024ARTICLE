from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _cluster_sort_key(value):
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value)


def _label_anchor(frame: pd.DataFrame) -> tuple[float, float]:
    xy = frame[["UMAP_GRAPH_PHATE_1", "UMAP_GRAPH_PHATE_2"]].to_numpy(float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if xy.size == 0:
        return np.nan, np.nan
    center = np.nanmedian(xy, axis=0)
    idx = int(np.argmin(np.sum((xy - center) ** 2, axis=1)))
    return float(xy[idx, 0]), float(xy[idx, 1])


def _contrast_text(color) -> str:
    r, g, b = to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.62 else "white"


def _candidate_offsets_points() -> list[tuple[float, float]]:
    offsets = [(0.0, 0.0)]
    directions = [
        (0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0),
        (0.7071, 0.7071), (0.7071, -0.7071),
        (-0.7071, -0.7071), (-0.7071, 0.7071),
    ]
    for radius in (9.0, 14.0, 20.0, 28.0, 38.0, 50.0, 64.0):
        offsets.extend((radius * dx, radius * dy) for dx, dy in directions)
    return offsets


def _overlap_area(first, second) -> float:
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def _place_cluster_labels(ax, label_specs: Sequence[dict]) -> None:
    if not label_specs:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_box = ax.get_window_extent(renderer=renderer)
    points_to_pixels = fig.dpi / 72.0
    boundary_padding = 3.0 * points_to_pixels
    separation_padding = 1.8 * points_to_pixels
    offsets = _candidate_offsets_points()

    anchors_display = np.asarray([
        ax.transData.transform((spec["anchor_x"], spec["anchor_y"]))
        for spec in label_specs
    ])
    if len(anchors_display) > 1:
        distances = np.linalg.norm(
            anchors_display[:, None, :] - anchors_display[None, :, :], axis=2
        )
        np.fill_diagonal(distances, np.inf)
        placement_order = np.argsort(distances.min(axis=1), kind="stable")
    else:
        placement_order = np.array([0])

    placed_boxes = []
    selected = {}

    for spec_index in placement_order:
        spec = label_specs[int(spec_index)]
        anchor = (float(spec["anchor_x"]), float(spec["anchor_y"]))
        anchor_px = ax.transData.transform(anchor)

        text = ax.text(
            *anchor,
            spec["label"],
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            fontfamily="DejaVu Sans Mono",
            color=spec["text_color"],
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": spec["color"],
                "edgecolor": "white",
                "linewidth": 0.55,
                "alpha": 0.96,
            },
            zorder=10,
        )

        best = None
        for dx, dy in offsets:
            candidate_px = anchor_px + points_to_pixels * np.array([dx, dy])
            candidate_xy = tuple(ax.transData.inverted().transform(candidate_px))
            text.set_position(candidate_xy)
            candidate_box = text.get_window_extent(renderer=renderer)

            outside = (
                max(0.0, axis_box.x0 + boundary_padding - candidate_box.x0)
                + max(0.0, candidate_box.x1 - axis_box.x1 + boundary_padding)
                + max(0.0, axis_box.y0 + boundary_padding - candidate_box.y0)
                + max(0.0, candidate_box.y1 - axis_box.y1 + boundary_padding)
            )
            overlap = sum(_overlap_area(candidate_box, old) for old in placed_boxes)
            score = outside * 1e9 + overlap * 1e6 + dx * dx + dy * dy
            if best is None or score < best[0]:
                best = (score, candidate_xy, candidate_box, (dx, dy))
            if outside == 0.0 and overlap == 0.0:
                break

        _, xy, box, offset = best
        text.set_position(xy)
        placed_boxes.append(
            box.expanded(
                1.0 + separation_padding / max(box.width, 1.0),
                1.0 + separation_padding / max(box.height, 1.0),
            )
        )
        selected[int(spec_index)] = (xy, offset)

    for spec_index, spec in enumerate(label_specs):
        xy, offset = selected[spec_index]
        if float(np.hypot(*offset)) < 4.0:
            continue
        ax.annotate(
            "",
            xy=(spec["anchor_x"], spec["anchor_y"]),
            xytext=xy,
            textcoords="data",
            arrowprops={
                "arrowstyle": "-",
                "color": "0.45",
                "linewidth": 0.38,
                "alpha": 0.60,
                "shrinkA": 6.0,
                "shrinkB": 1.5,
            },
            annotation_clip=True,
            zorder=7,
        )


def _style_panel(ax) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(False)
    ax.margins(0.035)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor("0.55")


def _plot_categorical_panel(
    ax,
    df: pd.DataFrame,
    column: str,
    *,
    draw_order: Sequence[str],
    legend_order: Sequence[str] | None = None,
    color_map: Mapping[str, str],
    point_size: float,
    alpha: float,
    alpha_map: Mapping[str, float] | None = None,
    edgecolor_map: Mapping[str, str] | None = None,
    linewidth_map: Mapping[str, float] | None = None,
    legend_location: str = "lower right",
):
    values = df[column].astype("object")
    x = df["UMAP_GRAPH_PHATE_1"]
    y = df["UMAP_GRAPH_PHATE_2"]

    legend_order = list(legend_order or draw_order)
    recognized_categories = set(draw_order) | set(legend_order)

    # The sequence is the physical layer order: first item is drawn at the
    # bottom and the last item is drawn on top.
    for layer, category in enumerate(draw_order, start=2):
        mask = values.eq(category)
        if not mask.any():
            continue
        ax.scatter(
            x[mask], y[mask],
            s=point_size,
            color=color_map[category],
            alpha=(alpha_map or {}).get(category, alpha),
            edgecolors=(edgecolor_map or {}).get(category, "none"),
            linewidths=(linewidth_map or {}).get(category, 0.0),
            rasterized=False,
            zorder=layer,
        )

    missing = values.isna() | ~values.isin(recognized_categories)
    if missing.any():
        ax.scatter(
            x[missing], y[missing],
            s=point_size,
            color="0.80",
            alpha=0.35,
            linewidths=0,
            rasterized=False,
            zorder=1,
        )

    handles = []
    for category in legend_order:
        n = int(values.eq(category).sum())
        if n == 0:
            continue
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=color_map[category],
                markeredgecolor=(edgecolor_map or {}).get(category, "none"),
                alpha=(alpha_map or {}).get(category, alpha),
                markersize=4.2,
                label=f"{category} (n={n})",
            )
        )
    if missing.any():
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor="0.80", markeredgecolor="none",
                markersize=4.2,
                label=f"Not classifiable (n={int(missing.sum())})",
            )
        )

    ax.legend(
        handles=handles,
        loc=legend_location,
        frameon=False,
        fontsize=6.2,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.8,
        markerscale=0.85,
    )
    _style_panel(ax)


def plot_umapgraph_phate_four_panels_nature(
    df: pd.DataFrame,
    *,
    cluster_col: str = "Cluster_SHAP",
    hba1c_col: str = "HB1AC",
    glycemic_col: str = "HbA1c_Status",
    metabolic_col: str = "Metabolic_Syndrome_Status",
    glycemic_colors: Mapping[str, str] | None = None,
    metabolic_colors: Mapping[str, str] | None = None,
    figure_size: tuple[float, float] = (8.8, 6.4),
    point_size: float = 3.6,
    alpha: float = 0.86,
):
    required = {
        "UMAP_GRAPH_PHATE_1", "UMAP_GRAPH_PHATE_2",
        cluster_col, hba1c_col, glycemic_col, metabolic_col,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing columns for the figure: {missing}")

    glycemic_colors = glycemic_colors or {
        "Normal": "#1B9E3E",
        "Prediabetes": "#F39C12",
        "Diabetes": "#D73027",
    }
    metabolic_colors = metabolic_colors or {
        "No metabolic syndrome": "#1976B6",
        "Metabolic syndrome": "#EF3B2C",
        "Not classifiable": "#BDBDBD",
    }

    fig, axes = plt.subplots(
        2, 2,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    x = df["UMAP_GRAPH_PHATE_1"]
    y = df["UMAP_GRAPH_PHATE_2"]

    # Panel 1: discrete microclusters
    clusters = sorted(df[cluster_col].dropna().unique(), key=_cluster_sort_key)
    cmap = plt.get_cmap("tab20", max(len(clusters), 1))
    label_width = max(3, max((len(f"C{c}") for c in clusters), default=3))
    label_specs = []

    for index, cluster in enumerate(clusters):
        mask = df[cluster_col].eq(cluster)
        color = cmap(index)
        axes[0, 0].scatter(
            x[mask], y[mask],
            s=point_size,
            color=color,
            alpha=alpha,
            linewidths=0,
            rasterized=False,
            zorder=2,
        )
        anchor_x, anchor_y = _label_anchor(df.loc[mask])
        if np.isfinite(anchor_x) and np.isfinite(anchor_y):
            label_specs.append({
                "label": f"C{cluster}".center(label_width),
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "color": color,
                "text_color": _contrast_text(color),
            })

    _style_panel(axes[0, 0])

    # Panel 2: HbA1c with a compact color bar inside the panel
    hba1c = pd.to_numeric(df[hba1c_col], errors="coerce")
    valid_hba1c = hba1c.notna()
    if (~valid_hba1c).any():
        axes[0, 1].scatter(
            x[~valid_hba1c], y[~valid_hba1c],
            s=point_size,
            color="0.82",
            alpha=0.30,
            linewidths=0,
            rasterized=False,
            zorder=1,
        )
    scatter = axes[0, 1].scatter(
        x[valid_hba1c], y[valid_hba1c],
        c=hba1c[valid_hba1c],
        s=point_size,
        cmap="viridis",
        alpha=alpha,
        linewidths=0,
        rasterized=False,
        zorder=2,
    )
    _style_panel(axes[0, 1])

    cax = inset_axes(
        axes[0, 1],
        width="28%",
        height="3.0%",
        loc="lower center",
        bbox_to_anchor=(0.0, 0.055, 1.0, 1.0),
        bbox_transform=axes[0, 1].transAxes,
        borderpad=0,
    )
    colorbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    colorbar.ax.set_title("HbA1c (%)", fontsize=6.0, pad=2.0)
    colorbar.ax.tick_params(labelsize=5.2, length=1.3, pad=1.0)
    colorbar.outline.set_linewidth(0.40)

    # Panel 3: glycemic status
    _plot_categorical_panel(
        axes[1, 0],
        df,
        glycemic_col,
        draw_order=["Normal", "Prediabetes", "Diabetes"],
        legend_order=["Normal", "Prediabetes", "Diabetes"],
        color_map=glycemic_colors,
        point_size=point_size,
        alpha=alpha,
        legend_location="lower right",
    )

    # Panel 4: metabolic syndrome
    _plot_categorical_panel(
        axes[1, 1],
        df,
        metabolic_col,
        # Draw gray first, blue second and red last. This keeps incomplete
        # observations in the background and makes the clinical classes clear.
        draw_order=[
            "Not classifiable",
            "No metabolic syndrome",
            "Metabolic syndrome",
        ],
        # Keep the legend in the conventional scientific order.
        legend_order=[
            "No metabolic syndrome",
            "Metabolic syndrome",
            "Not classifiable",
        ],
        color_map=metabolic_colors,
        point_size=point_size,
        alpha=alpha,
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
        legend_location="lower right",
    )

    # Identical limits and visual dimensions in all four panels
    finite_x = pd.to_numeric(x, errors="coerce").to_numpy(float)
    finite_y = pd.to_numeric(y, errors="coerce").to_numpy(float)
    finite = np.isfinite(finite_x) & np.isfinite(finite_y)
    x_min, x_max = finite_x[finite].min(), finite_x[finite].max()
    y_min, y_max = finite_y[finite].min(), finite_y[finite].max()
    x_pad = max((x_max - x_min) * 0.035, 1e-9)
    y_pad = max((y_max - y_min) * 0.035, 1e-9)
    for ax in axes.flat:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        bottom=0.030,
        top=0.985,
        wspace=0.035,
        hspace=0.045,
    )

    fig.canvas.draw()
    _place_cluster_labels(axes[0, 0], label_specs)
    fig.canvas.draw()
    return fig
