"""
Weighted Bayes complex heatmaps for the ENSANUT HbA1c project.

REPLACEMENT FILE
----------------
Copy this file to:

    02_naive_bayes_followup/py/nb_complex_heatmaps.py

Corrections in this version
---------------------------
1. NB+, NB- and net matrices retain the same real row/column labels.
2. Shared clustering is calculated once and reused by every panel.
3. Combined clustering no longer leaks temporary names such as
   ``NBplus__0`` into the order of the original cluster columns.
4. Column clustering treats clusters as observations by clustering a
   transposed matrix.
5. Reordering uses ``.loc`` after resolving labels, never a silent ``reindex``
   that can manufacture an all-NaN matrix.
6. Numeric matrices are validated before plotting and before writing CSVs.
7. Dendrograms remain hidden while similarity ordering is preserved.
8. Matplotlib 3.11 compatibility helpers for PyComplexHeatmap are included.
9. Secondary labels omit repeated primary-section prefixes.
10. A categorical left-side strip identifies the primary section.
11. Its legend is drawn in a dedicated bottom band, never over the heatmap or colorbars.
12. Explicit Matplotlib GridSpec layout prevents label truncation and legend overlap.

Public interface
----------------
    ComplexHeatmapConfig
    generate_weighted_complex_heatmaps(...)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import textwrap
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


try:
    from .translations import text as translated_text
    from .translations import (
        split_hierarchy_label,
        translate_group,
        translate_primary,
        translate_secondary,
    )
except ImportError:
    def translated_text(key: str, language: str = "en") -> str:
        fallback = {
            "primary_title": "Primary questionnaire sections · weighted Bayes evidence",
            "secondary_title": "Secondary questionnaire sections · weighted Bayes evidence",
            "positive_panel": "Weighted positive Bayes evidence (NB+)",
            "negative_panel": "Weighted negative Bayes evidence (NB−)",
            "primary_net_title": "Primary questionnaire sections · net weighted Bayes evidence",
            "secondary_net_title": "Secondary questionnaire sections · net weighted Bayes evidence",
            "net_panel": "Net weighted Bayes evidence",
            "positive_colorbar": "Mean weighted NB+",
            "negative_colorbar": "Mean weighted NB−",
            "net_colorbar": "Mean net weighted evidence",
            "primary_section": "Primary section",
        }
        return fallback.get(key, key)

    def split_hierarchy_label(value: object) -> tuple[str, str | None]:
        return str(value), None

    def translate_group(value: object, level: str, language: str = "en") -> str:
        return str(value).replace("_", " ")

    def translate_primary(value: object, language: str = "en") -> str:
        return str(value).replace("_", " ")

    def translate_secondary(value: object, language: str = "en") -> str:
        return str(value).replace("_", " ")


MODULE_VERSION = "2026-07-22-compact-cluster-strip-v7"


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

@dataclass
class ComplexHeatmapConfig:
    publication_dpi: int = 600
    rasterized: bool = False
    generate_net_heatmap: bool = True
    show_cell_values: bool = False

    language: str = "en"
    font_family: str = "DejaVu Sans"

    positive_cmap: str = "Reds"
    negative_cmap: str = "Blues"
    net_cmap: str = "RdBu_r"
    robust_percentile: float = 95.0

    panel_width_inches: float =1.0
    row_height_inches: float = 0.1
    minimum_figure_height: float = 4.0
    maximum_figure_height: float = 26.0
    column_gap_mm: float = 8.0
    row_split_gap_mm: float = 1.2

    row_label_fontsize: float = 7.5
    column_label_fontsize: float = 8.0
    title_fontsize: float = 13.0
    panel_title_fontsize: float = 10.5
    cluster_annotation_fontsize: float = 7.0

    # Compact publication layout. Titles can be restored from the notebook by
    # setting either option to True.
    show_figure_title: bool = False
    show_panel_titles: bool = False

    show_cluster_size_annotation: bool = True
    show_cluster_strip_annotation: bool = True
    show_bottom_cluster_numbers: bool = True
    # Relative GridSpec heights. The color strip was previously 0.10, which
    # made it visually dominate short heatmaps.
    cluster_size_height_ratio: float = 0.10
    cluster_strip_height_ratio: float = 0.035

    # Secondary-level row annotation: primary questionnaire section.
    show_primary_row_annotation: bool = True
    primary_row_annotation_cmap: str = "tab20"
    primary_row_annotation_width_mm: float = 5.0
    primary_row_annotation_legend: bool = True
    primary_row_annotation_legend_fontsize: float = 7.0
    primary_row_annotation_legend_title_fontsize: float = 8.0
    primary_row_annotation_legend_wrap_width: int = 28
    primary_row_annotation_legend_max_columns: int = 4
    # Retained for backward compatibility; v6 uses a dedicated bottom legend
    # band instead of a right-side legend sidebar.
    primary_row_annotation_sidebar_inches: float = 0.0

    # Shared similarity ordering.
    cluster_rows: bool = True
    cluster_columns: bool = True
    clustering_basis: str = "combined"  # combined, positive, negative, net
    clustering_metric: str = "correlation"
    clustering_method: str = "average"
    cluster_secondary_within_primary: bool = True

    # Trees are calculated externally but not drawn.
    row_dendrogram: bool = False
    col_dendrogram: bool = False


# -------------------------------------------------------------------------
# Compatibility
# -------------------------------------------------------------------------

def install_matplotlib_pycomplexheatmap_compatibility() -> dict[str, Any]:
    """Restore cmap helpers expected by older PyComplexHeatmap releases."""
    installed: list[str] = []

    if not hasattr(mpl_cm, "get_cmap"):
        def _get_cmap(name=None, lut=None):
            cmap = matplotlib.colormaps.get_cmap(name)
            if lut is not None and hasattr(cmap, "resampled"):
                cmap = cmap.resampled(lut)
            return cmap
        mpl_cm.get_cmap = _get_cmap  # type: ignore[attr-defined]
        installed.append("matplotlib.cm.get_cmap")

    if not hasattr(mpl_cm, "register_cmap"):
        def _register_cmap(name=None, cmap=None, override_builtin=False):
            if cmap is None:
                cmap = name
                name = getattr(cmap, "name", None)
            matplotlib.colormaps.register(
                cmap,
                name=name,
                force=bool(override_builtin),
            )
        mpl_cm.register_cmap = _register_cmap  # type: ignore[attr-defined]
        installed.append("matplotlib.cm.register_cmap")

    if not hasattr(mpl_cm, "unregister_cmap"):
        def _unregister_cmap(name):
            matplotlib.colormaps.unregister(name)
        mpl_cm.unregister_cmap = _unregister_cmap  # type: ignore[attr-defined]
        installed.append("matplotlib.cm.unregister_cmap")

    return {
        "matplotlib_version": matplotlib.__version__,
        "compatibility_helpers_installed": installed,
        "get_cmap_available": hasattr(mpl_cm, "get_cmap"),
    }


def ensure_matplotlib_pycomplexheatmap_compatibility() -> dict[str, Any]:
    """Backward-compatible public alias."""
    return install_matplotlib_pycomplexheatmap_compatibility()


# -------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------

def _first_existing(
    columns: Iterable[object],
    aliases: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    available = {str(column).lower(): str(column) for column in columns}
    for alias in aliases:
        match = available.get(str(alias).lower())
        if match is not None:
            return match
    if required:
        raise KeyError(
            "None of the required columns were found: "
            + ", ".join(map(str, aliases))
        )
    return None


def _canonical_level(value: object) -> str:
    text = str(value).strip().lower()
    if "sec" in text:
        return "secondary"
    if "prim" in text:
        return "primary"
    return text


def _canonical_cluster(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        number = float(match.group())
        if number.is_integer():
            return f"C{int(number)}"
        return f"C{number:g}"
    if text.upper().startswith("C"):
        return text.upper()
    return f"C{text}"


def _cluster_sort_key(value: object) -> tuple[int, float | str]:
    canonical = _canonical_cluster(value)
    match = re.search(r"-?\d+(?:\.\d+)?", canonical)
    if match:
        return (0, float(match.group()))
    return (1, canonical)


def _cluster_number_label(value: object) -> str:
    canonical = _canonical_cluster(value)
    return canonical[1:] if canonical.startswith("C") else canonical


def _make_unique(labels: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        output.append(label if counts[label] == 1 else f"{label} [{counts[label]}]")
    return output


def _normalize_visible_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .·>–—-")


def _secondary_row_label(
    value: object,
    *,
    parent: object | None,
    language: str,
) -> str:
    """Return a descriptive secondary label without repeating its parent.

    A previous version replaced parent-only rows with ``General``. When several
    source labels collapsed to that fallback, the figure displayed meaningless
    labels such as ``General [2]``. This version never invents ``General``:

    * explicit children are translated and retained;
    * a repeated parent prefix is removed only when a non-empty child remains;
    * a genuine parent-level row keeps its translated module name.
    """
    raw = _normalize_visible_label(value)
    parent_raw = _normalize_visible_label(parent)
    parent_display = _normalize_visible_label(
        translate_primary(parent_raw, language=language) if parent_raw else ""
    )

    if not raw:
        return "Unspecified subsection" if not language.lower().startswith("es") else "Subsección no especificada"

    # Use the translation module's hierarchy parser first.
    parsed_parent, parsed_child = split_hierarchy_label(raw)
    if parsed_child:
        child_text = _normalize_visible_label(
            translate_secondary(parsed_child, language=language)
        )
        if child_text:
            return child_text

    # Strip a known parent only when useful content remains after the prefix.
    for candidate in (parent_raw, parent_display):
        if not candidate:
            continue
        pattern = re.compile(
            rf"^\s*{re.escape(candidate)}\s*(?:>|·|:|[-–—]|\.)\s*",
            flags=re.IGNORECASE,
        )
        stripped = pattern.sub("", raw, count=1).strip()
        if stripped and stripped.casefold() != raw.casefold():
            translated = _normalize_visible_label(
                translate_secondary(stripped, language=language)
            )
            if translated:
                return translated

    # A Roman numeral reliably marks the beginning of a subsection even when
    # the parent prefix uses an unexpected separator.
    roman = (
        r"(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|"
        r"IX|VIII|VII|VI|V|IV|III|II|I)"
    )
    match = re.search(rf"\b({roman})\s*[.]?\s+.+$", raw, flags=re.IGNORECASE)
    if match and match.start() > 0:
        translated = _normalize_visible_label(
            translate_secondary(raw[match.start():], language=language)
        )
        if translated:
            return translated

    # Handle non-numbered hierarchical labels.
    for separator in (" · ", " > ", " - ", " – ", " — ", ": "):
        if separator in raw:
            child = raw.split(separator, 1)[1].strip()
            if child:
                translated = _normalize_visible_label(
                    translate_secondary(child, language=language)
                )
                if translated:
                    return translated

    # If this is a true parent-level row, retain the translated module name.
    # It is informative and avoids arbitrary ``General [n]`` labels.
    if any(
        raw.casefold() == candidate.casefold()
        for candidate in (parent_raw, parent_display)
        if candidate
    ):
        return parent_display or translate_secondary(raw, language=language)

    return _normalize_visible_label(translate_secondary(raw, language=language))

def _sample_category_colors(
    labels: list[str],
    cmap_name: str = "tab20",
) -> dict[str, str]:
    if not labels:
        return {}
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    positions = np.linspace(0.02, 0.98, max(len(labels), 2))
    return {
        label: matplotlib.colors.to_hex(cmap(positions[index]))
        for index, label in enumerate(labels)
    }


def _robust_positive_limit(matrix: pd.DataFrame, percentile: float) -> float:
    values = pd.to_numeric(
        pd.Series(matrix.to_numpy().ravel()),
        errors="coerce",
    ).dropna()
    values = values[values >= 0]
    if values.empty:
        return 1.0
    limit = float(np.nanpercentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(values.max())
    return max(limit, np.finfo(float).eps)


def _robust_symmetric_limit(matrix: pd.DataFrame, percentile: float) -> float:
    values = pd.to_numeric(
        pd.Series(matrix.to_numpy().ravel()),
        errors="coerce",
    ).dropna().abs()
    if values.empty:
        return 1.0
    limit = float(np.nanpercentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(values.max())
    return max(limit, np.finfo(float).eps)


def _save_figure_bundle(
    figure: plt.Figure,
    output_stem: Path,
    *,
    dpi: int,
) -> dict[str, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for extension in ("png", "pdf", "svg"):
        path = output_stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if extension == "png":
            kwargs["dpi"] = dpi
        figure.savefig(path, **kwargs)
        paths[extension] = path
    return paths


def _normalize_config(
    config: ComplexHeatmapConfig | Mapping[str, Any] | None,
) -> ComplexHeatmapConfig:
    if config is None:
        return ComplexHeatmapConfig()
    if isinstance(config, ComplexHeatmapConfig):
        return config
    if isinstance(config, Mapping):
        allowed = set(ComplexHeatmapConfig.__dataclass_fields__)
        return ComplexHeatmapConfig(
            **{key: value for key, value in config.items() if key in allowed}
        )
    raise TypeError("config must be ComplexHeatmapConfig, a mapping, or None.")


# -------------------------------------------------------------------------
# Hierarchy and matrix preparation
# -------------------------------------------------------------------------

def _hierarchy_parent_map(
    hierarchy_structure: pd.DataFrame | None,
) -> dict[str, str]:
    if hierarchy_structure is None or hierarchy_structure.empty:
        return {}

    primary_col = _first_existing(
        hierarchy_structure.columns,
        ["nivel_primario", "primary_group", "primary", "nivel_1"],
        required=False,
    )
    secondary_col = _first_existing(
        hierarchy_structure.columns,
        ["nivel_secundario", "secondary_group", "secondary", "nivel_2"],
        required=False,
    )
    if primary_col is None or secondary_col is None:
        return {}

    mapping: dict[str, str] = {}
    subset = hierarchy_structure[[primary_col, secondary_col]].dropna()
    for primary, secondary in subset.itertuples(index=False, name=None):
        primary_text = str(primary).strip()
        secondary_text = str(secondary).strip()
        mapping[secondary_text] = primary_text
        mapping[f"{primary_text} > {secondary_text}"] = primary_text
        mapping[f"{primary_text}.{secondary_text}"] = primary_text
    return mapping


def _hierarchy_orders(
    hierarchy_structure: pd.DataFrame | None,
) -> tuple[list[str], list[str]]:
    if hierarchy_structure is None or hierarchy_structure.empty:
        return [], []

    primary_col = _first_existing(
        hierarchy_structure.columns,
        ["nivel_primario", "primary_group", "primary", "nivel_1"],
        required=False,
    )
    secondary_col = _first_existing(
        hierarchy_structure.columns,
        ["nivel_secundario", "secondary_group", "secondary", "nivel_2"],
        required=False,
    )

    primary_order: list[str] = []
    secondary_order: list[str] = []

    if primary_col is not None:
        primary_order = (
            hierarchy_structure[primary_col]
            .dropna().astype(str).drop_duplicates().tolist()
        )

    if secondary_col is not None:
        if primary_col is not None:
            pairs = hierarchy_structure[[primary_col, secondary_col]].dropna()
            secondary_order = [
                f"{primary} > {secondary}"
                for primary, secondary in pairs.drop_duplicates().itertuples(
                    index=False,
                    name=None,
                )
            ]
        else:
            secondary_order = (
                hierarchy_structure[secondary_col]
                .dropna().astype(str).drop_duplicates().tolist()
            )

    return primary_order, secondary_order


def _resolve_summary_columns(summary: pd.DataFrame) -> dict[str, str | None]:
    return {
        "level": _first_existing(
            summary.columns,
            ["level", "hierarchy_level", "nivel"],
        ),
        "group": _first_existing(
            summary.columns,
            ["group", "group_name", "section", "grupo", "subcategoria"],
        ),
        "cluster": _first_existing(
            summary.columns,
            ["cluster", "cluster_label", "Cluster_SHAP", "microcluster"],
        ),
        "positive": _first_existing(
            summary.columns,
            [
                "positive_evidence_per_variable",
                "positive_evidence_weighted",
                "weighted_positive_evidence",
                "positive_rule_score_mean",
            ],
        ),
        "negative": _first_existing(
            summary.columns,
            [
                "negative_evidence_per_variable",
                "negative_evidence_weighted",
                "weighted_negative_evidence",
                "negative_rule_score_mean_abs",
            ],
        ),
        "net": _first_existing(
            summary.columns,
            [
                "net_evidence_per_variable",
                "net_evidence_weighted",
                "weighted_net_evidence",
                "net_rule_score_mean",
            ],
            required=False,
        ),
        "parent": _first_existing(
            summary.columns,
            [
                "parent_group",
                "primary_group",
                "nivel_primario",
                "primary_section",
            ],
            required=False,
        ),
        "cluster_size": _first_existing(
            summary.columns,
            [
                "cluster_size",
                "n_cluster",
                "cluster_n",
                "n_positive",
                "cluster_count",
            ],
            required=False,
        ),
        "variables": _first_existing(
            summary.columns,
            [
                "n_encoded_variables",
                "n_usable_variables",
                "effective_variable_count",
                "n_group_variables",
            ],
            required=False,
        ),
        "coverage": _first_existing(
            summary.columns,
            [
                "mean_variable_coverage",
                "variable_coverage_mean",
                "coverage",
            ],
            required=False,
        ),
    }


def _ordered_groups(
    available_groups: list[str],
    preferred_order: list[str],
) -> list[str]:
    available_set = set(available_groups)
    ordered: list[str] = []

    for value in preferred_order:
        if value in available_set and value not in ordered:
            ordered.append(value)

    for preferred in preferred_order:
        if ">" not in preferred:
            continue
        child = preferred.split(">", 1)[1].strip()
        matches = [value for value in available_groups if value == child]
        if len(matches) == 1 and matches[0] not in ordered:
            ordered.append(matches[0])

    ordered.extend(value for value in available_groups if value not in ordered)
    return ordered


def _numeric_pivot(
    data: pd.DataFrame,
    *,
    value_column: str,
    groups: list[str],
    clusters: list[str],
) -> pd.DataFrame:
    matrix = data.pivot_table(
        index="_group",
        columns="_cluster",
        values=value_column,
        aggfunc="mean",
    ).reindex(index=groups, columns=clusters)

    # A missing group x cluster combination means no observed evidence and is
    # represented as zero. This fill happens before clustering; a later broken
    # reorder is still detected by validation and is never silently filled.
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _prepare_level_data(
    summary: pd.DataFrame,
    hierarchy_structure: pd.DataFrame | None,
    *,
    level: str,
    language: str,
) -> dict[str, Any] | None:
    columns = _resolve_summary_columns(summary)
    data = summary.copy()
    data["_level"] = data[columns["level"]].map(_canonical_level)
    data = data.loc[data["_level"].eq(level)].copy()
    if data.empty:
        return None

    data["_group"] = data[columns["group"]].astype(str).str.strip()
    data["_cluster"] = data[columns["cluster"]].map(_canonical_cluster)
    data["_positive"] = pd.to_numeric(data[columns["positive"]], errors="coerce")
    data["_negative"] = pd.to_numeric(
        data[columns["negative"]],
        errors="coerce",
    ).abs()

    if columns["net"] is not None:
        data["_net"] = pd.to_numeric(data[columns["net"]], errors="coerce")
    else:
        data["_net"] = data["_positive"] - data["_negative"]

    if not np.isfinite(data[["_positive", "_negative", "_net"]].to_numpy(dtype=float)).any():
        raise ValueError(
            f"The {level} summary contains no finite weighted Bayes evidence values."
        )

    clusters = sorted(
        data["_cluster"].dropna().unique().tolist(),
        key=_cluster_sort_key,
    )

    primary_order, secondary_order = _hierarchy_orders(hierarchy_structure)
    preferred_order = primary_order if level == "primary" else secondary_order
    available_groups = data["_group"].drop_duplicates().tolist()
    groups = _ordered_groups(available_groups, preferred_order)

    positive = _numeric_pivot(
        data,
        value_column="_positive",
        groups=groups,
        clusters=clusters,
    )
    negative = _numeric_pivot(
        data,
        value_column="_negative",
        groups=groups,
        clusters=clusters,
    )
    net = _numeric_pivot(
        data,
        value_column="_net",
        groups=groups,
        clusters=clusters,
    )

    parent_map = _hierarchy_parent_map(hierarchy_structure)
    if columns["parent"] is not None:
        parent_lookup = (
            data[["_group", columns["parent"]]]
            .dropna().drop_duplicates("_group")
            .set_index("_group")[columns["parent"]]
            .astype(str).to_dict()
        )
        parent_map.update(parent_lookup)

    parent_values: list[str] = []
    for group in groups:
        parent = parent_map.get(group)
        parsed_parent, parsed_child = split_hierarchy_label(group)
        if parent is None and parsed_child:
            parent = parsed_parent
        parent_values.append(parent or group)

    if level == "secondary":
        # The primary section is encoded by the lateral color strip. Visible
        # row names therefore contain only the secondary subsection. Display
        # labels are allowed to repeat because the color strip disambiguates
        # their parent modules. Matrices keep the original unique group keys,
        # so arbitrary suffixes such as ``[2]`` are never required.
        translated_rows = [
            _secondary_row_label(
                group,
                parent=parent,
                language=language,
            )
            for group, parent in zip(groups, parent_values)
        ]
    else:
        translated_rows = [
            translate_group(group, level=level, language=language)
            for group in groups
        ]

    display_columns = [_cluster_number_label(cluster) for cluster in clusters]
    column_rename = dict(zip(clusters, display_columns))
    positive = positive.rename(columns=column_rename)
    negative = negative.rename(columns=column_rename)
    net = net.rename(columns=column_rename)

    cluster_sizes = pd.Series(index=display_columns, dtype=float)
    if columns["cluster_size"] is not None:
        size_lookup = (
            data[["_cluster", columns["cluster_size"]]]
            .assign(
                _size=lambda frame: pd.to_numeric(
                    frame[columns["cluster_size"]],
                    errors="coerce",
                )
            )
            .groupby("_cluster", observed=True)["_size"]
            .max()
        )
        cluster_sizes = pd.Series(
            {
                _cluster_number_label(cluster): size_lookup.get(cluster, np.nan)
                for cluster in clusters
            },
            dtype=float,
        )

    parent_display = [
        translate_group(parent, level="primary", language=language)
        for parent in parent_values
    ]
    row_split = pd.Series(
        parent_display,
        index=groups,
        name=translated_text("primary_section", language),
        dtype="object",
    )
    row_split_order = list(dict.fromkeys(parent_display))

    return {
        "level": level,
        "positive": positive,
        "negative": negative,
        "net": net,
        "clusters": clusters,
        "display_columns": display_columns,
        "cluster_sizes": cluster_sizes.reindex(display_columns),
        "row_split": row_split if level == "secondary" else None,
        "row_split_order": row_split_order if level == "secondary" else None,
        "row_metadata": pd.DataFrame(
            {
                "original_group": groups,
                "display_group": translated_rows,
                "parent_group": parent_values,
                "display_parent_group": parent_display,
            },
            index=groups,
        ),
        "column_metadata": pd.DataFrame(
            {
                "cluster": clusters,
                "display_column": display_columns,
                "cluster_size": cluster_sizes.reindex(display_columns).to_numpy(),
            },
            index=display_columns,
        ),
    }


def _sanitize_ticklabel_kws(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize legacy tick-label keywords for Matplotlib 3.11+."""
    cleaned = dict(values or {})
    if "fontsize" in cleaned and "labelsize" not in cleaned:
        cleaned["labelsize"] = cleaned.pop("fontsize")
    else:
        cleaned.pop("fontsize", None)
    if "rotation" in cleaned and "labelrotation" not in cleaned:
        cleaned["labelrotation"] = cleaned.pop("rotation")
    else:
        cleaned.pop("rotation", None)
    return cleaned


def prepare_complex_heatmap_data(
    summary: pd.DataFrame,
    hierarchy_structure: pd.DataFrame | None,
    *,
    level: str,
    language: str = "en",
) -> dict[str, Any]:
    """Public compatibility wrapper around the level-data preparation step."""
    prepared = _prepare_level_data(
        summary,
        hierarchy_structure,
        level=_canonical_level(level),
        language=language,
    )
    if prepared is None:
        raise ValueError(f"No rows were available for hierarchy level: {level}")
    return prepared


# -------------------------------------------------------------------------
# Validation and shared similarity ordering
# -------------------------------------------------------------------------

def _validate_numeric_heatmap_matrix(
    matrix: pd.DataFrame,
    *,
    matrix_name: str,
) -> pd.DataFrame:
    """Return a numeric matrix or stop before PyComplexHeatmap sees bad data."""
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError(f"{matrix_name} is not a pandas DataFrame.")
    if matrix.empty:
        raise ValueError(f"{matrix_name} is empty.")

    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    finite_mask = np.isfinite(values)

    if not finite_mask.any():
        raise ValueError(
            f"{matrix_name} contains zero finite values. "
            f"Rows sample={list(matrix.index[:5])}; "
            f"columns sample={list(matrix.columns[:5])}."
        )

    empty_rows = numeric.isna().all(axis=1)
    empty_columns = numeric.isna().all(axis=0)
    if empty_rows.any():
        labels = numeric.index[empty_rows].astype(str).tolist()
        raise ValueError(
            f"{matrix_name} contains completely empty rows: "
            + ", ".join(labels[:20])
        )
    if empty_columns.any():
        labels = numeric.columns[empty_columns].astype(str).tolist()
        raise ValueError(
            f"{matrix_name} contains completely empty columns: "
            + ", ".join(labels[:20])
        )

    return numeric


def _matrix_for_clustering(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    net: pd.DataFrame,
    *,
    basis: str,
) -> pd.DataFrame:
    basis_key = str(basis).strip().lower()

    if basis_key == "positive":
        matrix = positive.copy()
    elif basis_key == "negative":
        matrix = negative.copy()
    elif basis_key == "net":
        matrix = net.copy()
    elif basis_key == "combined":
        def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
            values = frame.astype(float)
            means = values.mean(axis=0)
            stds = values.std(axis=0, ddof=0).replace(0, 1.0)
            return (values - means) / stds

        pos = _standardize(positive.copy())
        neg = _standardize(negative.copy())
        pos.columns = [f"NBplus__{column}" for column in pos.columns]
        neg.columns = [f"NBminus__{column}" for column in neg.columns]
        matrix = pd.concat([pos, neg], axis=1)
    else:
        raise ValueError(
            "clustering_basis must be one of: combined, positive, negative, net."
        )

    return matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_leaves_order(
    matrix: pd.DataFrame,
    *,
    axis: int,
    metric: str,
    method: str,
) -> list[Any]:
    labels = matrix.index.tolist() if axis == 0 else matrix.columns.tolist()
    if len(labels) <= 2:
        return labels

    values = matrix.to_numpy(dtype=float)
    if axis == 1:
        values = values.T

    try:
        distances = pdist(values, metric=metric)
        if not np.isfinite(distances).all():
            raise ValueError("Non-finite clustering distances.")
    except Exception:
        distances = pdist(values, metric="euclidean")

    if distances.size == 0 or not np.isfinite(distances).all():
        return labels

    try:
        tree = linkage(distances, method=method, optimal_ordering=True)
        order = leaves_list(tree).tolist()
        return [labels[index] for index in order]
    except Exception:
        return labels


def _resolve_axis_order(
    requested_order: Iterable[Any],
    actual_labels: Iterable[Any],
) -> list[Any]:
    """Resolve an order to exact existing labels without changing their type."""
    actual = list(actual_labels)
    actual_by_text = {str(label): label for label in actual}
    resolved: list[Any] = []

    for requested in requested_order:
        if requested in actual and requested not in resolved:
            resolved.append(requested)
            continue
        exact = actual_by_text.get(str(requested))
        if exact is not None and exact not in resolved:
            resolved.append(exact)

    resolved.extend(label for label in actual if label not in resolved)
    return resolved


def _shared_similarity_orders(
    data: dict[str, Any],
    config: ComplexHeatmapConfig,
) -> tuple[list[Any], list[Any]]:
    positive = _validate_numeric_heatmap_matrix(
        data["positive"],
        matrix_name=f"{data['level']}_positive_before_order",
    )
    negative = _validate_numeric_heatmap_matrix(
        data["negative"],
        matrix_name=f"{data['level']}_negative_before_order",
    )
    net = _validate_numeric_heatmap_matrix(
        data["net"],
        matrix_name=f"{data['level']}_net_before_order",
    )

    # These are exact-label checks; unlike reindex they never create new axes.
    if not positive.index.equals(negative.index) or not positive.columns.equals(negative.columns):
        raise ValueError("NB+ and NB- matrices do not have identical axes before ordering.")
    if not positive.index.equals(net.index) or not positive.columns.equals(net.columns):
        raise ValueError("NB+ and net matrices do not have identical axes before ordering.")

    # Rows: questionnaire sections are observations.
    row_clustering_matrix = _matrix_for_clustering(
        positive,
        negative,
        net,
        basis=config.clustering_basis,
    )

    # Columns: clusters must be observations. Transpose FIRST, then construct
    # the selected clustering representation. The resulting index contains the
    # true cluster labels, not temporary NBplus__/NBminus__ feature names.
    column_clustering_matrix = _matrix_for_clustering(
        positive.T,
        negative.T,
        net.T,
        basis=config.clustering_basis,
    )

    if config.cluster_columns:
        column_order = _safe_leaves_order(
            column_clustering_matrix,
            axis=0,
            metric=config.clustering_metric,
            method=config.clustering_method,
        )
    else:
        column_order = positive.columns.tolist()
    column_order = _resolve_axis_order(column_order, positive.columns)

    if not config.cluster_rows:
        row_order = positive.index.tolist()
    elif (
        data["level"] == "secondary"
        and config.cluster_secondary_within_primary
        and data.get("row_split") is not None
    ):
        row_order: list[Any] = []
        split_series = data["row_split"].loc[positive.index]
        split_order = data.get("row_split_order") or (
            split_series.dropna().drop_duplicates().tolist()
        )

        for parent in split_order:
            block_rows = split_series.index[split_series.eq(parent)].tolist()
            if not block_rows:
                continue
            block_matrix = row_clustering_matrix.loc[block_rows]
            row_order.extend(
                _safe_leaves_order(
                    block_matrix,
                    axis=0,
                    metric=config.clustering_metric,
                    method=config.clustering_method,
                )
            )
        row_order = _resolve_axis_order(row_order, positive.index)
    else:
        row_order = _safe_leaves_order(
            row_clustering_matrix,
            axis=0,
            metric=config.clustering_metric,
            method=config.clustering_method,
        )
        row_order = _resolve_axis_order(row_order, positive.index)

    return row_order, column_order


def _apply_shared_similarity_order(
    data: dict[str, Any],
    config: ComplexHeatmapConfig,
) -> dict[str, Any]:
    row_order, column_order = _shared_similarity_orders(data, config)

    ordered = dict(data)
    for key in ("positive", "negative", "net"):
        # .loc fails loudly if an order contains an invalid label. That is the
        # intended behavior; reindex would silently manufacture NaNs.
        ordered[key] = data[key].loc[row_order, column_order].copy()
        ordered[key] = _validate_numeric_heatmap_matrix(
            ordered[key],
            matrix_name=f"{data['level']}_{key}_after_order",
        )

    ordered["display_columns"] = list(column_order)
    ordered["cluster_sizes"] = data["cluster_sizes"].loc[column_order]
    ordered["row_metadata"] = data["row_metadata"].loc[row_order]
    ordered["column_metadata"] = data["column_metadata"].loc[column_order]

    if data.get("row_split") is not None:
        ordered["row_split"] = data["row_split"].loc[row_order]

    ordered["similarity_row_order"] = list(row_order)
    ordered["similarity_column_order"] = list(column_order)
    return ordered


# -------------------------------------------------------------------------
# PyComplexHeatmap construction
# -------------------------------------------------------------------------

def _import_pch():
    install_matplotlib_pycomplexheatmap_compatibility()
    try:
        import PyComplexHeatmap as pch
    except ImportError as exc:
        raise ImportError(
            "PyComplexHeatmap is required. Install it with:\n"
            "pip install PyComplexHeatmap==1.8.5"
        ) from exc
    return pch


def _top_annotation(
    pch,
    display_columns: list[str],
    cluster_sizes: pd.Series,
    *,
    config: ComplexHeatmapConfig,
):
    annotations: dict[str, Any] = {}

    canonical_labels = [f"C{column}" for column in display_columns]
    cluster_colors = _sample_category_colors(canonical_labels, "tab20")
    cluster_series = pd.Series(
        canonical_labels,
        index=display_columns,
        name="Cluster",
    )

    if config.show_cluster_strip_annotation:
        annotations["Cluster"] = pch.anno_simple(
            cluster_series,
            colors=cluster_colors,
            legend=False,
            height=5,
            add_text=True,
            text_kws={
                "color": "white",
                "fontsize": config.cluster_annotation_fontsize,
                "fontweight": "bold",
            },
        )

    if (
        config.show_cluster_size_annotation
        and cluster_sizes.notna().any()
        and float(cluster_sizes.fillna(0).max()) > 0
    ):
        annotations["N"] = pch.anno_barplot(
            cluster_sizes.fillna(0),
            cmap="Greys",
            height=8,
            linewidth=0.35,
            grid=False,
            legend=False,
            label=False,
        )

    if not annotations:
        return None

    return pch.HeatmapAnnotation(
        **annotations,
        axis=1,
        verbose=0,
        label_kws={"visible": False},
    )


def _primary_annotation_spec(
    row_split: pd.Series | None,
    row_split_order: list[str] | None,
    *,
    config: ComplexHeatmapConfig,
) -> tuple[list[str], dict[str, str]]:
    """Return the ordered primary-section categories and their fixed colors."""
    if row_split is None or row_split.empty:
        return [], {}

    categories = list(row_split_order or [])
    categories.extend(
        value
        for value in row_split.dropna().astype(str).drop_duplicates().tolist()
        if value not in categories
    )
    return categories, _sample_category_colors(
        categories,
        config.primary_row_annotation_cmap,
    )


def _left_primary_annotation(
    pch,
    row_split: pd.Series | None,
    row_split_order: list[str] | None,
    *,
    config: ComplexHeatmapConfig,
):
    """Create the categorical row strip without an automatic PCH legend.

    PyComplexHeatmap places annotation legends in the same generic right-side
    legend area used by heatmap legends. This module draws numerical colorbars
    at fixed right-side coordinates, so an automatic annotation legend can
    overlap those colorbars. The strip is therefore rendered here with every
    internal legend switch disabled; a separate Matplotlib legend is added by
    ``_add_primary_section_legend`` in a reserved sidebar.
    """
    if (
        not config.show_primary_row_annotation
        or row_split is None
        or row_split.empty
    ):
        return None

    categories, colors = _primary_annotation_spec(
        row_split,
        row_split_order,
        config=config,
    )
    if not categories:
        return None

    annotation_name = translated_text("primary_section", config.language)

    return pch.HeatmapAnnotation(
        **{
            annotation_name: pch.anno_simple(
                row_split.astype(str),
                colors=colors,
                add_text=False,
                legend=False,
                height=config.primary_row_annotation_width_mm,
            )
        },
        axis=0,
        label_side="top",
        wgap=0.2,
        verbose=0,
        legend=False,
        plot_legend=False,
        label_kws={"visible": False},
    )


def _has_primary_section_legend(
    data: dict[str, Any],
    config: ComplexHeatmapConfig,
) -> bool:
    return bool(
        data.get("level") == "secondary"
        and config.show_primary_row_annotation
        and config.primary_row_annotation_legend
        and data.get("row_split") is not None
        and not data["row_split"].empty
    )


def _add_primary_section_legend(
    figure: plt.Figure,
    *,
    row_split: pd.Series | None,
    row_split_order: list[str] | None,
    config: ComplexHeatmapConfig,
    anchor_x: float,
    anchor_y: float = 0.50,
) -> None:
    """Draw the module legend in its own sidebar, away from colorbars."""
    if (
        not config.primary_row_annotation_legend
        or row_split is None
        or row_split.empty
    ):
        return

    categories, colors = _primary_annotation_spec(
        row_split,
        row_split_order,
        config=config,
    )
    if not categories:
        return

    wrap_width = max(int(config.primary_row_annotation_legend_wrap_width), 12)
    handles = [
        Patch(
            facecolor=colors[category],
            edgecolor="none",
            label=textwrap.fill(str(category), width=wrap_width),
        )
        for category in categories
    ]

    legend = figure.legend(
        handles=handles,
        title=translated_text("primary_section", config.language),
        loc="center left",
        bbox_to_anchor=(anchor_x, anchor_y),
        bbox_transform=figure.transFigure,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.1,
        handleheight=0.9,
        handletextpad=0.55,
        labelspacing=0.55,
        fontsize=config.primary_row_annotation_legend_fontsize,
        ncol=1,
    )
    legend.get_title().set_fontsize(
        config.primary_row_annotation_legend_title_fontsize
    )
    legend.get_title().set_fontweight("bold")


def _heatmap_plotter(
    pch,
    matrix: pd.DataFrame,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    top_annotation,
    left_annotation,
    row_split: pd.Series | None,
    row_split_order: list[str] | None,
    show_rownames: bool,
    config: ComplexHeatmapConfig,
):
    matrix = _validate_numeric_heatmap_matrix(
        matrix,
        matrix_name="plot_matrix",
    )

    kwargs: dict[str, Any] = {
        "data": matrix,
        "top_annotation": top_annotation,
        "left_annotation": left_annotation,
        "row_cluster": False,
        "col_cluster": False,
        "row_dendrogram": False,
        "col_dendrogram": False,
        "show_rownames": show_rownames,
        "show_colnames": config.show_bottom_cluster_numbers,
        "row_names_side": "left",
        "col_names_side": "bottom",
        "xticklabels_kws": {
            "labelsize": config.column_label_fontsize,
            "labelrotation": 0,
            "pad": 2,
        },
        "yticklabels_kws": {
            "labelsize": config.row_label_fontsize,
            "labelrotation": 0,
            "pad": 3,
        },
        "row_split": row_split,
        "row_split_order": row_split_order,
        "row_split_gap": config.row_split_gap_mm,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "rasterized": config.rasterized,
        "legend": False,
        "plot_legend": False,
        "plot": False,
        "verbose": 0,
        "linewidths": 0.35,
        "linecolor": "white",
    }

    if config.show_cell_values:
        kwargs["annot"] = matrix.round(2)
        kwargs["fmt"] = ".2f"
        kwargs["annot_kws"] = {"fontsize": 5.5}

    return pch.ClusterMapPlotter(**kwargs)


def _estimate_row_label_width_inches(
    labels: Sequence[object],
    *,
    fontsize: float,
) -> float:
    """Estimate enough physical width to prevent row-label clipping."""
    normalized = [str(label) for label in labels if str(label)]
    longest = max((len(label) for label in normalized), default=0)
    estimate = longest * float(fontsize) * 0.56 / 72.0 + 0.55
    return float(np.clip(estimate, 2.2, 7.5))


def _row_group_boundaries(row_split: pd.Series | None, index: pd.Index) -> list[float]:
    if row_split is None or row_split.empty or len(index) <= 1:
        return []
    values = row_split.reindex(index).astype(str).tolist()
    return [
        position - 0.5
        for position in range(1, len(values))
        if values[position] != values[position - 1]
    ]


def _copy_cmap_with_white_nan(cmap_name: str):
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    try:
        cmap = cmap.copy()
    except AttributeError:
        pass
    try:
        cmap.set_bad("white")
    except Exception:
        pass
    return cmap


def _draw_heatmap_axis(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    *,
    cmap: str,
    norm,
    config: ComplexHeatmapConfig,
    show_ylabels: bool,
    ylabels: Sequence[str],
    title: str,
    row_boundaries: Sequence[float],
) -> Any:
    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    image = axis.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=_copy_cmap_with_white_nan(cmap),
        norm=norm,
        rasterized=config.rasterized,
    )

    if config.show_panel_titles and str(title).strip():
        axis.set_title(
            title,
            fontsize=config.panel_title_fontsize,
            fontweight="bold",
            pad=6,
        )
    else:
        # Explicitly clear the axis title so rerunning a notebook cell cannot
        # retain a title from an earlier version of the plotting code.
        axis.set_title("")
    axis.set_xticks(np.arange(matrix.shape[1]))
    if config.show_bottom_cluster_numbers:
        axis.set_xticklabels(
            [str(column) for column in matrix.columns],
            fontsize=config.column_label_fontsize,
            rotation=0,
        )
    else:
        axis.set_xticklabels([])

    axis.set_yticks(np.arange(matrix.shape[0]))
    if show_ylabels:
        axis.set_yticklabels(
            list(ylabels),
            fontsize=config.row_label_fontsize,
            rotation=0,
        )
        axis.tick_params(axis="y", pad=5, length=0)
        for tick in axis.get_yticklabels():
            tick.set_horizontalalignment("right")
            tick.set_clip_on(False)
    else:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)

    axis.tick_params(axis="x", pad=3, length=0)
    axis.set_xlim(-0.5, matrix.shape[1] - 0.5)
    axis.set_ylim(matrix.shape[0] - 0.5, -0.5)

    for boundary in row_boundaries:
        axis.axhline(boundary, color="white", linewidth=2.2, zorder=5)

    axis.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.35)
    axis.tick_params(which="minor", bottom=False, left=False)

    if config.show_cell_values:
        values = matrix.to_numpy(dtype=float)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                    )

    return image


def _draw_cluster_strip(
    axis: plt.Axes,
    columns: Sequence[object],
    *,
    config: ComplexHeatmapConfig,
) -> None:
    if not config.show_cluster_strip_annotation:
        axis.axis("off")
        return

    canonical = [f"C{column}" for column in columns]
    colors = _sample_category_colors(canonical, "tab20")
    rgba = np.asarray([
        matplotlib.colors.to_rgba(colors[label]) for label in canonical
    ])[None, :, :]
    axis.imshow(rgba, aspect="auto", interpolation="nearest", origin="upper")
    axis.set_xlim(-0.5, len(columns) - 0.5)
    axis.set_ylim(0.5, -0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    for position, label in enumerate(canonical):
        axis.text(
            position,
            0,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=config.cluster_annotation_fontsize,
            fontweight="bold",
            clip_on=True,
        )
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_cluster_sizes(
    axis: plt.Axes,
    columns: Sequence[object],
    cluster_sizes: pd.Series,
    *,
    config: ComplexHeatmapConfig,
) -> None:
    sizes = pd.to_numeric(
        cluster_sizes.reindex([str(column) for column in columns]),
        errors="coerce",
    ).fillna(0.0)
    if (
        not config.show_cluster_size_annotation
        or sizes.empty
        or float(sizes.max()) <= 0
    ):
        axis.axis("off")
        return

    axis.bar(np.arange(len(columns)), sizes.to_numpy(dtype=float), width=0.82)
    axis.set_xlim(-0.5, len(columns) - 0.5)
    axis.set_xticks([])
    axis.tick_params(axis="y", labelsize=6, length=2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.set_ylabel("N", fontsize=7, rotation=0, labelpad=7, va="center")


def _draw_primary_strip(
    axis: plt.Axes,
    *,
    row_split: pd.Series,
    row_split_order: list[str] | None,
    row_index: pd.Index,
    row_labels: Sequence[str],
    config: ComplexHeatmapConfig,
    row_boundaries: Sequence[float],
) -> tuple[list[str], dict[str, str]]:
    categories, colors = _primary_annotation_spec(
        row_split,
        row_split_order,
        config=config,
    )
    values = row_split.reindex(row_index).astype(str)
    category_to_code = {category: index for index, category in enumerate(categories)}
    codes = values.map(category_to_code).fillna(-1).to_numpy(dtype=int)[:, None]

    listed_colors = [colors[category] for category in categories]
    listed_colors.append("white")
    cmap = matplotlib.colors.ListedColormap(listed_colors)
    shown_codes = codes.copy()
    shown_codes[shown_codes < 0] = len(categories)

    axis.imshow(
        shown_codes,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(listed_colors) - 0.5,
    )
    axis.set_xticks([])
    axis.set_yticks(np.arange(len(row_index)))
    axis.set_yticklabels(
        list(row_labels),
        fontsize=config.row_label_fontsize,
        rotation=0,
    )
    axis.tick_params(axis="y", pad=5, length=0)
    for tick in axis.get_yticklabels():
        tick.set_horizontalalignment("right")
        tick.set_clip_on(False)

    axis.set_ylim(len(row_index) - 0.5, -0.5)
    for boundary in row_boundaries:
        axis.axhline(boundary, color="white", linewidth=2.2, zorder=5)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return categories, colors


def _legend_layout(
    categories: Sequence[str],
    config: ComplexHeatmapConfig,
) -> tuple[int, int, float]:
    if not categories:
        return 1, 0, 0.0
    ncol = min(max(int(config.primary_row_annotation_legend_max_columns), 1), len(categories))
    nrows = int(np.ceil(len(categories) / ncol))
    height_inches = 0.48 + 0.38 * nrows
    return ncol, nrows, height_inches


def _draw_primary_legend_axis(
    axis: plt.Axes,
    *,
    categories: Sequence[str],
    colors: Mapping[str, str],
    config: ComplexHeatmapConfig,
) -> None:
    axis.axis("off")
    if not categories or not config.primary_row_annotation_legend:
        return
    ncol, _, _ = _legend_layout(categories, config)
    wrap_width = max(int(config.primary_row_annotation_legend_wrap_width), 12)
    handles = [
        Patch(
            facecolor=colors[category],
            edgecolor="none",
            label=textwrap.fill(str(category), width=wrap_width),
        )
        for category in categories
    ]
    legend = axis.legend(
        handles=handles,
        title=translated_text("primary_section", config.language),
        loc="upper center",
        frameon=False,
        ncol=ncol,
        borderaxespad=0.0,
        handlelength=1.25,
        handleheight=0.9,
        handletextpad=0.5,
        columnspacing=4.0,
        labelspacing=0.55,
        fontsize=config.primary_row_annotation_legend_fontsize,
    )
    legend.get_title().set_fontsize(config.primary_row_annotation_legend_title_fontsize)
    legend.get_title().set_fontweight("bold")


def _plot_pair(
    data: dict[str, Any],
    config: ComplexHeatmapConfig,
) -> tuple[plt.Figure, dict[str, float]]:
    positive = _validate_numeric_heatmap_matrix(
        data["positive"],
        matrix_name=f"{data['level']}_positive_plot",
    )
    negative = _validate_numeric_heatmap_matrix(
        data["negative"].loc[positive.index, positive.columns],
        matrix_name=f"{data['level']}_negative_plot",
    )

    positive_limit = _robust_positive_limit(positive, config.robust_percentile)
    negative_limit = _robust_positive_limit(negative, config.robust_percentile)
    positive_norm = Normalize(vmin=0.0, vmax=positive_limit)
    negative_norm = Normalize(vmin=0.0, vmax=negative_limit)

    row_labels = (
        data.get("row_metadata", pd.DataFrame())
        .reindex(positive.index)
        .get("display_group", pd.Series(index=positive.index, data=positive.index))
        .astype(str)
        .tolist()
    )
    label_width = _estimate_row_label_width_inches(
        row_labels,
        fontsize=config.row_label_fontsize,
    )
    has_strip = bool(
        data["level"] == "secondary"
        and config.show_primary_row_annotation
        and data.get("row_split") is not None
        and not data["row_split"].empty
    )
    row_boundaries = _row_group_boundaries(data.get("row_split"), positive.index)

    categories: list[str] = []
    category_colors: dict[str, str] = {}
    if has_strip:
        categories, category_colors = _primary_annotation_spec(
            data["row_split"],
            data.get("row_split_order"),
            config=config,
        )
    show_legend = bool(has_strip and config.primary_row_annotation_legend)
    _, _, legend_height = _legend_layout(categories if show_legend else [], config)

    row_count = max(len(positive.index), 1)
    heatmap_height = min(
        max(config.minimum_figure_height, row_count * config.row_height_inches + 1.8),
        config.maximum_figure_height,
    )
    figure_height = heatmap_height + legend_height
    figure_width = 2 * config.panel_width_inches + label_width + 2.0

    figure = plt.figure(figsize=(figure_width, figure_height))
    top_margin = 0.93 if config.show_figure_title else 0.98
    bottom_margin = max(0.055, (legend_height + 0.16) / figure_height if show_legend else 0.065)
    left_margin = min(0.42, max(0.10, (label_width + 0.22) / figure_width))
    right_margin = 0.955

    numeric_cluster_sizes = pd.to_numeric(data["cluster_sizes"], errors="coerce").fillna(0.0)
    has_cluster_sizes = bool(
        config.show_cluster_size_annotation
        and not numeric_cluster_sizes.empty
        and float(numeric_cluster_sizes.max()) > 0
    )
    size_row_ratio = (
        max(float(config.cluster_size_height_ratio), 0.001)
        if has_cluster_sizes else 0.001
    )
    strip_row_ratio = (
        max(float(config.cluster_strip_height_ratio), 0.001)
        if config.show_cluster_strip_annotation else 0.001
    )

    grid = figure.add_gridspec(
        nrows=4 if show_legend else 3,
        ncols=4,
        left=left_margin,
        right=right_margin,
        top=top_margin,
        bottom=0.045,
        width_ratios=[0.055 if has_strip else 0.001, 1.0, 1.0, 0.075],
        height_ratios=(
            [size_row_ratio, strip_row_ratio, 1.0, legend_height / max(heatmap_height, 1.0)]
            if show_legend
            else [size_row_ratio, strip_row_ratio, 1.0]
        ),
        hspace=0.05,
        wspace=0.10,
    )

    size_pos = figure.add_subplot(grid[0, 1])
    size_neg = figure.add_subplot(grid[0, 2])
    _draw_cluster_sizes(size_pos, positive.columns, data["cluster_sizes"], config=config)
    _draw_cluster_sizes(size_neg, negative.columns, data["cluster_sizes"], config=config)

    strip_pos = figure.add_subplot(grid[1, 1])
    strip_neg = figure.add_subplot(grid[1, 2])
    _draw_cluster_strip(strip_pos, positive.columns, config=config)
    _draw_cluster_strip(strip_neg, negative.columns, config=config)

    positive_ax = figure.add_subplot(grid[2, 1])
    negative_ax = figure.add_subplot(grid[2, 2])

    if has_strip:
        primary_ax = figure.add_subplot(grid[2, 0])
        categories, category_colors = _draw_primary_strip(
            primary_ax,
            row_split=data["row_split"],
            row_split_order=data.get("row_split_order"),
            row_index=positive.index,
            row_labels=row_labels,
            config=config,
            row_boundaries=row_boundaries,
        )
        show_ylabels_on_heatmap = False
    else:
        show_ylabels_on_heatmap = True

    positive_image = _draw_heatmap_axis(
        positive_ax,
        positive,
        cmap=config.positive_cmap,
        norm=positive_norm,
        config=config,
        show_ylabels=show_ylabels_on_heatmap,
        ylabels=row_labels,
        title=translated_text("positive_panel", config.language),
        row_boundaries=row_boundaries,
    )
    negative_image = _draw_heatmap_axis(
        negative_ax,
        negative,
        cmap=config.negative_cmap,
        norm=negative_norm,
        config=config,
        show_ylabels=False,
        ylabels=row_labels,
        title=translated_text("negative_panel", config.language),
        row_boundaries=row_boundaries,
    )

    color_grid = grid[2, 3].subgridspec(2, 1, hspace=0.50)
    positive_cax = figure.add_subplot(color_grid[0, 0])
    negative_cax = figure.add_subplot(color_grid[1, 0])
    positive_cb = figure.colorbar(positive_image, cax=positive_cax)
    negative_cb = figure.colorbar(negative_image, cax=negative_cax)
    positive_cb.set_label(
        translated_text("positive_colorbar", config.language),
        fontsize=8,
        labelpad=6,
    )
    negative_cb.set_label(
        translated_text("negative_colorbar", config.language),
        fontsize=8,
        labelpad=6,
    )
    positive_cb.ax.tick_params(labelsize=7)
    negative_cb.ax.tick_params(labelsize=7)

    if show_legend:
        legend_ax = figure.add_subplot(grid[3, :])
        _draw_primary_legend_axis(
            legend_ax,
            categories=categories,
            colors=category_colors,
            config=config,
        )

    if config.show_figure_title:
        title_key = "primary_title" if data["level"] == "primary" else "secondary_title"
        figure.suptitle(
            translated_text(title_key, config.language),
            fontsize=config.title_fontsize,
            fontweight="bold",
            y=0.985,
        )

    return figure, {
        "positive_vmax": positive_limit,
        "negative_vmax": negative_limit,
    }


def _plot_net(
    data: dict[str, Any],
    config: ComplexHeatmapConfig,
) -> tuple[plt.Figure, dict[str, float]]:
    net = _validate_numeric_heatmap_matrix(
        data["net"].loc[data["positive"].index, data["positive"].columns],
        matrix_name=f"{data['level']}_net_plot",
    )
    limit = _robust_symmetric_limit(net, config.robust_percentile)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    row_labels = (
        data.get("row_metadata", pd.DataFrame())
        .reindex(net.index)
        .get("display_group", pd.Series(index=net.index, data=net.index))
        .astype(str)
        .tolist()
    )
    label_width = _estimate_row_label_width_inches(
        row_labels,
        fontsize=config.row_label_fontsize,
    )
    has_strip = bool(
        data["level"] == "secondary"
        and config.show_primary_row_annotation
        and data.get("row_split") is not None
        and not data["row_split"].empty
    )
    row_boundaries = _row_group_boundaries(data.get("row_split"), net.index)

    categories: list[str] = []
    category_colors: dict[str, str] = {}
    if has_strip:
        categories, category_colors = _primary_annotation_spec(
            data["row_split"],
            data.get("row_split_order"),
            config=config,
        )
    show_legend = bool(has_strip and config.primary_row_annotation_legend)
    _, _, legend_height = _legend_layout(categories if show_legend else [], config)

    row_count = max(len(net.index), 1)
    heatmap_height = min(
        max(config.minimum_figure_height, row_count * config.row_height_inches + 1.8),
        config.maximum_figure_height,
    )
    figure_height = heatmap_height + legend_height
    figure_width = config.panel_width_inches + label_width + 2.0

    figure = plt.figure(figsize=(figure_width, figure_height))
    left_margin = min(0.50, max(0.12, (label_width + 0.22) / figure_width))
    numeric_cluster_sizes = pd.to_numeric(data["cluster_sizes"], errors="coerce").fillna(0.0)
    has_cluster_sizes = bool(
        config.show_cluster_size_annotation
        and not numeric_cluster_sizes.empty
        and float(numeric_cluster_sizes.max()) > 0
    )
    size_row_ratio = (
        max(float(config.cluster_size_height_ratio), 0.001)
        if has_cluster_sizes else 0.001
    )
    strip_row_ratio = (
        max(float(config.cluster_strip_height_ratio), 0.001)
        if config.show_cluster_strip_annotation else 0.001
    )
    grid = figure.add_gridspec(
        nrows=4 if show_legend else 3,
        ncols=3,
        left=left_margin,
        right=0.95,
        top=0.93 if config.show_figure_title else 0.98,
        bottom=0.045,
        width_ratios=[0.055 if has_strip else 0.001, 1.0, 0.085],
        height_ratios=(
            [size_row_ratio, strip_row_ratio, 1.0, legend_height / max(heatmap_height, 1.0)]
            if show_legend
            else [size_row_ratio, strip_row_ratio, 1.0]
        ),
        hspace=0.05,
        wspace=0.10,
    )

    size_ax = figure.add_subplot(grid[0, 1])
    _draw_cluster_sizes(size_ax, net.columns, data["cluster_sizes"], config=config)
    cluster_ax = figure.add_subplot(grid[1, 1])
    _draw_cluster_strip(cluster_ax, net.columns, config=config)

    heatmap_ax = figure.add_subplot(grid[2, 1])
    if has_strip:
        primary_ax = figure.add_subplot(grid[2, 0])
        categories, category_colors = _draw_primary_strip(
            primary_ax,
            row_split=data["row_split"],
            row_split_order=data.get("row_split_order"),
            row_index=net.index,
            row_labels=row_labels,
            config=config,
            row_boundaries=row_boundaries,
        )
        show_ylabels_on_heatmap = False
    else:
        show_ylabels_on_heatmap = True

    image = _draw_heatmap_axis(
        heatmap_ax,
        net,
        cmap=config.net_cmap,
        norm=norm,
        config=config,
        show_ylabels=show_ylabels_on_heatmap,
        ylabels=row_labels,
        title=translated_text("net_panel", config.language),
        row_boundaries=row_boundaries,
    )

    color_grid = grid[2, 2].subgridspec(2, 1, height_ratios=[0.4, 0.6])
    colorbar_ax = figure.add_subplot(color_grid[0, 0])
    colorbar = figure.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(
        translated_text("net_colorbar", config.language),
        fontsize=8,
        labelpad=6,
    )
    colorbar.ax.tick_params(labelsize=7)

    if show_legend:
        legend_ax = figure.add_subplot(grid[3, :])
        _draw_primary_legend_axis(
            legend_ax,
            categories=categories,
            colors=category_colors,
            config=config,
        )

    if config.show_figure_title:
        title_key = (
            "primary_net_title" if data["level"] == "primary" else "secondary_net_title"
        )
        figure.suptitle(
            translated_text(title_key, config.language),
            fontsize=config.title_fontsize,
            fontweight="bold",
            y=0.985,
        )

    return figure, {"net_vmin": -limit, "net_vmax": limit}

# -------------------------------------------------------------------------
# Public function
# -------------------------------------------------------------------------

def generate_weighted_complex_heatmaps(
    summary: pd.DataFrame,
    hierarchy_structure: pd.DataFrame | None,
    output_dir: str | Path,
    config: ComplexHeatmapConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate weighted NB+, NB- and net PyComplexHeatmap figures."""
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        raise ValueError("summary must be a non-empty pandas DataFrame.")

    config = _normalize_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = config.font_family
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    compatibility = install_matplotlib_pycomplexheatmap_compatibility()

    figure_paths: dict[str, dict[str, Path]] = {}
    matrix_paths: dict[str, Path] = {}
    scales: dict[str, dict[str, float]] = {}
    prepared_data: dict[str, dict[str, Any]] = {}

    for level in ("primary", "secondary"):
        data = _prepare_level_data(
            summary,
            hierarchy_structure,
            level=level,
            language=config.language,
        )
        if data is None:
            continue

        data = _apply_shared_similarity_order(data, config)

        # Final validation is deliberately performed before any CSV or figure
        # is written. Therefore, an all-NaN output can no longer be produced.
        for matrix_name in ("positive", "negative", "net"):
            data[matrix_name] = _validate_numeric_heatmap_matrix(
                data[matrix_name],
                matrix_name=f"{level}_{matrix_name}_before_save",
            )

        prepared_data[level] = data

        for matrix_name in ("positive", "negative", "net"):
            path = output_dir / f"{level}_{matrix_name}_weighted_evidence_matrix.csv"
            data[matrix_name].to_csv(path)
            matrix_paths[f"{level}_{matrix_name}"] = path

        row_path = output_dir / f"{level}_row_metadata.csv"
        data["row_metadata"].to_csv(row_path)
        matrix_paths[f"{level}_rows"] = row_path

        column_path = output_dir / f"{level}_column_metadata.csv"
        data["column_metadata"].to_csv(column_path)
        matrix_paths[f"{level}_columns"] = column_path

        pair_figure, pair_scales = _plot_pair(data, config)
        pair_key = f"{level}_positive_negative"
        figure_paths[pair_key] = _save_figure_bundle(
            pair_figure,
            output_dir / pair_key,
            dpi=config.publication_dpi,
        )
        scales[pair_key] = pair_scales
        plt.close(pair_figure)

        if config.generate_net_heatmap:
            net_figure, net_scales = _plot_net(data, config)
            net_key = f"{level}_net"
            figure_paths[net_key] = _save_figure_bundle(
                net_figure,
                output_dir / net_key,
                dpi=config.publication_dpi,
            )
            scales[net_key] = net_scales
            plt.close(net_figure)

    if not figure_paths:
        raise ValueError("No primary or secondary hierarchy rows were found in summary.")

    audit = {
        "module_version": MODULE_VERSION,
        "config": asdict(config),
        "compatibility": compatibility,
        "shared_similarity_cluster_order": {
            level: data.get("similarity_column_order", data["display_columns"])
            for level, data in prepared_data.items()
        },
        "shared_similarity_row_order": {
            level: data.get("similarity_row_order", data["row_metadata"].index.tolist())
            for level, data in prepared_data.items()
        },
        "numeric_matrix_checks": {
            f"{level}_{name}": {
                "shape": list(data[name].shape),
                "finite_cells": int(np.isfinite(data[name].to_numpy(dtype=float)).sum()),
                "nan_cells": int(np.isnan(data[name].to_numpy(dtype=float)).sum()),
            }
            for level, data in prepared_data.items()
            for name in ("positive", "negative", "net")
        },
        "secondary_display_labels": (
            prepared_data.get("secondary", {})
            .get("row_metadata", pd.DataFrame())
            .get("display_group", pd.Series(dtype=object))
            .astype(str)
            .tolist()
        ),
        "secondary_primary_sections": (
            prepared_data.get("secondary", {})
            .get("row_metadata", pd.DataFrame())
            .get("display_parent_group", pd.Series(dtype=object))
            .astype(str)
            .drop_duplicates()
            .tolist()
        ),
        "scales": scales,
    }
    audit_path = output_dir / "complex_heatmap_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "output_dir": output_dir,
        "figure_paths": figure_paths,
        "matrix_paths": matrix_paths,
        "scales": scales,
        "audit_path": audit_path,
        "compatibility": compatibility,
        "module_version": MODULE_VERSION,
        "figures": figure_paths,
        "matrices": matrix_paths,
    }


__all__ = [
    "MODULE_VERSION",
    "ComplexHeatmapConfig",
    "generate_weighted_complex_heatmaps",
    "install_matplotlib_pycomplexheatmap_compatibility",
    "ensure_matplotlib_pycomplexheatmap_compatibility",
]
