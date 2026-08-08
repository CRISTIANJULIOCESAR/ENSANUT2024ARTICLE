"""Load, validate, filter, and reshape Naive Bayes evidence.

The same filtered evidence object is used by both the final heatmap and the
cluster-classification workflow. This prevents the two analyses from silently
using different predictor sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NBFilterConfig:
    """Thresholds used to define the final NB-supported variable set."""

    min_n_cluster_x: int = 10
    min_score: float = 0.40
    min_coverage: float = 0.07
    sheet_name: str = "NB_internal_operations"


@dataclass
class NBEvidenceBundle:
    """Validated evidence and the four matrices required by the heatmap."""

    source_path: Path
    config: NBFilterConfig
    evidence_df: pd.DataFrame
    filtered_evidence_df: pd.DataFrame
    representative_df: pd.DataFrame
    score_matrix: pd.DataFrame
    coverage_matrix: pd.DataFrame
    count_matrix: pd.DataFrame
    category_matrix: pd.DataFrame
    variable_order: list[str]
    cluster_order: list[str]
    category_source_column: str
    variable_audit: pd.DataFrame

    @property
    def n_variables(self) -> int:
        return len(self.variable_order)

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_order)

    def export(self, output_dir: str | Path) -> dict[str, Path]:
        """Export the filtered evidence, matrices, and variable audit."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        workbook_path = output_dir / "nb_filtered_heatmap_evidence.xlsx"
        variable_csv_path = output_dir / "nb_filtered_variable_list.csv"
        audit_csv_path = output_dir / "nb_filtered_variable_audit.csv"

        with pd.ExcelWriter(workbook_path) as writer:
            self.filtered_evidence_df.to_excel(
                writer, sheet_name="filtered_evidence", index=False
            )
            self.score_matrix.to_excel(writer, sheet_name="score_matrix")
            self.coverage_matrix.to_excel(writer, sheet_name="coverage_matrix")
            self.count_matrix.to_excel(writer, sheet_name="count_matrix")
            self.category_matrix.to_excel(writer, sheet_name="category_matrix")
            self.variable_audit.to_excel(
                writer, sheet_name="variable_audit", index=False
            )

        pd.DataFrame(
            {
                "variable_order": np.arange(1, len(self.variable_order) + 1),
                "var": self.variable_order,
            }
        ).to_csv(variable_csv_path, index=False)
        self.variable_audit.to_csv(audit_csv_path, index=False)

        return {
            "workbook": workbook_path,
            "variable_list": variable_csv_path,
            "variable_audit": audit_csv_path,
        }


def clean_text(value: object) -> str:
    """Convert a value to normalized single-space text."""

    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _first_existing_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _build_variable_audit(
    filtered_evidence_df: pd.DataFrame,
    variable_order: list[str],
) -> pd.DataFrame:
    cluster_column = "cluster_plot" if "cluster_plot" in filtered_evidence_df else "cluster"
    audit = (
        filtered_evidence_df.groupby("var", as_index=False)
        .agg(
            surviving_evidence_rows=("var", "size"),
            clusters_passing=(cluster_column, "nunique"),
            maximum_nb_score=("score", "max"),
            maximum_coverage=("coverage_in_cluster", "max"),
            maximum_n_cluster_x=("n_cluster_x", "max"),
        )
    )
    order_lookup = {variable: index + 1 for index, variable in enumerate(variable_order)}
    audit["heatmap_order"] = audit["var"].map(order_lookup)
    audit["approved_by_nb_filter"] = True
    audit = audit.sort_values("heatmap_order", kind="stable").reset_index(drop=True)
    return audit


def load_nb_evidence(
    excel_path: str | Path,
    *,
    config: NBFilterConfig | None = None,
    prohibited_variables: Iterable[str] = (),
) -> NBEvidenceBundle:
    """Load the NB operations workbook and construct the final filtered matrices.

    Variables survive when at least one cluster-variable evidence row satisfies
    all three thresholds: minimum count, minimum NB score, and minimum coverage.
    The original first-appearance order is retained.
    """

    config = config or NBFilterConfig()
    excel_path = Path(excel_path)
    if not excel_path.is_file():
        raise FileNotFoundError(
            "The NB internal-operations workbook was not found: "
            f"{excel_path}"
        )

    evidence_df = pd.read_excel(excel_path, sheet_name=config.sheet_name)
    required_columns = {"cluster", "var", "n_cluster", "n_cluster_x", "score"}
    missing_columns = sorted(required_columns.difference(evidence_df.columns))
    if missing_columns:
        raise ValueError(
            f"Sheet {config.sheet_name!r} is missing required columns: "
            + ", ".join(missing_columns)
        )

    evidence_df = evidence_df.copy()
    evidence_df["var"] = evidence_df["var"].map(clean_text)
    for column in ("n_cluster", "n_cluster_x", "score"):
        evidence_df[column] = pd.to_numeric(evidence_df[column], errors="coerce")

    evidence_df["_cluster_numeric"] = pd.to_numeric(
        evidence_df["cluster"]
        .astype(str)
        .str.strip()
        .str.replace(r"^[Cc]", "", regex=True),
        errors="coerce",
    )
    evidence_df = evidence_df.dropna(
        subset=["_cluster_numeric", "var", "n_cluster", "n_cluster_x", "score"]
    ).copy()
    evidence_df = evidence_df.loc[evidence_df["var"].ne("")].copy()
    evidence_df["cluster_plot"] = (
        "C" + evidence_df["_cluster_numeric"].astype(int).astype(str)
    )
    evidence_df["coverage_in_cluster"] = (
        evidence_df["n_cluster_x"] / evidence_df["n_cluster"]
    )
    evidence_df = (
        evidence_df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["coverage_in_cluster"])
        .reset_index(drop=True)
    )

    prohibited_set = {str(variable) for variable in prohibited_variables}
    if prohibited_set:
        evidence_df = evidence_df.loc[
            ~evidence_df["var"].astype(str).isin(prohibited_set)
        ].copy()

    category_source_column = _first_existing_column(
        evidence_df,
        ("categoria_o_rango", "category", "categoria", "rango"),
    )
    if category_source_column is None:
        category_source_column = "category"
        evidence_df[category_source_column] = ""

    filtered_evidence_df = evidence_df.loc[
        (evidence_df["n_cluster_x"] >= int(config.min_n_cluster_x))
        & (evidence_df["score"] >= float(config.min_score))
        & (evidence_df["coverage_in_cluster"] >= float(config.min_coverage))
    ].copy()
    if filtered_evidence_df.empty:
        raise ValueError(
            "No evidence row passed the NB count, score, and coverage thresholds."
        )

    cluster_order = list(
        dict.fromkeys(filtered_evidence_df["cluster_plot"].astype(str))
    )
    variable_order = list(dict.fromkeys(filtered_evidence_df["var"].astype(str)))

    representative_df = (
        filtered_evidence_df.sort_values(
            ["score", "coverage_in_cluster", "n_cluster_x"],
            ascending=[False, False, False],
            kind="stable",
        )
        .drop_duplicates(subset=["cluster_plot", "var"], keep="first")
        .reset_index(drop=True)
    )

    def pivot(values: str, fill_value: object) -> pd.DataFrame:
        return (
            representative_df.pivot(
                index="var", columns="cluster_plot", values=values
            )
            .reindex(index=variable_order, columns=cluster_order)
            .fillna(fill_value)
        )

    score_matrix = pivot("score", 0.0).astype(float)
    coverage_matrix = pivot("coverage_in_cluster", 0.0).astype(float)
    count_matrix = pivot("n_cluster_x", 0.0).astype(float)
    category_matrix = pivot(category_source_column, "").astype(object)
    variable_audit = _build_variable_audit(filtered_evidence_df, variable_order)

    return NBEvidenceBundle(
        source_path=excel_path,
        config=config,
        evidence_df=evidence_df.reset_index(drop=True),
        filtered_evidence_df=filtered_evidence_df.reset_index(drop=True),
        representative_df=representative_df,
        score_matrix=score_matrix,
        coverage_matrix=coverage_matrix,
        count_matrix=count_matrix,
        category_matrix=category_matrix,
        variable_order=variable_order,
        cluster_order=cluster_order,
        category_source_column=category_source_column,
        variable_audit=variable_audit,
    )
