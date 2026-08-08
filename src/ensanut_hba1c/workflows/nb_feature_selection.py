"""Select the model predictors from the exact NB-filtered heatmap variable set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .nb_evidence import NBEvidenceBundle


@dataclass
class NBFeatureSelectionResult:
    """Final predictor matrix and a complete selection audit."""

    X: pd.DataFrame
    selected_variables: list[str]
    nb_approved_variables: list[str]
    missing_after_model_preparation: list[str]
    excluded_for_quality: list[str]
    audit: pd.DataFrame

    @property
    def n_selected(self) -> int:
        return len(self.selected_variables)

    def export(self, output_dir: str | Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_path = output_dir / "nb_xgboost_feature_selection_audit.csv"
        selected_path = output_dir / "nb_xgboost_selected_variables.csv"
        self.audit.to_csv(audit_path, index=False)
        pd.DataFrame(
            {
                "model_order": np.arange(1, len(self.selected_variables) + 1),
                "var": self.selected_variables,
            }
        ).to_csv(selected_path, index=False)
        return {"audit": audit_path, "selected_variables": selected_path}


def select_nb_features(
    prepared_matrix: pd.DataFrame,
    evidence: NBEvidenceBundle,
    *,
    forbidden_predictors: Iterable[str] = (),
    max_features: int | None = None,
) -> NBFeatureSelectionResult:
    """Restrict a prepared model matrix to the NB-supported heatmap variables.

    By default, ``max_features=None`` keeps every variable approved by the NB
    heatmap filter. In the current project this is expected to be roughly
    500–550 variables before availability and quality checks. No importance-
    based top-k selection is applied.
    """

    if not isinstance(prepared_matrix, pd.DataFrame):
        prepared_matrix = pd.DataFrame(prepared_matrix)
    prepared_matrix = prepared_matrix.copy()
    prepared_matrix.columns = prepared_matrix.columns.map(str)

    forbidden = {str(variable) for variable in forbidden_predictors}
    approved_order = [
        str(variable)
        for variable in evidence.variable_order
        if str(variable) not in forbidden
    ]
    approved_set = set(approved_order)

    # Defensive second pass: add any filtered variable that was not present in
    # the retained first-appearance order, while preserving evidence order.
    for variable in evidence.filtered_evidence_df["var"].dropna().astype(str):
        if variable not in forbidden and variable not in approved_set:
            approved_order.append(variable)
            approved_set.add(variable)

    prepared_columns = set(prepared_matrix.columns)
    available = [variable for variable in approved_order if variable in prepared_columns]
    missing_after_prepare = [
        variable for variable in approved_order if variable not in prepared_columns
    ]
    if not available:
        raise ValueError(
            "No NB-approved heatmap variable is available in the prepared model matrix."
        )

    quality_rows: list[dict[str, object]] = []
    quality_passed: list[str] = []
    excluded_for_quality: list[str] = []
    for variable in available:
        series = prepared_matrix[variable]
        n_nonmissing = int(series.notna().sum())
        n_unique = int(series.nunique(dropna=True))
        exclusion_reason = ""
        if n_nonmissing == 0:
            exclusion_reason = "all_missing"
        elif n_unique <= 1:
            exclusion_reason = "constant"
        else:
            quality_passed.append(variable)
        if exclusion_reason:
            excluded_for_quality.append(variable)
        quality_rows.append(
            {
                "var": variable,
                "available_after_prepare_model_matrix": True,
                "n_nonmissing": n_nonmissing,
                "missing_fraction": float(series.isna().mean()),
                "n_unique_nonmissing": n_unique,
                "quality_exclusion_reason": exclusion_reason,
            }
        )

    if max_features is not None:
        if int(max_features) <= 0:
            raise ValueError("max_features must be positive or None.")
        quality_passed = quality_passed[: int(max_features)]

    selected_variables = quality_passed
    if not selected_variables:
        raise ValueError("All NB-approved variables are empty or constant.")

    X = prepared_matrix.loc[:, selected_variables].copy()
    prohibited_inside_x = sorted(set(X.columns).intersection(forbidden))
    if prohibited_inside_x:
        raise AssertionError(
            "Forbidden predictors were detected in the final model matrix: "
            + ", ".join(prohibited_inside_x)
        )
    if not set(X.columns).issubset(approved_set):
        raise AssertionError(
            "The final model matrix contains predictors that did not pass the NB filter."
        )

    evidence_audit = evidence.variable_audit.copy()
    missing_rows = pd.DataFrame(
        {
            "var": missing_after_prepare,
            "available_after_prepare_model_matrix": False,
            "n_nonmissing": np.nan,
            "missing_fraction": np.nan,
            "n_unique_nonmissing": np.nan,
            "quality_exclusion_reason": "not_available_after_prepare_model_matrix",
        }
    )
    quality_df = pd.concat(
        [pd.DataFrame(quality_rows), missing_rows], ignore_index=True
    )
    audit = evidence_audit.merge(quality_df, on="var", how="left")
    audit["used_by_xgboost"] = audit["var"].isin(selected_variables)
    audit["model_order"] = audit["var"].map(
        {variable: index + 1 for index, variable in enumerate(selected_variables)}
    )
    audit["maximum_feature_limit"] = max_features
    audit = audit.sort_values("heatmap_order", kind="stable").reset_index(drop=True)

    return NBFeatureSelectionResult(
        X=X,
        selected_variables=selected_variables,
        nb_approved_variables=approved_order,
        missing_after_model_preparation=missing_after_prepare,
        excluded_for_quality=excluded_for_quality,
        audit=audit,
    )
