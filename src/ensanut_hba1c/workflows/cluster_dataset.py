"""Prepare the participant-level dataset for cluster classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import re

import numpy as np
import pandas as pd

from .nb_evidence import NBEvidenceBundle
from .nb_feature_selection import NBFeatureSelectionResult, select_nb_features


@dataclass(frozen=True)
class ClusterDatasetConfig:
    target_column_for_cohort: str = "HB1AC"
    weight_column: str | None = None
    id_columns: tuple[str, ...] = ("FOLIO_I", "FOLIO_INT")
    cluster_column: str = "Cluster_SHAP"
    max_categorical_levels: int = 100
    max_nb_features: int | None = None


@dataclass
class PreparedClusterDataset:
    X: pd.DataFrame
    y: pd.Series
    sample_weights: np.ndarray
    participant_keys: pd.DataFrame
    cohort_df: pd.DataFrame
    feature_selection: NBFeatureSelectionResult
    model_audit: object
    n_participants_without_cluster: int


def normalize_id_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".")[0]
    return text


def normalize_cluster_label(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        return f"C{int(text)}"
    if re.fullmatch(r"C\d+", text, flags=re.IGNORECASE):
        return f"C{int(text[1:])}"
    return text


def prepare_cluster_dataset(
    ensanut_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    *,
    prepare_model_matrix_fn: Callable,
    evidence: NBEvidenceBundle,
    config: ClusterDatasetConfig | None = None,
    model_exclude_columns: Sequence[str] = (),
    forbidden_predictors: Iterable[str] = (),
) -> PreparedClusterDataset:
    """Build the cohort, link cluster labels by IDs, and apply NB selection."""

    config = config or ClusterDatasetConfig()
    id_columns = list(config.id_columns)

    X_all, _cohort_target, sample_weights, model_audit, cohort_df = (
        prepare_model_matrix_fn(
            ensanut_df,
            target_col=config.target_column_for_cohort,
            weight_col=config.weight_column,
            id_columns=id_columns,
            exclude_columns=list(model_exclude_columns),
            max_categorical_levels=int(config.max_categorical_levels),
        )
    )

    if not isinstance(X_all, pd.DataFrame):
        X_all = pd.DataFrame(X_all)
    X_all = X_all.reset_index(drop=True)
    cohort_df = cohort_df.reset_index(drop=True)

    if len(X_all) != len(cohort_df):
        raise ValueError(
            "prepare_model_matrix returned objects with inconsistent lengths: "
            f"X={len(X_all)}, cohort={len(cohort_df)}."
        )

    if sample_weights is None:
        base_weights = np.ones(len(X_all), dtype=float)
    else:
        base_weights = np.asarray(sample_weights, dtype=float).reshape(-1)
        if len(base_weights) != len(X_all):
            raise ValueError("sample_weights does not match the prepared matrix length.")
        invalid = (~np.isfinite(base_weights)) | (base_weights <= 0)
        base_weights[invalid] = 1.0

    missing_cohort_ids = [column for column in id_columns if column not in cohort_df]
    missing_cluster_ids = [column for column in id_columns if column not in cluster_df]
    if missing_cohort_ids or missing_cluster_ids:
        raise KeyError(
            "Cluster labels cannot be linked. Missing cohort IDs: "
            f"{missing_cohort_ids}; missing cluster IDs: {missing_cluster_ids}."
        )
    if config.cluster_column not in cluster_df:
        raise KeyError(
            f"cluster_df does not contain {config.cluster_column!r}."
        )

    participant_keys = cohort_df[id_columns].copy()
    cluster_lookup = cluster_df[id_columns + [config.cluster_column]].copy()
    for column in id_columns:
        participant_keys[column] = participant_keys[column].map(normalize_id_value)
        cluster_lookup[column] = cluster_lookup[column].map(normalize_id_value)

    duplicated = cluster_lookup.duplicated(id_columns, keep=False)
    if duplicated.any():
        conflicting = (
            cluster_lookup.loc[duplicated]
            .groupby(id_columns, dropna=False)[config.cluster_column]
            .nunique(dropna=True)
        )
        if (conflicting > 1).any():
            raise ValueError("At least one participant ID maps to multiple clusters.")
        cluster_lookup = cluster_lookup.drop_duplicates(id_columns, keep="first")

    linked_target = participant_keys.merge(
        cluster_lookup,
        on=id_columns,
        how="left",
        validate="many_to_one",
        sort=False,
    )[config.cluster_column]

    target_available = linked_target.notna().to_numpy()
    n_without_cluster = int((~target_available).sum())
    X_all = X_all.loc[target_available].reset_index(drop=True)
    cohort_df = cohort_df.loc[target_available].reset_index(drop=True)
    participant_keys = participant_keys.loc[target_available].reset_index(drop=True)
    y = linked_target.loc[target_available].map(normalize_cluster_label).reset_index(drop=True)
    base_weights = base_weights[target_available]

    if y.nunique() < 2:
        raise ValueError("At least two clusters are required for classification.")

    full_forbidden = {str(variable) for variable in forbidden_predictors}
    full_forbidden.update(id_columns)
    full_forbidden.update(
        {config.cluster_column, config.target_column_for_cohort, "_merge"}
    )

    feature_selection = select_nb_features(
        X_all,
        evidence,
        forbidden_predictors=full_forbidden,
        max_features=config.max_nb_features,
    )

    return PreparedClusterDataset(
        X=feature_selection.X.reset_index(drop=True),
        y=y.astype(str),
        sample_weights=base_weights,
        participant_keys=participant_keys,
        cohort_df=cohort_df,
        feature_selection=feature_selection,
        model_audit=model_audit,
        n_participants_without_cluster=n_without_cluster,
    )
