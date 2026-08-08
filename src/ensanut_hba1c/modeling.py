"""Reusable modeling functions for the ENSANUT HbA1c notebook.

Scientific choices and tunable parameters are supplied by the notebook. This
module contains the longer implementation details for the regression matrix,
out-of-fold XGBoost evaluation, SHAP calculation and XGBoost evaluation. Dimensionality reduction and clustering live in dedicated modules.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def _deduplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of duplicated column names."""
    return frame.loc[:, ~frame.columns.duplicated()].copy()


def prepare_model_matrix(
    df: pd.DataFrame,
    target_col: str,
    weight_col: str | None = None,
    id_columns: Iterable[str] | None = None,
    exclude_columns: Iterable[str] | None = None,
    keep_high_cardinality: Iterable[str] | None = None,
    max_categorical_levels: int = 100,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, dict[str, Any], pd.DataFrame]:
    """Create the numeric one-hot matrix used by XGBoost.

    Returns
    -------
    X, y, weights, audit, cohort
        ``cohort`` is the target-valid dataframe in the exact row order used by
        the model and downstream SHAP analysis.
    """

    if target_col not in df.columns:
        raise KeyError(f"Target does not exist: {target_col}")

    frame = _deduplicate_columns(df)
    id_columns = list(id_columns or [])
    exclude_columns = list(exclude_columns or [])
    keep_high_cardinality = set(keep_high_cardinality or [])

    y_all = pd.to_numeric(frame[target_col], errors="coerce")
    valid_target = y_all.notna() & np.isfinite(y_all)
    cohort = frame.loc[valid_target].reset_index(drop=True)
    y = y_all.loc[valid_target].reset_index(drop=True).astype(np.float32)

    drop_cols = set(id_columns + exclude_columns + [target_col])
    if weight_col:
        drop_cols.add(weight_col)

    X_raw = cohort.drop(
        columns=[column for column in drop_cols if column in cohort.columns]
    ).copy()

    weights: pd.Series | None = None
    if weight_col:
        if weight_col not in cohort.columns:
            raise KeyError(f"Sampling-weight column does not exist: {weight_col}")
        weights = pd.to_numeric(cohort[weight_col], errors="coerce")
        if weights.notna().sum() == 0:
            raise ValueError(
                f"Sampling-weight column {weight_col} does not contain valid values."
            )
        weights = weights.replace([np.inf, -np.inf], np.nan)
        weights = weights.fillna(weights.median()).astype(np.float32)

    empty_or_constant = [
        column
        for column in X_raw.columns
        if X_raw[column].notna().sum() == 0
        or X_raw[column].nunique(dropna=True) <= 1
    ]
    X_raw = X_raw.drop(columns=empty_or_constant)

    numeric_cols = X_raw.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [column for column in X_raw.columns if column not in numeric_cols]

    high_cardinality = [
        column
        for column in categorical_cols
        if X_raw[column].nunique(dropna=True) > max_categorical_levels
        and column not in keep_high_cardinality
    ]
    X_raw = X_raw.drop(columns=high_cardinality)

    numeric_cols = X_raw.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [column for column in X_raw.columns if column not in numeric_cols]

    X_num = X_raw[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if categorical_cols:
        X_cat = X_raw[categorical_cols].astype("string").fillna("MISSING")
        X_cat = pd.get_dummies(X_cat, drop_first=False, dtype=np.uint8)
    else:
        X_cat = pd.DataFrame(index=X_raw.index)

    X = pd.concat([X_num, X_cat], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    if X.shape[1] == 0:
        raise ValueError("The modeling matrix has no predictors.")

    audit = {
        "empty_or_constant": empty_or_constant,
        "high_cardinality": high_cardinality,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "n_rows": int(X.shape[0]),
        "n_predictors": int(X.shape[1]),
        "target": target_col,
        "weight_column": weight_col,
    }
    return X, y, weights, audit, cohort


def make_xgb_regressor(
    params: dict[str, Any],
    random_state: int,
    device: str = "cpu",
    n_jobs: int = 4,
):
    """Build an XGBRegressor while handling XGBoost device-version changes."""

    import xgboost as xgb

    major_version = int(xgb.__version__.split(".")[0])
    common: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": 0,
        **params,
    }
    if major_version >= 2:
        common["device"] = device
    elif device == "cuda":
        common["tree_method"] = "gpu_hist"
    return xgb.XGBRegressor(**common)


def _safe_spearman(y_true, y_pred) -> tuple[float, float]:
    correlation, p_value = spearmanr(y_true, y_pred)
    if np.isnan(correlation):
        return np.nan, np.nan
    return float(correlation), float(p_value)


def cross_validate_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    weights: pd.Series | None,
    n_splits: int,
    random_state: int,
    device: str = "cpu",
    n_jobs: int = 4,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Generate out-of-fold predictions and fold/global regression metrics."""

    if len(y) < n_splits:
        raise ValueError(
            f"There are {len(y)} observations, fewer than n_splits={n_splits}."
        )

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=float)
    rows: list[dict[str, float]] = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X), start=1):
        model = make_xgb_regressor(params, random_state, device, n_jobs)
        fit_kwargs: dict[str, Any] = {}
        if weights is not None:
            fit_kwargs["sample_weight"] = weights.iloc[train_idx].to_numpy()
        model.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)
        prediction = np.asarray(model.predict(X.iloc[valid_idx]), dtype=float)
        oof[valid_idx] = prediction
        correlation, p_value = _safe_spearman(y.iloc[valid_idx], prediction)
        rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "R2": r2_score(y.iloc[valid_idx], prediction),
                "MAE": mean_absolute_error(y.iloc[valid_idx], prediction),
                "RMSE": mean_squared_error(y.iloc[valid_idx], prediction) ** 0.5,
                "Spearman": correlation,
                "Spearman_p": p_value,
            }
        )

    fold_metrics = pd.DataFrame(rows)
    correlation, p_value = _safe_spearman(y, oof)
    global_metrics = pd.DataFrame(
        [
            {
                "R2_OOF": r2_score(y, oof),
                "MAE_OOF": mean_absolute_error(y, oof),
                "RMSE_OOF": mean_squared_error(y, oof) ** 0.5,
                "Spearman_OOF": correlation,
                "Spearman_p_OOF": p_value,
                "n": len(y),
            }
        ]
    )
    return oof, fold_metrics, global_metrics


def plot_oof_predictions(
    y: pd.Series,
    oof_prediction: np.ndarray,
    output_path: str | Path,
):
    """Create and save the observed-versus-OOF prediction plot."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, oof_prediction, alpha=0.45, s=16)
    limits = [
        min(float(y.min()), float(np.min(oof_prediction))),
        max(float(y.max()), float(np.max(oof_prediction))),
    ]
    ax.plot(limits, limits, linestyle="--", linewidth=1)
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="Observed HbA1c",
        ylabel="Predicted HbA1c (OOF)",
        title="Out-of-fold cross-validation",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def select_shap_rows(
    X: pd.DataFrame,
    cohort: pd.DataFrame,
    max_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Select every row or a reproducible subset for SHAP."""

    if max_rows is None or len(X) <= max_rows:
        return (
            X.reset_index(drop=True),
            cohort.reset_index(drop=True),
            np.arange(len(X)),
        )
    rng = np.random.default_rng(random_state)
    selected = np.sort(rng.choice(len(X), size=int(max_rows), replace=False))
    return (
        X.iloc[selected].reset_index(drop=True),
        cohort.iloc[selected].reset_index(drop=True),
        selected,
    )


def select_original_shap_matrix(shap_exp, top_n: int | None = None):
    """Return the SHAP matrix used for PCA50 and the shared UMAP-graph workflow.

    When ``top_n`` is ``None``, every SHAP column is retained in the exact
    model-feature order. This is the production all-variable mode. A finite
    integer is the only way to request a ranked subset.
    """

    values = np.asarray(shap_exp.values, dtype=float)
    if values.ndim != 2:
        raise ValueError("SHAP values must be a 2D matrix.")
    if not np.isfinite(values).all():
        raise ValueError("SHAP values contain NaN or infinite values.")

    n_features = values.shape[1]
    names = (
        np.array([f"feature_{i}" for i in range(n_features)], dtype=object)
        if shap_exp.feature_names is None
        else np.asarray(shap_exp.feature_names, dtype=object)
    )
    importance = np.mean(np.abs(values), axis=0)
    ranked_idx = np.argsort(-importance, kind="stable")
    rank_by_column = np.empty(n_features, dtype=int)
    rank_by_column[ranked_idx] = np.arange(1, n_features + 1)

    if top_n is None:
        selected_idx = np.arange(n_features, dtype=int)
        selection_mode = "all_model_features_in_original_order"
    else:
        requested = min(max(int(top_n), 1), n_features)
        selected_idx = ranked_idx[:requested]
        selection_mode = f"top_{requested}_by_mean_abs_shap"

    ranking = pd.DataFrame(
        {
            "matrix_column_order": np.arange(1, len(selected_idx) + 1),
            "global_importance_rank": rank_by_column[selected_idx],
            "feature": names[selected_idx],
            "mean_abs_shap": importance[selected_idx],
            "selection_mode": selection_mode,
        }
    )
    return values[:, selected_idx], names[selected_idx], selected_idx, ranking

def build_normalized_shap_explanation(shap_exp, top_n: int):
    """Scale each selected SHAP feature independently to [-1, 1]."""

    import shap

    values = np.asarray(shap_exp.values, dtype=float)
    importance = np.mean(np.abs(values), axis=0)
    selected = np.argsort(-importance, kind="stable")[: min(top_n, values.shape[1])]
    selected_values = values[:, selected]
    scale = np.max(np.abs(selected_values), axis=0)
    scale[scale == 0] = 1.0
    normalized = selected_values / scale
    feature_names = np.asarray(shap_exp.feature_names)[selected]
    data = None
    if getattr(shap_exp, "data", None) is not None:
        data = np.asarray(shap_exp.data)[:, selected]
    explanation = shap.Explanation(
        values=normalized,
        base_values=getattr(shap_exp, "base_values", None),
        data=data,
        feature_names=feature_names.tolist(),
    )
    ranking = pd.DataFrame(
        {
            "ranking": np.arange(1, len(selected) + 1),
            "feature": feature_names,
            "mean_abs_original_shap": importance[selected],
            "normalization_max_abs": scale,
        }
    )
    return explanation, ranking


def fit_final_xgb_and_shap(
    X: pd.DataFrame,
    y: pd.Series,
    cohort: pd.DataFrame,
    params: dict[str, Any],
    weights: pd.Series | None,
    output_dir: str | Path,
    *,
    max_rows: int | None = None,
    random_state: int = 42,
    device: str = "cpu",
    n_jobs: int = 4,
    max_display: int = 20,
    normalized_top_n: int = 20,
) -> dict[str, Any]:
    """Fit the final XGBoost model, calculate SHAP and export core outputs."""

    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_xgb_regressor(params, random_state, device, n_jobs)
    fit_kwargs: dict[str, Any] = {}
    if weights is not None:
        fit_kwargs["sample_weight"] = weights.to_numpy()
    model.fit(X, y, **fit_kwargs)

    X_shap, cohort_shap, source_rows = select_shap_rows(
        X, cohort, max_rows=max_rows, random_state=random_state
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)
    shap_array = np.asarray(shap_values.values, dtype=float)
    if shap_array.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape for regression: {shap_array.shape}")

    importance = (
        pd.DataFrame(
            {
                "feature": X_shap.columns,
                "mean_abs_shap": np.nanmean(np.abs(shap_array), axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.insert(0, "ranking", np.arange(1, len(importance) + 1))

    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.xlabel("Mean absolute SHAP importance")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_global_importance.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "shap_global_importance.pdf", bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.xlabel("Original SHAP value")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "shap_beeswarm.pdf", bbox_inches="tight")
    plt.close()

    normalized_values, normalized_ranking = build_normalized_shap_explanation(
        shap_values, normalized_top_n
    )
    shap.plots.beeswarm(
        normalized_values,
        max_display=normalized_values.values.shape[1],
        order=np.arange(normalized_values.values.shape[1]),
        show=False,
    )
    plt.xlabel("Normalized SHAP value [-1, +1]")
    plt.xlim(-1.05, 1.05)
    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_beeswarm_normalized_minus1_plus1.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        output_dir / "shap_beeswarm_normalized_minus1_plus1.pdf",
        bbox_inches="tight",
    )
    plt.close()

    with open(output_dir / "xgboost_model.pkl", "wb") as file:
        pickle.dump(model, file)
    with open(output_dir / "shap_values.pkl", "wb") as file:
        pickle.dump(shap_values, file)
    importance.to_csv(output_dir / "shap_importance.csv", index=False)
    normalized_ranking.to_csv(
        output_dir / "shap_ranking_used_in_normalized_beeswarm.csv", index=False
    )
    pd.DataFrame({"source_row": source_rows}).to_csv(
        output_dir / "shap_rows_used.csv", index=False
    )

    return {
        "final_model": model,
        "X_shap": X_shap,
        "cohort_shap": cohort_shap,
        "shap_source_rows": source_rows,
        "shap_values": shap_values,
        "shap_array": shap_array,
        "shap_importance": importance,
        "normalized_shap_values": normalized_values,
        "shap_normalized_ranking": normalized_ranking,
    }



__all__ = [
    "prepare_model_matrix", "make_xgb_regressor", "cross_validate_xgb",
    "plot_oof_predictions", "fit_final_xgb_and_shap",
    "select_original_shap_matrix", "select_shap_rows",
    "build_normalized_shap_explanation",
]
