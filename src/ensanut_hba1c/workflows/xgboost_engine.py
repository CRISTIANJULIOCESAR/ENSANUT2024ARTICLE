"""Nested cross-validated XGBoost classifier for ENSANUT cluster labels.

The engine produces out-of-fold predictions and retains the diagnostic objects
required by the notebook report: fold stability, learning curves, Optuna
history/parameter importance, global feature importance, and class-specific
SHAP importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from .progress import create_progress_bar


@dataclass(frozen=True)
class XGBoostConfig:
    random_state: int = 42
    outer_cv_splits: int = 5
    inner_cv_splits: int = 2
    run_optuna: bool = True
    trials_per_outer_fold: int = 8
    final_trials: int = 5
    optuna_timeout_seconds: int | None = None
    optuna_study_basename: str = "xgboost_cluster_classifier"
    reuse_existing_studies: bool = True
    optuna_startup_trials: int = 4
    optuna_pruner_startup_trials: int = 3
    optuna_pruner_warmup_steps: int = 1
    objective_metric: str = "macro_f1"
    optuna_n_jobs: int = 1
    max_estimators: int = 1000
    early_stopping_rounds: int = 100
    tree_method: str = "hist"
    n_jobs: int = -1
    use_balanced_class_weights: bool = True
    shap_max_rows_per_fold: int = 300
    calculate_shap_diagnostics: bool = True
    strict_shap_diagnostics: bool = False
    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.20
    max_depth_min: int = 3
    max_depth_max: int = 12
    min_child_weight_min: float = 0.5
    min_child_weight_max: float = 20.0
    subsample_min: float = 0.60
    subsample_max: float = 1.00
    colsample_bytree_min: float = 0.50
    colsample_bytree_max: float = 1.00
    gamma_min: float = 1e-8
    gamma_max: float = 10.0
    reg_alpha_min: float = 1e-8
    reg_alpha_max: float = 10.0
    reg_lambda_min: float = 0.10
    reg_lambda_max: float = 50.0
    max_bin_options: tuple[int, ...] = (128, 256, 512)
    max_cat_to_onehot_options: tuple[int, ...] = (1, 4, 8, 16)
    max_cat_threshold_options: tuple[int, ...] = (16, 32, 64, 128)
    show_progress_bars: bool = True
    progress_leave: bool = True


@dataclass
class XGBoostPipelineResult:
    config: XGBoostConfig
    class_labels: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    oof_probabilities: np.ndarray
    fold_assignment: np.ndarray
    global_metrics: pd.DataFrame
    metrics_by_fold: pd.DataFrame
    metrics_by_cluster: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_importance_by_fold: pd.DataFrame
    shap_importance_by_cluster: pd.DataFrame
    shap_importance_by_fold: pd.DataFrame
    learning_curves: pd.DataFrame
    learning_curve_summary: pd.DataFrame
    optuna_trials: pd.DataFrame
    optuna_parameter_importance: pd.DataFrame
    best_parameters_by_fold: pd.DataFrame
    final_best_parameters: dict[str, Any]
    final_best_value: float | None
    final_n_estimators: int
    final_model: Any
    label_encoder: LabelEncoder


def _prepare_native_fold(
    X_train_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_columns: dict[str, pd.Series] = {}
    valid_columns: dict[str, pd.Series] = {}
    for column in X_train_raw.columns:
        train_series = X_train_raw[column]
        valid_series = X_valid_raw[column]
        if is_numeric_dtype(train_series.dtype):
            train_columns[column] = pd.to_numeric(
                train_series, errors="coerce"
            ).astype("float32")
            valid_columns[column] = pd.to_numeric(
                valid_series, errors="coerce"
            ).astype("float32")
        else:
            train_text = train_series.astype("string")
            valid_text = valid_series.astype("string")
            categories = pd.Index(train_text.dropna().unique()).tolist()
            train_columns[column] = pd.Series(
                pd.Categorical(train_text, categories=categories),
                index=X_train_raw.index,
                name=column,
            )
            valid_columns[column] = pd.Series(
                pd.Categorical(
                    valid_text.where(valid_text.isin(categories)),
                    categories=categories,
                ),
                index=X_valid_raw.index,
                name=column,
            )
    return (
        pd.DataFrame(train_columns, index=X_train_raw.index).copy(),
        pd.DataFrame(valid_columns, index=X_valid_raw.index).copy(),
    )


def _prepare_native_full(X_raw: pd.DataFrame) -> pd.DataFrame:
    native_columns: dict[str, pd.Series] = {}
    for column in X_raw.columns:
        series = X_raw[column]
        if is_numeric_dtype(series.dtype):
            native_columns[column] = pd.to_numeric(series, errors="coerce").astype(
                "float32"
            )
        else:
            text = series.astype("string")
            categories = pd.Index(text.dropna().unique()).tolist()
            native_columns[column] = pd.Series(
                pd.Categorical(text, categories=categories),
                index=X_raw.index,
                name=column,
            )
    return pd.DataFrame(native_columns, index=X_raw.index).copy()


def _training_weights(
    base_weights: np.ndarray,
    y: np.ndarray,
    use_balanced: bool,
) -> np.ndarray:
    weights = np.asarray(base_weights, dtype=float).copy()
    invalid = (~np.isfinite(weights)) | (weights <= 0)
    weights[invalid] = 1.0
    if use_balanced:
        weights *= compute_sample_weight(class_weight="balanced", y=y)
    mean_weight = float(np.mean(weights))
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Training weights could not be normalized.")
    return weights / mean_weight


def _make_classifier(
    config: XGBoostConfig,
    n_classes: int,
    params: dict[str, Any],
    n_estimators: int,
    seed: int,
    early_stopping: bool,
):
    import xgboost as xgb

    kwargs: dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": int(n_classes),
        "n_estimators": int(n_estimators),
        "tree_method": config.tree_method,
        "n_jobs": int(config.n_jobs),
        "random_state": int(seed),
        "eval_metric": "mlogloss",
        "enable_categorical": True,
        "missing": np.nan,
        "verbosity": 0,
    }
    kwargs.update(params)
    if early_stopping:
        kwargs["early_stopping_rounds"] = int(config.early_stopping_rounds)
    return xgb.XGBClassifier(**kwargs)


def _default_parameters() -> dict[str, Any]:
    return {
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "max_cat_to_onehot": 4,
        "max_cat_threshold": 64,
    }


def _score_predictions(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metric = metric.strip().lower()
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError("objective_metric must be 'macro_f1' or 'balanced_accuracy'.")


def _suggest_parameters(trial: Any, config: XGBoostConfig) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            config.learning_rate_min,
            config.learning_rate_max,
            log=True,
        ),
        "max_depth": trial.suggest_int(
            "max_depth", config.max_depth_min, config.max_depth_max
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            config.min_child_weight_min,
            config.min_child_weight_max,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample", config.subsample_min, config.subsample_max
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            config.colsample_bytree_min,
            config.colsample_bytree_max,
        ),
        "gamma": trial.suggest_float(
            "gamma", config.gamma_min, config.gamma_max, log=True
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", config.reg_alpha_min, config.reg_alpha_max, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", config.reg_lambda_min, config.reg_lambda_max, log=True
        ),
        "max_bin": trial.suggest_categorical("max_bin", list(config.max_bin_options)),
        "max_cat_to_onehot": trial.suggest_categorical(
            "max_cat_to_onehot", list(config.max_cat_to_onehot_options)
        ),
        "max_cat_threshold": trial.suggest_categorical(
            "max_cat_threshold", list(config.max_cat_threshold_options)
        ),
    }


def _effective_splits(y: np.ndarray, requested: int) -> int:
    minimum_class_size = int(pd.Series(y).value_counts().min())
    splits = min(int(requested), minimum_class_size)
    if splits < 2:
        raise ValueError("Each class must contain at least two participants.")
    return splits


def _run_optuna_study(
    X: pd.DataFrame,
    y: np.ndarray,
    base_weights: np.ndarray,
    *,
    config: XGBoostConfig,
    storage_path: Path,
    study_name: str,
    n_trials: int,
    seed: int,
    scope: str,
    outer_fold: int | None,
) -> tuple[dict[str, Any], int, pd.DataFrame, pd.DataFrame, float | None]:
    if not config.run_optuna or n_trials <= 0:
        return (
            _default_parameters(),
            min(500, config.max_estimators),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
        )

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    inner_splits = _effective_splits(y, config.inner_cv_splits)
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    def objective(trial: Any) -> float:
        params = _suggest_parameters(trial, config)
        scores: list[float] = []
        best_estimators: list[int] = []
        for fold_index, (train_index, valid_index) in enumerate(
            splitter.split(X, y), start=1
        ):
            trial_progress.set_postfix_str(
                f"trial={trial.number + 1}, inner fold={fold_index}/{inner_splits}",
                refresh=True,
            )
            X_train, X_valid = _prepare_native_fold(
                X.iloc[train_index], X.iloc[valid_index]
            )
            y_train, y_valid = y[train_index], y[valid_index]
            train_weights = _training_weights(
                base_weights[train_index], y_train, config.use_balanced_class_weights
            )
            model = _make_classifier(
                config,
                n_classes=len(np.unique(y)),
                params=params,
                n_estimators=config.max_estimators,
                seed=seed + fold_index,
                early_stopping=True,
            )
            model.fit(
                X_train,
                y_train,
                sample_weight=train_weights,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
            predictions = model.predict(X_valid)
            scores.append(
                _score_predictions(config.objective_metric, y_valid, predictions)
            )
            best_iteration = getattr(model, "best_iteration", None)
            best_estimators.append(
                int(best_iteration) + 1
                if best_iteration is not None
                else config.max_estimators
            )
            trial.report(float(np.mean(scores)), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr(
            "median_best_n_estimators", int(np.median(best_estimators))
        )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(config.optuna_startup_trials, max(1, n_trials)),
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(
            config.optuna_pruner_startup_trials, max(1, n_trials)
        ),
        n_warmup_steps=config.optuna_pruner_warmup_steps,
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=config.reuse_existing_studies,
        sampler=sampler,
        pruner=pruner,
    )
    # When a previous execution stopped after Optuna finished, resume the
    # existing study instead of adding another full batch of trials.
    finished_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    finished_trial_count = sum(
        trial.state in finished_states for trial in study.get_trials(deepcopy=False)
    )
    remaining_trials = max(0, int(n_trials) - int(finished_trial_count))
    scope_label = (
        f"outer fold {outer_fold}" if outer_fold is not None else "final full data"
    )
    trial_progress = create_progress_bar(
        total=max(1, int(n_trials)),
        initial=min(int(finished_trial_count), int(n_trials)),
        description=f"Optuna · {scope_label}",
        position=2,
        leave=False,
        enabled=config.show_progress_bars,
        unit="trial",
    )

    def _update_trial_progress(_study: Any, completed_trial: Any) -> None:
        trial_progress.update(1)
        value = completed_trial.value
        value_text = "n/a" if value is None else f"{float(value):.4f}"
        trial_progress.set_postfix_str(
            f"trial={completed_trial.number + 1}, score={value_text}",
            refresh=True,
        )

    try:
        if remaining_trials > 0:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=config.optuna_timeout_seconds,
                n_jobs=int(config.optuna_n_jobs),
                show_progress_bar=False,
                callbacks=[_update_trial_progress],
            )
    finally:
        trial_progress.close()

    if not any(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.get_trials(deepcopy=False)
    ):
        raise RuntimeError(
            f"Optuna study {study.study_name!r} has no completed trials."
        )

    best_estimators = int(
        study.best_trial.user_attrs.get(
            "median_best_n_estimators", min(500, config.max_estimators)
        )
    )
    trial_rows: list[dict[str, object]] = []
    for trial in study.trials:
        row: dict[str, object] = {
            "scope": scope,
            "outer_fold": outer_fold,
            "study_name": study.study_name,
            "trial_number": trial.number,
            "state": trial.state.name,
            "objective_value": trial.value,
            "median_best_n_estimators": trial.user_attrs.get(
                "median_best_n_estimators"
            ),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        trial_rows.append(row)

    importance_rows: list[dict[str, object]] = []
    try:
        importances = optuna.importance.get_param_importances(study)
        importance_rows = [
            {
                "scope": scope,
                "outer_fold": outer_fold,
                "study_name": study.study_name,
                "parameter": parameter,
                "importance": float(importance),
            }
            for parameter, importance in importances.items()
        ]
    except Exception:
        importance_rows = []

    return (
        dict(study.best_params),
        best_estimators,
        pd.DataFrame(trial_rows),
        pd.DataFrame(importance_rows),
        float(study.best_value),
    )


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_classes = len(class_labels)
    class_indices = np.arange(n_classes)
    y_binary = label_binarize(y_true, classes=class_indices)
    if y_binary.ndim == 1:
        y_binary = y_binary.reshape(-1, 1)

    global_row: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "multiclass_log_loss": float(
            log_loss(y_true, probabilities, labels=class_indices)
        ),
    }
    try:
        global_row["macro_roc_auc_ovr"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                average="macro",
                multi_class="ovr",
                labels=class_indices,
            )
        )
    except ValueError:
        global_row["macro_roc_auc_ovr"] = np.nan

    y_onehot = np.eye(n_classes, dtype=float)[y_true]
    global_row["multiclass_brier_score"] = float(
        np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))
    )

    class_rows: list[dict[str, object]] = []
    for class_index, class_label in enumerate(class_labels):
        true_binary = (y_true == class_index).astype(int)
        pred_binary = (y_pred == class_index).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            true_binary, pred_binary, labels=[0, 1]
        ).ravel()
        # With average="binary", scikit-learn returns support=None by design.
        # The class support is therefore computed explicitly from y_true.
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_binary,
            pred_binary,
            average="binary",
            zero_division=0,
        )
        class_support = int(true_binary.sum())
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        row: dict[str, object] = {
            "cluster": class_label,
            "n_true": int(true_binary.sum()),
            "prevalence": float(true_binary.mean()),
            "n_predicted": int(pred_binary.sum()),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "precision": float(precision),
            "sensitivity_recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "support": class_support,
        }
        try:
            row["roc_auc_ovr"] = float(
                roc_auc_score(true_binary, probabilities[:, class_index])
            )
            row["pr_auc_ovr"] = float(
                average_precision_score(true_binary, probabilities[:, class_index])
            )
        except ValueError:
            row["roc_auc_ovr"] = np.nan
            row["pr_auc_ovr"] = np.nan
        class_rows.append(row)
    return pd.DataFrame([global_row]), pd.DataFrame(class_rows)


def _collect_shap_importance(
    model: Any,
    X_valid_native: pd.DataFrame,
    feature_names: list[str],
    class_labels: list[str],
    fold_number: int,
    config: XGBoostConfig,
) -> list[dict[str, object]]:
    if not config.calculate_shap_diagnostics:
        return []
    shap_limit = min(int(config.shap_max_rows_per_fold), len(X_valid_native))
    if shap_limit <= 0:
        return []

    try:
        import xgboost as xgb

        rng = np.random.default_rng(int(config.random_state) + fold_number)
        sampled_positions = (
            np.arange(len(X_valid_native))
            if shap_limit == len(X_valid_native)
            else np.sort(
                rng.choice(len(X_valid_native), size=shap_limit, replace=False)
            )
        )
        X_shap = X_valid_native.iloc[sampled_positions]
        booster = model.get_booster()
        shap_matrix = booster.predict(
            xgb.DMatrix(X_shap, enable_categorical=True),
            pred_contribs=True,
            strict_shape=True,
        )
        shap_matrix = np.asarray(shap_matrix)
        n_classes = len(class_labels)
        n_features = len(feature_names)
        if shap_matrix.ndim == 4 and shap_matrix.shape[1] == 1:
            shap_matrix = shap_matrix[:, 0, :, :]
        if shap_matrix.ndim == 2:
            expected = n_classes * (n_features + 1)
            if shap_matrix.shape[1] != expected:
                raise ValueError(
                    f"Unexpected multiclass pred_contribs shape: {shap_matrix.shape}."
                )
            shap_matrix = shap_matrix.reshape(
                shap_matrix.shape[0], n_classes, n_features + 1
            )
        if shap_matrix.ndim != 3:
            raise ValueError(
                f"Multiclass pred_contribs could not be interpreted: {shap_matrix.shape}."
            )
        mean_abs_shap = np.abs(shap_matrix[:, :, :-1]).mean(axis=0)
        rows: list[dict[str, object]] = []
        for class_index, class_label in enumerate(class_labels):
            for feature_index, feature_name in enumerate(feature_names):
                rows.append(
                    {
                        "fold": fold_number,
                        "cluster": class_label,
                        "feature": feature_name,
                        "mean_abs_shap": float(
                            mean_abs_shap[class_index, feature_index]
                        ),
                        "n_shap_rows": int(shap_matrix.shape[0]),
                    }
                )
        return rows
    except Exception as exc:
        if config.strict_shap_diagnostics:
            raise
        warnings.warn(
            f"Class-specific SHAP diagnostics were skipped for fold {fold_number}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []


def run_xgboost_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray,
    output_dir: str | Path,
    *,
    config: XGBoostConfig | None = None,
) -> XGBoostPipelineResult:
    """Run nested CV, generate OOF diagnostics, and train the final model."""

    config = config or XGBoostConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = output_dir / "optuna_xgboost_clusters.sqlite3"

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.reset_index(drop=True)
    X.columns = X.columns.map(str)
    y = pd.Series(y).reset_index(drop=True).astype(str)
    sample_weights = np.asarray(sample_weights, dtype=float).reshape(-1)
    if not (len(X) == len(y) == len(sample_weights)):
        raise ValueError("X, y, and sample_weights must have the same length.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_labels = list(label_encoder.classes_)
    n_classes = len(class_labels)
    outer_splits = _effective_splits(y_encoded, config.outer_cv_splits)
    splitter = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=config.random_state
    )

    oof_probabilities = np.full((len(X), n_classes), np.nan, dtype=float)
    oof_predictions = np.full(len(X), -1, dtype=int)
    fold_assignment = np.full(len(X), -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    shap_rows: list[dict[str, Any]] = []
    learning_curve_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []

    pipeline_progress = create_progress_bar(
        total=outer_splits + 3,
        description="XGBoost pipeline",
        position=1,
        leave=config.progress_leave,
        enabled=config.show_progress_bars,
        unit="stage",
    )

    for outer_fold, (train_index, valid_index) in enumerate(
        splitter.split(X, y_encoded), start=1
    ):
        pipeline_progress.set_postfix_str(
            f"outer fold {outer_fold}/{outer_splits} · Optuna",
            refresh=True,
        )
        (
            best_params,
            best_estimators,
            trials_df,
            optuna_importance_df,
            best_inner_value,
        ) = _run_optuna_study(
            X.iloc[train_index],
            y_encoded[train_index],
            sample_weights[train_index],
            config=config,
            storage_path=storage_path,
            study_name=f"{config.optuna_study_basename}_outer_{outer_fold}",
            n_trials=config.trials_per_outer_fold,
            seed=config.random_state + outer_fold,
            scope="outer_fold",
            outer_fold=outer_fold,
        )
        if not trials_df.empty:
            trial_frames.append(trials_df)
        if not optuna_importance_df.empty:
            importance_frames.append(optuna_importance_df)

        pipeline_progress.set_postfix_str(
            f"outer fold {outer_fold}/{outer_splits} · train and evaluate",
            refresh=True,
        )
        X_train, X_valid = _prepare_native_fold(
            X.iloc[train_index], X.iloc[valid_index]
        )
        y_train, y_valid = y_encoded[train_index], y_encoded[valid_index]
        train_weights = _training_weights(
            sample_weights[train_index], y_train, config.use_balanced_class_weights
        )
        model = _make_classifier(
            config,
            n_classes,
            best_params,
            best_estimators,
            seed=config.random_state + 1000 + outer_fold,
            early_stopping=False,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False,
        )
        probabilities = np.asarray(model.predict_proba(X_valid), dtype=float)
        probabilities = np.clip(probabilities, 0.0, None)
        probability_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = np.divide(
            probabilities,
            probability_sums,
            out=np.full_like(probabilities, 1.0 / n_classes),
            where=probability_sums > 0,
        )
        predictions = np.argmax(probabilities, axis=1)
        oof_probabilities[valid_index] = probabilities
        oof_predictions[valid_index] = predictions
        fold_assignment[valid_index] = outer_fold

        fold_global, _ = _compute_metrics(
            y_valid, predictions, probabilities, class_labels
        )
        fold_row: dict[str, object] = {
            "fold": outer_fold,
            "n_train": len(train_index),
            "n_valid": len(valid_index),
            "optuna_best_inner_score": best_inner_value,
            "best_trees": best_estimators,
        }
        fold_row.update(fold_global.iloc[0].to_dict())
        fold_rows.append(fold_row)

        booster = model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")
        weight_scores = booster.get_score(importance_type="weight")
        for feature_index, feature_name in enumerate(X.columns):
            importance_rows.append(
                {
                    "fold": outer_fold,
                    "feature": feature_name,
                    "gain": float(
                        gain_scores.get(
                            feature_name, gain_scores.get(f"f{feature_index}", 0.0)
                        )
                    ),
                    "split_count": float(
                        weight_scores.get(
                            feature_name, weight_scores.get(f"f{feature_index}", 0.0)
                        )
                    ),
                }
            )

        try:
            evals_result = model.evals_result()
        except Exception:
            evals_result = {}
        for dataset_name, metric_dict in evals_result.items():
            for metric_name, values in metric_dict.items():
                for iteration, value in enumerate(values, start=1):
                    learning_curve_rows.append(
                        {
                            "fold": outer_fold,
                            "dataset": dataset_name,
                            "metric": metric_name,
                            "iteration": iteration,
                            "value": float(value),
                        }
                    )

        pipeline_progress.set_postfix_str(
            f"outer fold {outer_fold}/{outer_splits} · SHAP diagnostics",
            refresh=True,
        )
        shap_rows.extend(
            _collect_shap_importance(
                model,
                X_valid,
                list(X.columns),
                class_labels,
                outer_fold,
                config,
            )
        )

        parameter_row: dict[str, object] = {
            "fold": outer_fold,
            "best_objective_value": best_inner_value,
            "n_estimators": best_estimators,
        }
        parameter_row.update(best_params)
        parameter_rows.append(parameter_row)
        pipeline_progress.update(1)

    pipeline_progress.set_postfix_str("OOF metrics and aggregation", refresh=True)
    if np.isnan(oof_probabilities).any() or (oof_predictions < 0).any():
        raise RuntimeError("OOF predictions were not generated for every participant.")

    global_metrics, metrics_by_cluster = _compute_metrics(
        y_encoded, oof_predictions, oof_probabilities, class_labels
    )

    maximum_probability = oof_probabilities.max(axis=1)
    correct = oof_predictions == y_encoded
    calibration_edges = np.linspace(0.0, 1.0, 11)
    calibration_bin = np.digitize(
        maximum_probability, calibration_edges[1:-1], right=True
    )
    expected_calibration_error = 0.0
    for bin_index in range(10):
        mask = calibration_bin == bin_index
        if not mask.any():
            continue
        expected_calibration_error += float(mask.mean()) * abs(
            float(maximum_probability[mask].mean()) - float(correct[mask].mean())
        )
    global_metrics["top_label_ece"] = expected_calibration_error

    feature_importance_by_fold = pd.DataFrame(importance_rows)
    feature_importance = (
        feature_importance_by_fold.groupby("feature", as_index=False)
        .agg(
            mean_gain=("gain", "mean"),
            standard_deviation_gain=("gain", "std"),
            mean_split_count=("split_count", "mean"),
            folds_observed=("fold", "nunique"),
        )
        .sort_values("mean_gain", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    shap_importance_by_fold = pd.DataFrame(shap_rows)
    if shap_importance_by_fold.empty:
        shap_importance_by_cluster = pd.DataFrame(
            columns=[
                "cluster",
                "feature",
                "mean_abs_shap",
                "sd_abs_shap",
                "folds",
            ]
        )
    else:
        shap_importance_by_cluster = (
            shap_importance_by_fold.groupby(
                ["cluster", "feature"], as_index=False
            )
            .agg(
                mean_abs_shap=("mean_abs_shap", "mean"),
                sd_abs_shap=("mean_abs_shap", "std"),
                folds=("fold", "nunique"),
            )
            .sort_values(
                ["cluster", "mean_abs_shap"],
                ascending=[True, False],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    learning_curves = pd.DataFrame(learning_curve_rows)
    if learning_curves.empty:
        learning_curve_summary = pd.DataFrame(
            columns=[
                "dataset",
                "metric",
                "iteration",
                "mean_value",
                "sd_value",
                "folds",
            ]
        )
    else:
        learning_curve_summary = (
            learning_curves.groupby(
                ["dataset", "metric", "iteration"], as_index=False
            )
            .agg(
                mean_value=("value", "mean"),
                sd_value=("value", "std"),
                folds=("fold", "nunique"),
            )
        )

    pipeline_progress.update(1)
    pipeline_progress.set_postfix_str("final Optuna study", refresh=True)

    (
        final_params,
        final_n_estimators,
        final_trials,
        final_optuna_importance,
        final_best_value,
    ) = _run_optuna_study(
        X,
        y_encoded,
        sample_weights,
        config=config,
        storage_path=storage_path,
        study_name=f"{config.optuna_study_basename}_final_full_data",
        n_trials=config.final_trials,
        seed=config.random_state + 9999,
        scope="final_full_data",
        outer_fold=None,
    )
    if not final_trials.empty:
        trial_frames.append(final_trials)
    if not final_optuna_importance.empty:
        importance_frames.append(final_optuna_importance)

    pipeline_progress.update(1)
    pipeline_progress.set_postfix_str("train final model", refresh=True)
    X_native = _prepare_native_full(X)
    final_weights = _training_weights(
        sample_weights, y_encoded, config.use_balanced_class_weights
    )
    final_model = _make_classifier(
        config,
        n_classes,
        final_params,
        final_n_estimators,
        seed=config.random_state + 20000,
        early_stopping=False,
    )
    final_model.fit(X_native, y_encoded, sample_weight=final_weights, verbose=False)
    pipeline_progress.update(1)
    pipeline_progress.set_postfix_str("complete", refresh=True)
    pipeline_progress.close()

    optuna_trials = (
        pd.concat(trial_frames, ignore_index=True, sort=False)
        if trial_frames
        else pd.DataFrame()
    )
    optuna_parameter_importance = (
        pd.concat(importance_frames, ignore_index=True, sort=False)
        if importance_frames
        else pd.DataFrame(
            columns=[
                "scope",
                "outer_fold",
                "study_name",
                "parameter",
                "importance",
            ]
        )
    )

    return XGBoostPipelineResult(
        config=config,
        class_labels=class_labels,
        y_true=y_encoded,
        y_pred=oof_predictions,
        oof_probabilities=oof_probabilities,
        fold_assignment=fold_assignment,
        global_metrics=global_metrics,
        metrics_by_fold=pd.DataFrame(fold_rows),
        metrics_by_cluster=metrics_by_cluster,
        feature_importance=feature_importance,
        feature_importance_by_fold=feature_importance_by_fold,
        shap_importance_by_cluster=shap_importance_by_cluster,
        shap_importance_by_fold=shap_importance_by_fold,
        learning_curves=learning_curves,
        learning_curve_summary=learning_curve_summary,
        optuna_trials=optuna_trials,
        optuna_parameter_importance=optuna_parameter_importance,
        best_parameters_by_fold=pd.DataFrame(parameter_rows),
        final_best_parameters=final_params,
        final_best_value=final_best_value,
        final_n_estimators=final_n_estimators,
        final_model=final_model,
        label_encoder=label_encoder,
    )
