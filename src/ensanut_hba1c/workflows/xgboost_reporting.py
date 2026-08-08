"""Complete CV reporting for the modular XGBoost cluster classifier.

Every diagnostic figure is saved to the PDF/PNG report and, when requested,
displayed directly in Jupyter.  The notebook therefore contains the visual
quality-control record without printing large audit tables or long status logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import pickle
import re
import zipfile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .xgboost_engine import XGBoostPipelineResult
from .progress import create_progress_bar


@dataclass(frozen=True)
class XGBoostReportConfig:
    dpi: int = 300
    top_global_features: int = 60
    top_features_per_cluster: int = 30
    top_confusion_pairs: int = 20
    top_optuna_parameters: int = 15
    calibration_bins: int = 10
    save_individual_plots: bool = True
    show_in_notebook: bool = True
    show_summary_page_in_notebook: bool = False
    close_after_display: bool = True
    verbose: bool = False
    show_progress_bars: bool = True
    progress_leave: bool = True


def _safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _display_and_close(
    fig: plt.Figure,
    *,
    show: bool,
    close_after_display: bool,
) -> None:
    if show:
        try:
            from IPython.display import display

            display(fig)
        except Exception:
            plt.show()
    if close_after_display:
        plt.close(fig)


def _save_report_figure(
    fig: plt.Figure,
    pdf: PdfPages,
    plots_dir: Path,
    filename: str,
    *,
    config: XGBoostReportConfig,
    show_in_notebook: bool = True,
) -> None:
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    if config.save_individual_plots:
        fig.savefig(
            plots_dir / filename,
            dpi=int(config.dpi),
            bbox_inches="tight",
            facecolor="white",
        )
    _display_and_close(
        fig,
        show=config.show_in_notebook and show_in_notebook,
        close_after_display=config.close_after_display,
    )


def _build_oof_table(
    result: XGBoostPipelineResult,
    participant_keys: pd.DataFrame | None,
) -> pd.DataFrame:
    if participant_keys is None:
        frame = pd.DataFrame(index=np.arange(len(result.y_true)))
    else:
        frame = participant_keys.reset_index(drop=True).copy()

    true_labels = result.label_encoder.inverse_transform(result.y_true)
    predicted_labels = result.label_encoder.inverse_transform(result.y_pred)
    sorted_probabilities = np.sort(result.oof_probabilities, axis=1)
    probability_margin = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        if result.oof_probabilities.shape[1] > 1
        else sorted_probabilities[:, -1]
    )
    entropy = -np.sum(
        result.oof_probabilities
        * np.log(np.clip(result.oof_probabilities, 1e-15, 1.0)),
        axis=1,
    )
    normalized_entropy = entropy / np.log(max(len(result.class_labels), 2))

    frame["true_cluster"] = true_labels
    frame["predicted_cluster"] = predicted_labels
    frame["prediction_correct"] = true_labels == predicted_labels
    frame["fold"] = result.fold_assignment
    frame["maximum_predicted_probability"] = result.oof_probabilities.max(axis=1)
    frame["true_cluster_probability"] = result.oof_probabilities[
        np.arange(len(result.y_true)), result.y_true
    ]
    frame["probability_margin_top1_top2"] = probability_margin
    frame["normalized_probability_entropy"] = normalized_entropy
    for class_index, class_label in enumerate(result.class_labels):
        safe_label = _safe_filename(class_label)
        frame[f"probability_{safe_label}"] = result.oof_probabilities[:, class_index]
    return frame


def _build_confusion_tables(
    result: XGBoostPipelineResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    labels = np.arange(len(result.class_labels))
    counts = confusion_matrix(result.y_true, result.y_pred, labels=labels)
    row_pct = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    counts_df = pd.DataFrame(
        counts, index=result.class_labels, columns=result.class_labels
    )
    counts_df.index.name = "true_cluster"
    row_pct_df = pd.DataFrame(
        row_pct, index=result.class_labels, columns=result.class_labels
    )
    row_pct_df.index.name = "true_cluster"

    pair_rows: list[dict[str, object]] = []
    for true_index, true_label in enumerate(result.class_labels):
        true_total = int(counts[true_index].sum())
        for predicted_index, predicted_label in enumerate(result.class_labels):
            if true_index == predicted_index:
                continue
            count = int(counts[true_index, predicted_index])
            if count <= 0:
                continue
            pair_rows.append(
                {
                    "true_cluster": true_label,
                    "predicted_cluster": predicted_label,
                    "count": count,
                    "fraction_within_true_cluster": (
                        count / true_total if true_total else np.nan
                    ),
                    "pair": f"{true_label} → {predicted_label}",
                }
            )
    if pair_rows:
        pairs_df = (
            pd.DataFrame(pair_rows)
            .sort_values(
                ["count", "fraction_within_true_cluster"],
                ascending=False,
                kind="stable",
            )
            .reset_index(drop=True)
        )
    else:
        pairs_df = pd.DataFrame(
            columns=[
                "true_cluster",
                "predicted_cluster",
                "count",
                "fraction_within_true_cluster",
                "pair",
            ]
        )
    return counts_df, row_pct_df, pairs_df, counts, row_pct


def _build_calibration_table(
    result: XGBoostPipelineResult,
    n_bins: int,
) -> tuple[pd.DataFrame, float]:
    confidence = result.oof_probabilities.max(axis=1)
    correct = result.y_pred == result.y_true
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_index = np.digitize(confidence, edges[1:-1], right=True)
    rows: list[dict[str, object]] = []
    expected_calibration_error = 0.0
    for index in range(int(n_bins)):
        mask = bin_index == index
        n_bin = int(mask.sum())
        if n_bin:
            mean_confidence = float(confidence[mask].mean())
            observed_accuracy = float(correct[mask].mean())
            absolute_gap = abs(mean_confidence - observed_accuracy)
            expected_calibration_error += (n_bin / len(confidence)) * absolute_gap
        else:
            mean_confidence = np.nan
            observed_accuracy = np.nan
            absolute_gap = np.nan
        rows.append(
            {
                "bin": index + 1,
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "n": n_bin,
                "mean_confidence": mean_confidence,
                "observed_accuracy": observed_accuracy,
                "absolute_gap": absolute_gap,
            }
        )
    return pd.DataFrame(rows), float(expected_calibration_error)


def _build_confidence_by_cluster(oof_table: pd.DataFrame) -> pd.DataFrame:
    return (
        oof_table.groupby("true_cluster", as_index=False)
        .agg(
            n=("prediction_correct", "size"),
            accuracy=("prediction_correct", "mean"),
            mean_max_probability=("maximum_predicted_probability", "mean"),
            median_max_probability=("maximum_predicted_probability", "median"),
            mean_true_cluster_probability=("true_cluster_probability", "mean"),
            mean_probability_margin=("probability_margin_top1_top2", "mean"),
            mean_normalized_entropy=("normalized_probability_entropy", "mean"),
        )
        .rename(columns={"true_cluster": "cluster"})
    )


def _build_curve_tables(
    result: XGBoostPipelineResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    roc_rows: list[dict[str, object]] = []
    pr_rows: list[dict[str, object]] = []
    for class_index, class_label in enumerate(result.class_labels):
        true_binary = (result.y_true == class_index).astype(int)
        if np.unique(true_binary).size < 2:
            continue
        fpr, tpr, roc_thresholds = roc_curve(
            true_binary, result.oof_probabilities[:, class_index]
        )
        for point, (fpr_value, tpr_value, threshold) in enumerate(
            zip(fpr, tpr, roc_thresholds)
        ):
            roc_rows.append(
                {
                    "cluster": class_label,
                    "point": point,
                    "false_positive_rate": float(fpr_value),
                    "true_positive_rate": float(tpr_value),
                    "threshold": float(threshold),
                }
            )
        precision, recall, pr_thresholds = precision_recall_curve(
            true_binary, result.oof_probabilities[:, class_index]
        )
        for point, (precision_value, recall_value) in enumerate(
            zip(precision, recall)
        ):
            threshold = (
                float(pr_thresholds[point]) if point < len(pr_thresholds) else np.nan
            )
            pr_rows.append(
                {
                    "cluster": class_label,
                    "point": point,
                    "recall": float(recall_value),
                    "precision": float(precision_value),
                    "threshold": threshold,
                }
            )
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def _model_audit_to_frame(model_audit: object | None) -> pd.DataFrame:
    if model_audit is None:
        return pd.DataFrame()
    if isinstance(model_audit, pd.DataFrame):
        return model_audit.copy()
    if isinstance(model_audit, dict):
        return pd.DataFrame([model_audit])
    try:
        return pd.DataFrame([vars(model_audit)])
    except Exception:
        return pd.DataFrame([{"model_audit": str(model_audit)}])


def export_xgboost_results(
    result: XGBoostPipelineResult,
    output_dir: str | Path,
    *,
    participant_keys: pd.DataFrame | None = None,
    feature_selection_audit: pd.DataFrame | None = None,
    model_audit: object | None = None,
    config: XGBoostReportConfig | None = None,
) -> dict[str, Path]:
    """Export and display the complete OOF/CV diagnostic report."""

    config = config or XGBoostReportConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = output_dir / "xgboost_cluster_classification_report.xlsx"
    pdf_path = output_dir / "xgboost_cluster_classification_report.pdf"
    model_json_path = output_dir / "xgboost_cluster_classifier.json"
    model_bundle_path = output_dir / "xgboost_cluster_classifier_bundle.pkl"
    manifest_path = output_dir / "xgboost_cluster_classification_manifest.json"
    zip_path = output_dir.parent / "xgboost_cluster_classification_results.zip"

    report_progress = create_progress_bar(
        total=5,
        description="Diagnostic report",
        position=1,
        leave=config.progress_leave,
        enabled=config.show_progress_bars,
        unit="stage",
    )
    report_progress.set_postfix_str("build OOF diagnostic tables", refresh=True)

    oof_table = _build_oof_table(result, participant_keys)
    (
        confusion_counts_df,
        confusion_row_pct_df,
        top_confusion_pairs_df,
        confusion_counts,
        confusion_row_pct,
    ) = _build_confusion_tables(result)
    calibration_df, expected_calibration_error = _build_calibration_table(
        result, config.calibration_bins
    )
    confidence_by_cluster_df = _build_confidence_by_cluster(oof_table)
    roc_curves_df, precision_recall_curves_df = _build_curve_tables(result)

    global_metrics_df = result.global_metrics.copy()
    global_metrics_df["top_label_ece"] = expected_calibration_error
    model_audit_df = _model_audit_to_frame(model_audit)

    hyperparameters_df = pd.DataFrame(
        [
            {"parameter": parameter, "value": value}
            for parameter, value in {
                "selection_method": "Optuna TPE with nested stratified CV",
                "objective": "multi:softprob",
                "optimization_metric": result.config.objective_metric,
                "outer_cv_splits": result.config.outer_cv_splits,
                "inner_cv_splits": result.config.inner_cv_splits,
                "trials_per_outer_fold": result.config.trials_per_outer_fold,
                "final_trials": result.config.final_trials,
                "final_n_estimators": result.final_n_estimators,
                "final_best_value": result.final_best_value,
                "tree_method": result.config.tree_method,
                "balanced_class_weights": result.config.use_balanced_class_weights,
                "imputation": "none",
                **result.final_best_parameters,
            }.items()
        ]
    )

    report_progress.update(1)
    report_progress.set_postfix_str("write Excel and CSV tables", refresh=True)

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        global_metrics_df.to_excel(writer, sheet_name="Global_metrics", index=False)
        result.metrics_by_cluster.to_excel(
            writer, sheet_name="Metrics_by_cluster", index=False
        )
        result.metrics_by_fold.to_excel(
            writer, sheet_name="Metrics_by_fold", index=False
        )
        oof_table.to_excel(writer, sheet_name="OOF_predictions", index=False)
        confusion_counts_df.to_excel(writer, sheet_name="Confusion_counts")
        confusion_row_pct_df.to_excel(writer, sheet_name="Confusion_row_pct")
        top_confusion_pairs_df.to_excel(
            writer, sheet_name="Top_confusions", index=False
        )
        calibration_df.to_excel(writer, sheet_name="OOF_calibration", index=False)
        confidence_by_cluster_df.to_excel(
            writer, sheet_name="OOF_confidence", index=False
        )
        roc_curves_df.to_excel(writer, sheet_name="ROC_curves", index=False)
        precision_recall_curves_df.to_excel(
            writer, sheet_name="PR_curves", index=False
        )
        result.feature_importance.to_excel(
            writer, sheet_name="Feature_importance", index=False
        )
        result.feature_importance_by_fold.to_excel(
            writer, sheet_name="Importance_by_fold", index=False
        )
        result.shap_importance_by_cluster.to_excel(
            writer, sheet_name="SHAP_by_cluster", index=False
        )
        result.shap_importance_by_fold.to_excel(
            writer, sheet_name="SHAP_by_fold", index=False
        )
        result.learning_curve_summary.to_excel(
            writer, sheet_name="Learning_curve", index=False
        )
        result.best_parameters_by_fold.to_excel(
            writer, sheet_name="Optuna_outer_best", index=False
        )
        result.optuna_trials.to_excel(
            writer, sheet_name="Optuna_trials", index=False
        )
        result.optuna_parameter_importance.to_excel(
            writer, sheet_name="Optuna_importance", index=False
        )
        hyperparameters_df.to_excel(
            writer, sheet_name="Hyperparameters", index=False
        )
        if feature_selection_audit is not None:
            feature_selection_audit.to_excel(
                writer, sheet_name="Feature_selection", index=False
            )
        if not model_audit_df.empty:
            model_audit_df.to_excel(
                writer, sheet_name="Prepare_matrix_audit", index=False
            )

    csv_tables: dict[str, pd.DataFrame] = {
        "global_metrics_oof.csv": global_metrics_df,
        "metrics_by_cluster_oof.csv": result.metrics_by_cluster,
        "metrics_by_fold.csv": result.metrics_by_fold,
        "oof_predictions.csv": oof_table,
        "confusion_counts.csv": confusion_counts_df.reset_index(),
        "confusion_row_pct.csv": confusion_row_pct_df.reset_index(),
        "top_confusion_pairs.csv": top_confusion_pairs_df,
        "oof_calibration.csv": calibration_df,
        "oof_confidence_by_cluster.csv": confidence_by_cluster_df,
        "roc_curves_by_cluster.csv": roc_curves_df,
        "precision_recall_curves_by_cluster.csv": precision_recall_curves_df,
        "feature_importance_global.csv": result.feature_importance,
        "feature_importance_by_fold.csv": result.feature_importance_by_fold,
        "shap_importance_by_cluster.csv": result.shap_importance_by_cluster,
        "shap_importance_by_fold.csv": result.shap_importance_by_fold,
        "learning_curves.csv": result.learning_curves,
        "learning_curve_summary.csv": result.learning_curve_summary,
        "optuna_trials.csv": result.optuna_trials,
        "optuna_best_by_outer_fold.csv": result.best_parameters_by_fold,
        "optuna_parameter_importance.csv": result.optuna_parameter_importance,
        "hyperparameters.csv": hyperparameters_df,
    }
    if feature_selection_audit is not None:
        csv_tables["feature_selection_audit.csv"] = feature_selection_audit
    if not model_audit_df.empty:
        csv_tables["prepare_model_matrix_audit.csv"] = model_audit_df
    for filename, table in csv_tables.items():
        table.to_csv(output_dir / filename, index=False)

    report_progress.update(1)
    report_progress.set_postfix_str("save final model", refresh=True)
    result.final_model.save_model(model_json_path)
    with model_bundle_path.open("wb") as handle:
        pickle.dump(
            {
                "model": result.final_model,
                "label_encoder": result.label_encoder,
                "class_labels": result.class_labels,
                "configuration": asdict(result.config),
                "final_best_parameters": result.final_best_parameters,
                "final_best_value": result.final_best_value,
                "final_n_estimators": result.final_n_estimators,
            },
            handle,
        )

    report_progress.update(1)
    report_progress.set_postfix_str("generate diagnostic figures and PDF", refresh=True)

    class_labels = result.class_labels
    n_classes = len(class_labels)
    maximum_probability = result.oof_probabilities.max(axis=1)
    prediction_correct = result.y_pred == result.y_true

    with PdfPages(pdf_path) as pdf:
        # Summary page is retained in the exported PDF but hidden in the
        # notebook by default to avoid unnecessary text output.
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("XGBoost multiclass OOF cluster classification", fontsize=17)
        metrics = global_metrics_df.iloc[0]
        summary_lines = [
            f"Participants: {len(result.y_true):,}",
            f"Clusters: {n_classes}",
            f"Predictors: {len(result.feature_importance):,}",
            f"Outer stratified CV: {result.metrics_by_fold['fold'].nunique()} folds",
            f"Final Optuna objective: {result.final_best_value if result.final_best_value is not None else np.nan:.3f}",
            f"Final trees: {result.final_n_estimators}",
            f"OOF accuracy: {metrics['accuracy']:.3f}",
            f"OOF balanced accuracy: {metrics['balanced_accuracy']:.3f}",
            f"OOF macro-F1: {metrics['macro_f1']:.3f}",
            f"OOF MCC: {metrics['matthews_correlation_coefficient']:.3f}",
            f"OOF macro ROC-AUC: {metrics.get('macro_roc_auc_ovr', np.nan):.3f}",
            f"OOF log-loss: {metrics['multiclass_log_loss']:.3f}",
            f"OOF multiclass Brier score: {metrics['multiclass_brier_score']:.3f}",
            f"OOF top-label ECE: {expected_calibration_error:.3f}",
        ]
        fig.text(0.08, 0.88, "\n".join(summary_lines), va="top", fontsize=11.5)
        plt.axis("off")
        _save_report_figure(
            fig,
            pdf,
            plots_dir,
            "00_summary.png",
            config=config,
            show_in_notebook=config.show_summary_page_in_notebook,
        )

        fig, ax = plt.subplots(figsize=(11, 9))
        image = ax.imshow(confusion_row_pct, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(n_classes), labels=class_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(n_classes), labels=class_labels)
        ax.set_xlabel("Predicted cluster")
        ax.set_ylabel("True cluster")
        ax.set_title("OOF confusion matrix normalized by true cluster")
        for row in range(n_classes):
            for column in range(n_classes):
                ax.text(
                    column,
                    row,
                    f"{confusion_row_pct[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        fig.colorbar(image, ax=ax, label="Proportion")
        _save_report_figure(
            fig, pdf, plots_dir, "01_confusion_matrix_normalized.png", config=config
        )

        metric_columns = [
            "precision",
            "sensitivity_recall",
            "specificity",
            "f1",
        ]
        plot_metrics = result.metrics_by_cluster.set_index("cluster")[metric_columns]
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_metrics.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("OOF metric")
        ax.set_title("One-vs-rest performance by cluster")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="lower right")
        _save_report_figure(
            fig, pdf, plots_dir, "02_metrics_by_cluster.png", config=config
        )

        auc_plot = result.metrics_by_cluster.set_index("cluster")[[
            "roc_auc_ovr",
            "pr_auc_ovr",
        ]]
        fig, ax = plt.subplots(figsize=(12, 6.5))
        auc_plot.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("OOF area under the curve")
        ax.set_title("ROC and precision-recall discrimination by cluster")
        ax.tick_params(axis="x", rotation=45)
        _save_report_figure(
            fig, pdf, plots_dir, "03_auc_by_cluster.png", config=config
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        for cluster, cluster_df in roc_curves_df.groupby("cluster", sort=False):
            ax.plot(
                cluster_df["false_positive_rate"],
                cluster_df["true_positive_rate"],
                label=cluster,
            )
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.set_title("One-vs-rest OOF ROC curves")
        ax.legend(ncol=2, fontsize=8)
        _save_report_figure(
            fig, pdf, plots_dir, "04_roc_curves.png", config=config
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        for cluster, cluster_df in precision_recall_curves_df.groupby(
            "cluster", sort=False
        ):
            ax.plot(cluster_df["recall"], cluster_df["precision"], label=cluster)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("One-vs-rest OOF precision-recall curves")
        ax.legend(ncol=2, fontsize=8)
        _save_report_figure(
            fig, pdf, plots_dir, "05_precision_recall_curves.png", config=config
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(
            result.metrics_by_cluster["n_true"],
            result.metrics_by_cluster["f1"],
            s=np.maximum(30, result.metrics_by_cluster["prevalence"] * 1400),
            alpha=0.8,
        )
        for _, row in result.metrics_by_cluster.iterrows():
            ax.annotate(str(row["cluster"]), (row["n_true"], row["f1"]), fontsize=8)
        ax.set_xlabel("True participants in cluster")
        ax.set_ylabel("OOF F1")
        ax.set_ylim(0, 1.05)
        ax.set_title("Cluster support versus classification performance")
        _save_report_figure(
            fig, pdf, plots_dir, "06_support_vs_f1.png", config=config
        )

        top_pairs = top_confusion_pairs_df.head(
            int(config.top_confusion_pairs)
        ).copy()
        if not top_pairs.empty:
            top_pairs = top_pairs.sort_values("count")
            fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(top_pairs))))
            ax.barh(top_pairs["pair"], top_pairs["count"])
            ax.set_xlabel("Misclassified OOF participants")
            ax.set_title("Main confusion routes between clusters")
            _save_report_figure(
                fig, pdf, plots_dir, "07_top_confusion_routes.png", config=config
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.linspace(0, 1, 21)
        ax.hist(
            maximum_probability[prediction_correct],
            bins=bins,
            alpha=0.65,
            label="Correct prediction",
        )
        ax.hist(
            maximum_probability[~prediction_correct],
            bins=bins,
            alpha=0.65,
            label="Incorrect prediction",
        )
        ax.set_xlabel("Maximum OOF probability")
        ax.set_ylabel("Participants")
        ax.set_title("Model confidence in correct and incorrect predictions")
        ax.legend()
        _save_report_figure(
            fig, pdf, plots_dir, "08_confidence_correct_vs_incorrect.png", config=config
        )

        valid_calibration = calibration_df.loc[calibration_df["n"] > 0].copy()
        fig, ax = plt.subplots(figsize=(7.5, 7))
        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        ax.plot(
            valid_calibration["mean_confidence"],
            valid_calibration["observed_accuracy"],
            marker="o",
            label=f"OOF · ECE={expected_calibration_error:.3f}",
        )
        for _, row in valid_calibration.iterrows():
            ax.annotate(
                str(int(row["n"])),
                (row["mean_confidence"], row["observed_accuracy"]),
                fontsize=8,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean OOF confidence")
        ax.set_ylabel("Observed accuracy")
        ax.set_title("Top-label reliability diagram\n(number beside point = participants)")
        ax.legend()
        _save_report_figure(
            fig, pdf, plots_dir, "09_oof_calibration.png", config=config
        )

        fold_columns = [
            column
            for column in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "matthews_correlation_coefficient",
            ]
            if column in result.metrics_by_fold.columns
        ]
        if fold_columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            result.metrics_by_fold.set_index("fold")[fold_columns].plot(
                marker="o", ax=ax
            )
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Outer fold")
            ax.set_ylabel("Metric")
            ax.set_title("Performance stability across outer folds")
            ax.legend(loc="best")
            _save_report_figure(
                fig, pdf, plots_dir, "10_outer_fold_stability.png", config=config
            )

        final_trials = result.optuna_trials.loc[
            result.optuna_trials.get("scope", pd.Series(index=result.optuna_trials.index, dtype=object)).eq(
                "final_full_data"
            )
            & result.optuna_trials.get("state", pd.Series(index=result.optuna_trials.index, dtype=object)).eq(
                "COMPLETE"
            )
            & result.optuna_trials.get(
                "objective_value",
                pd.Series(index=result.optuna_trials.index, dtype=float),
            ).notna()
        ].sort_values("trial_number") if not result.optuna_trials.empty else pd.DataFrame()
        if not final_trials.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            running_best = final_trials["objective_value"].cummax()
            ax.scatter(
                final_trials["trial_number"],
                final_trials["objective_value"],
                alpha=0.65,
                label="Completed trial",
            )
            ax.plot(
                final_trials["trial_number"],
                running_best,
                linewidth=2,
                label="Running best",
            )
            ax.set_xlabel("Trial number")
            ax.set_ylabel(result.config.objective_metric)
            ax.set_title("Optuna optimization history · final study")
            ax.legend()
            _save_report_figure(
                fig, pdf, plots_dir, "11_optuna_history.png", config=config
            )

        final_parameter_importance = result.optuna_parameter_importance.loc[
            result.optuna_parameter_importance.get(
                "scope",
                pd.Series(
                    index=result.optuna_parameter_importance.index, dtype=object
                ),
            ).eq("final_full_data")
        ].head(int(config.top_optuna_parameters)).sort_values("importance") if not result.optuna_parameter_importance.empty else pd.DataFrame()
        if not final_parameter_importance.empty:
            fig, ax = plt.subplots(
                figsize=(10, max(5, 0.35 * len(final_parameter_importance)))
            )
            ax.barh(
                final_parameter_importance["parameter"],
                final_parameter_importance["importance"],
            )
            ax.set_xlabel("Importance for the Optuna objective")
            ax.set_title("Most influential hyperparameters")
            _save_report_figure(
                fig,
                pdf,
                plots_dir,
                "12_optuna_parameter_importance.png",
                config=config,
            )

        if not result.learning_curve_summary.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            for dataset_name, dataset_df in result.learning_curve_summary.groupby(
                "dataset", sort=False
            ):
                dataset_df = dataset_df.sort_values("iteration")
                ax.plot(
                    dataset_df["iteration"],
                    dataset_df["mean_value"],
                    label=str(dataset_name),
                )
            ax.set_xlabel("Boosting iteration")
            ax.set_ylabel("Mean mlogloss")
            ax.set_title("Mean learning curves across outer folds")
            ax.legend()
            _save_report_figure(
                fig, pdf, plots_dir, "13_learning_curves.png", config=config
            )

        top_gain = result.feature_importance.head(
            int(config.top_global_features)
        ).sort_values("mean_gain")
        if not top_gain.empty:
            fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(top_gain))))
            ax.barh(top_gain["feature"], top_gain["mean_gain"])
            ax.set_xlabel("Mean gain across outer folds")
            ax.set_title("Global feature importance")
            _save_report_figure(
                fig, pdf, plots_dir, "14_global_feature_importance.png", config=config
            )

        if not result.shap_importance_by_cluster.empty:
            for class_position, class_label in enumerate(class_labels, start=1):
                top_class = (
                    result.shap_importance_by_cluster.loc[
                        result.shap_importance_by_cluster["cluster"].eq(class_label)
                    ]
                    .head(int(config.top_features_per_cluster))
                    .sort_values("mean_abs_shap")
                )
                if top_class.empty:
                    continue
                fig, ax = plt.subplots(
                    figsize=(10, max(5, 0.28 * len(top_class)))
                )
                ax.barh(top_class["feature"], top_class["mean_abs_shap"])
                ax.set_xlabel("Mean |SHAP contribution|")
                ax.set_title(f"Features supporting the probability of {class_label}")
                _save_report_figure(
                    fig,
                    pdf,
                    plots_dir,
                    f"15_{class_position:02d}_shap_{_safe_filename(class_label)}.png",
                    config=config,
                )

    report_progress.update(1)
    report_progress.set_postfix_str("write manifest and ZIP archive", refresh=True)

    manifest = {
        "analysis": "XGBoost multiclass cluster classification",
        "evaluation": "nested cross-validation with OOF predictions",
        "n_participants": int(len(result.y_true)),
        "n_clusters": int(len(result.class_labels)),
        "cluster_labels": result.class_labels,
        "n_predictors": int(len(result.feature_importance)),
        "configuration": asdict(result.config),
        "report_configuration": asdict(config),
        "final_best_parameters": result.final_best_parameters,
        "final_best_value": result.final_best_value,
        "final_n_estimators": int(result.final_n_estimators),
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "diagnostic_figures": sorted(path.name for path in plots_dir.glob("*.png")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(output_dir.parent))

    report_progress.update(1)
    report_progress.set_postfix_str("complete", refresh=True)
    report_progress.close()

    if config.verbose:
        print(f"Workbook: {workbook_path}")
        print(f"PDF report: {pdf_path}")
        print(f"Diagnostic figures: {plots_dir}")

    return {
        "workbook": workbook_path,
        "pdf": pdf_path,
        "plots": plots_dir,
        "model_json": model_json_path,
        "model_bundle": model_bundle_path,
        "manifest": manifest_path,
        "zip": zip_path,
    }
