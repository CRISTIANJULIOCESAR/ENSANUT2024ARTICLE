from pathlib import Path

import pandas as pd

from ensanut_hba1c.paths import ProjectPaths
from ensanut_hba1c.workflows.nb_evidence import NBFilterConfig


def test_project_paths():
    root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths(root).ensure()
    assert paths.input_dir.is_dir()
    assert paths.notebook1_results.is_dir()
    assert paths.notebook2_results.is_dir()


def test_default_nb_filter_is_full_heatmap_filter():
    config = NBFilterConfig()
    assert config.min_n_cluster_x == 10
    assert config.min_score == 0.40
    assert config.min_coverage == 0.07


def test_supervised_heatmap_defaults_preserve_original_design():
    from ensanut_hba1c.workflows.supervised_heatmap import SupervisedHeatmapConfig

    config = SupervisedHeatmapConfig()
    assert config.cmap == "YlGnBu"
    assert config.annotation_font_size == 4.5
    assert config.annotation_wrap_width == 45
    assert config.heatmap_width == 3.35


def test_xgboost_report_displays_diagnostics_by_default():
    from ensanut_hba1c.workflows.xgboost_reporting import XGBoostReportConfig

    config = XGBoostReportConfig()
    assert config.show_in_notebook is True
    assert config.top_features_per_cluster == 30


def test_reduced_xgboost_runtime_defaults_and_progress():
    from ensanut_hba1c.workflows.xgboost_engine import XGBoostConfig
    from ensanut_hba1c.workflows.xgboost_reporting import XGBoostReportConfig

    config = XGBoostConfig()
    assert config.outer_cv_splits == 5
    assert config.inner_cv_splits == 2
    assert config.trials_per_outer_fold == 8
    assert config.final_trials == 5
    assert config.max_estimators == 1000
    assert config.early_stopping_rounds == 100
    assert config.shap_max_rows_per_fold == 300
    assert config.show_progress_bars is True

    report_config = XGBoostReportConfig()
    assert report_config.show_progress_bars is True
