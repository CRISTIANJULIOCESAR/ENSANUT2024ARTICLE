# Notebook guide

The repository contains one primary workflow and several complementary/diagnostic analyses. The supplied notebooks are retained without changing their cells or parameters.

| Notebook | Role | Main prerequisite | Main purpose |
|---|---|---|---|
| `01_modeling_clustering.ipynb` | **Primary pipeline** | `data/input/ENSANUT_2024_mx.csv` + `ENSANUT_2024.csv` | HbA1c XGBoost regression, SHAP representation, PCA/shared UMAP graph, Leiden communities, visualizations, and handoff for downstream analyses. |
| `01.1_supplementary_stability_and_glucose_ablation.ipynb` | **Robustness / Supplementary** | Raw input CSVs | Re-runs XGBoost/SHAP and tests fixed-graph Leiden stability across 20 seeds, resolution sensitivity, glucose-only performance, and full reconstruction without direct glycemic measurements. |
| `02_naive_bayes_followup.ipynb` | **Canonical characterization** | Notebook 01 handoff | Hierarchical signed Naive-Bayes-style evidence, questionnaire-domain summaries, supervised evidence heatmap, and NB-filtered XGBoost cluster-label recoverability. |
| `02.1_naive_bayes_followup_binary_xgboost_no_glucose_fixed.ipynb` | **Recoverability audit** | Notebook 01 handoff | Keeps the hierarchical evidence analysis but uses a non-glycemic **binary rule matrix** for the final XGBoost label-recoverability audit. |
| `02.2_naive_bayes_followup_raw_xgboost_no_glucose_no_missing_categories.ipynb` | **Recoverability audit** | Notebook 01 handoff + raw input CSVs | Uses prepared **raw source variables**, excludes glycemic predictors, and converts textual missing/non-response categories to missing values before XGBoost label recovery. |
| `02_clusters_fusionados.ipynb` | **Cluster-merging sensitivity** | Notebook 01 handoff + raw input CSVs | Compares the original 10-community solution with a sensitivity analysis in which C5–C8 are merged; includes evidence and XGBoost analyses. |
| `02_clusters_fusionados_sin_GLU_SUERO.ipynb` | **Merged + glucose-excluded sensitivity** | Notebook 01 handoff + raw input CSVs | Repeats original and C5–C8 merged analyses while excluding `GLU_SUERO` from both signed evidence and XGBoost predictors. |
| `03_naive_bayes_filtrado_vs_sin_filtro_no_cv.ipynb` | **NB filter comparison** | `bayes_internal_operations_all_variables.xlsx` from Notebook 02 | Compares filtered and unfiltered multiclass NB log-odds models using apparent/in-sample metrics; explicitly no cross-validation. |
| `04_naive_bayes_best_bins.ipynb` | **Best-bin extension** | Notebook 02 evidence workbook; executes filtered/unfiltered NB first | Extends the filtered-vs-unfiltered comparison by selecting the most predictive bins and reporting a third best-bin model and associated metrics. |

## Recommended execution order

### Main reproducible path

```text
1. data/input/ENSANUT_2024_mx.csv
   data/input/ENSANUT_2024.csv
            │
            ▼
2. 01_modeling_clustering.ipynb
            │
            ├── handoff_to_notebook2/
            ▼
3. 02_naive_bayes_followup.ipynb
```

### Complementary analyses

After the input files are available:

- run `01.1_supplementary_stability_and_glucose_ablation.ipynb` independently for stability/glucose-ablation results;
- after Notebook 01, run any of `02.1`, `02.2`, or the two `02_clusters_fusionados*` notebooks;
- after canonical Notebook 02 has generated `bayes_internal_operations_all_variables.xlsx`, run Notebook 03 and then Notebook 04 as needed.

## Important output note

Several Notebook 02 variants intentionally write into the common `results/02_naive_bayes/` tree. Because their original code is preserved, running multiple variants in the same working tree can replace files with identical names. For archival runs, copy or rename the relevant `results/` folder between variants.

## Input discovery

Notebook 02-family analyses first look for the canonical handoff under:

```text
results/01_modeling_clustering/handoff_to_notebook2/
```

Some notebooks also contain fallback discovery and editable external-path variables. For a clean GitHub clone, leave the external paths as `None` and generate the handoff with Notebook 01.
