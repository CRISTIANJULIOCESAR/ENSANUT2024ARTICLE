# GitHub notebook rendering

GitHub-oriented notebook copies are stored in `notebooks/github/`. The originals in `notebooks/` remain the scientific record and are unchanged.

## What was changed

No notebook was executed. The conversion operates only on already-saved output MIME bundles:

- `185` embedded `image/svg+xml` outputs were rasterized to embedded `image/png`.
- `4` Plotly MIME outputs already carrying PNG fallbacks were reduced to the static PNG representation.
- Code cells, markdown sources, parameters, execution counts, output order, and stored text/numerical output were preserved.
- Raw ENSANUT data were not required and are not included.

This is a display-layer transformation for GitHub, not a scientific recomputation.

## Exception: Notebook 01

`01_modeling_clustering.ipynb` was saved with **zero outputs**. There are therefore no embedded figures to recover or convert. Its copy in `notebooks/github/` is identical to the original source-only notebook. To show its figures, an already-executed copy would be needed, or the notebook would need to be executed with the required ENSANUT inputs.

## Per-notebook conversion

| Notebook | Saved outputs | SVG→PNG | Plotly→static PNG | PNG outputs after |
|---|---:|---:|---:|---:|
| `01.1_supplementary_stability_and_glucose_ablation.ipynb` | 61 | 8 | 1 | 9 |
| `01_modeling_clustering.ipynb` | 0 | 0 | 0 | 0 |
| `02.1_naive_bayes_followup_binary_xgboost_no_glucose_fixed.ipynb` | 56 | 24 | 1 | 26 |
| `02.2_naive_bayes_followup_raw_xgboost_no_glucose_no_missing_categories.ipynb` | 56 | 24 | 1 | 26 |
| `02_clusters_fusionados.ipynb` | 69 | 45 | 0 | 45 |
| `02_clusters_fusionados_sin_GLU_SUERO.ipynb` | 82 | 45 | 0 | 45 |
| `02_naive_bayes_followup.ipynb` | 48 | 24 | 1 | 26 |
| `03_naive_bayes_filtrado_vs_sin_filtro_no_cv.ipynb` | 28 | 6 | 0 | 6 |
| `04_naive_bayes_best_bins.ipynb` | 44 | 9 | 0 | 9 |

## Integrity

`NOTEBOOK_SHA256SUMS.txt` continues to refer to the preserved notebooks in `notebooks/`. `GITHUB_NOTEBOOK_SHA256SUMS.txt` records the hashes of the display-only copies. The two sets are intentionally different wherever output MIME bundles were converted.
