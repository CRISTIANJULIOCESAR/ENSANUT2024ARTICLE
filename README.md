# ENSANUT 2024 HbA1c explainable communities

Reproducible analysis code for an explainable machine-learning workflow built around **XGBoost HbA1c regression, participant-level SHAP profiles, graph-based Leiden communities, hierarchical signed evidence, and downstream community-recoverability / sensitivity analyses** in ENSANUT 2024.

The repository keeps the analysis notebooks as the scientific record and places reusable implementations in `src/ensanut_hba1c/`.

## Repository map

```text
ENSANUT_HbA1c_explainable_communities/
├── README.md
├── data/
│   ├── input/                  # PUT THE TWO INPUT CSV FILES HERE
│   ├── interim/                # generated; Git-ignored
│   └── processed/              # generated; Git-ignored
├── notebooks/
│   ├── 01_modeling_clustering.ipynb
│   ├── 01.1_supplementary_stability_and_glucose_ablation.ipynb
│   ├── 02_naive_bayes_followup.ipynb
│   ├── 02.1_naive_bayes_followup_binary_xgboost_no_glucose_fixed.ipynb
│   ├── 02.2_naive_bayes_followup_raw_xgboost_no_glucose_no_missing_categories.ipynb
│   ├── 02_clusters_fusionados.ipynb
│   ├── 02_clusters_fusionados_sin_GLU_SUERO.ipynb
│   ├── 03_naive_bayes_filtrado_vs_sin_filtro_no_cv.ipynb
│   └── 04_naive_bayes_best_bins.ipynb
├── src/ensanut_hba1c/          # local reusable analysis package
├── scripts/
│   ├── validate_project.py
│   └── run_pipeline.py         # canonical Notebook 01 → Notebook 02 path
├── docs/
│   ├── NOTEBOOK_GUIDE.md
│   └── REPRODUCIBILITY.md
├── tests/
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## 1. Input data: exact location

Place these **two files with these exact names** in `data/input/`:

```text
data/input/ENSANUT_2024_mx.csv
data/input/ENSANUT_2024.csv
```

- `ENSANUT_2024_mx.csv`: participant-level analytical matrix.
- `ENSANUT_2024.csv`: variable/data dictionary used to recover questionnaire hierarchy and descriptions.

The repository intentionally does not include these data files, and `.gitignore` prevents accidental commits from `data/input/`. See [`data/input/README.md`](data/input/README.md).

## 2. Main workflow

The canonical path is:

```text
ENSANUT inputs
      │
      ▼
01_modeling_clustering.ipynb
  XGBoost HbA1c regression
  → SHAP participant profiles
  → PCA / shared UMAP fuzzy graph
  → Leiden communities
      │
      ├── results/01_modeling_clustering/
      └── handoff_to_notebook2/
                 │
                 ▼
02_naive_bayes_followup.ipynb
  hierarchical signed evidence
  → questionnaire-domain summaries
  → supervised evidence heatmap
  → label-recoverability audit
```

The canonical handoff is created automatically at:

```text
results/01_modeling_clustering/handoff_to_notebook2/
```

## 3. What each notebook does

| Notebook | What it is for |
|---|---|
| **01_modeling_clustering** | Primary HbA1c regression → SHAP → graph → Leiden pipeline and downstream handoff. |
| **01.1 supplementary stability/glucose ablation** | 20-seed Leiden stability, resolution sensitivity, glucose-only comparison, and complete no-direct-glycemia reconstruction. |
| **02 Naive Bayes follow-up** | Canonical hierarchical signed-evidence characterization and NB-filtered XGBoost community-label recoverability. |
| **02.1 binary no-glucose** | Recoverability audit using a non-glycemic binary rule matrix. |
| **02.2 raw no-glucose / no missing categories** | Recoverability audit using raw prepared source variables, excluding glycemic predictors and textual missing categories. |
| **02 merged C5–C8** | Sensitivity analysis comparing original communities with a C5–C8 merged solution. |
| **02 merged C5–C8, no GLU_SUERO** | Same merger sensitivity analysis with `GLU_SUERO` excluded from evidence and XGBoost. |
| **03 filtered vs unfiltered NB** | Apparent/in-sample comparison of NB models with and without the evidence filter; no CV. |
| **04 best bins** | Extends Notebook 03 with best-bin selection and a third NB model. |

For prerequisites and execution order, see [`docs/NOTEBOOK_GUIDE.md`](docs/NOTEBOOK_GUIDE.md).

## 4. Environment

### Conda

```bash
conda env create -f environment.yml
conda activate ensanut-hba1c
pip install -e .
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Python 3.12 is the target environment.

## 5. Validate the clone

Before adding data:

```bash
python scripts/validate_project.py
```

After placing the two CSV files in `data/input/`:

```bash
python scripts/validate_project.py --require-data
```

## 6. Run

Interactive execution is preferred because the notebooks expose the scientific parameters directly:

```bash
jupyter lab
```

Run the canonical pair in order:

```text
notebooks/01_modeling_clustering.ipynb
notebooks/02_naive_bayes_followup.ipynb
```

Or execute only the canonical two-notebook path non-interactively:

```bash
python scripts/run_pipeline.py
```

The complementary notebooks should be run according to [`docs/NOTEBOOK_GUIDE.md`](docs/NOTEBOOK_GUIDE.md), rather than blindly executing every variant in sequence.

## 7. Results and Git hygiene

Generated outputs go under `results/` and are ignored by Git. Input CSV/XLSX/PKL files under `data/input/` are also ignored. This prevents accidental upload of source data, models, intermediate artifacts, and large result files.

The repository includes `NOTEBOOK_SHA256SUMS.txt` and `ORIGINAL_UPLOAD_MAP.md` so the preserved analysis notebooks can be audited.
