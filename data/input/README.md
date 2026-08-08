# Input data

The notebooks expect the two input CSV files **in this directory** and with these exact names:

```text
data/input/
├── ENSANUT_2024_mx.csv   # participant-level analytical matrix
└── ENSANUT_2024.csv      # variable/data dictionary
```

These data files are intentionally **not included in the repository** and are ignored by Git.

## Minimum analytical columns

The main workflow expects at least:

- `FOLIO_I` — household identifier;
- `FOLIO_INT` — participant identifier;
- `HB1AC` — HbA1c target used to define the analytical cohort;
- `GLU_SUERO` — used in analyses that explicitly compare full vs glucose-excluded models.

The dictionary must contain the variable codes and questionnaire hierarchy used by the hierarchical signed-evidence workflow.

## Generated handoff

Do **not** manually place the Notebook 01 handoff here. Running `notebooks/01_modeling_clustering.ipynb` creates it under:

```text
results/01_modeling_clustering/handoff_to_notebook2/
```

Typical handoff files are:

```text
cluster_dataset_for_naive_bayes.pkl
cluster_dataset_for_naive_bayes.csv
data_dictionary_for_naive_bayes.pkl
data_dictionary_for_naive_bayes.csv
handoff_metadata.json
```

Notebooks 03 and 04 additionally use the Notebook 02 output:

```text
results/02_naive_bayes/bayes_internal_operations_all_variables.xlsx
```
