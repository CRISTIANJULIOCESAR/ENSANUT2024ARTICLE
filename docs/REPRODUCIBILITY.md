# Reproducibility notes

## What is version-controlled

- all analysis notebooks;
- the local `ensanut_hba1c` Python package used by the notebooks;
- environment/dependency files;
- validation and execution helpers;
- documentation of expected input and intermediate files.

## What is intentionally not version-controlled

- ENSANUT input CSV files;
- generated results, figures, models, workbooks, pickles, NumPy arrays, and Optuna databases;
- executed notebook copies.

This separation keeps the repository small and avoids accidentally committing data or large generated artifacts.

## Notebook integrity

`NOTEBOOK_SHA256SUMS.txt` records the SHA-256 hash of each notebook in the packaged repository. `ORIGINAL_UPLOAD_MAP.md` maps the eight supplied filenames to their cleaned repository filenames. The notebook bytes themselves were not rewritten.

## Validation

Without data:

```bash
python scripts/validate_project.py
```

With the two CSV files present:

```bash
python scripts/validate_project.py --require-data
```

The validator checks the repository structure, Python source syntax, notebook JSON/code syntax, and—when requested—the required input files and core columns.
