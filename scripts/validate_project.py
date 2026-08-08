#!/usr/bin/env python3
"""Validate repository structure, source syntax, notebooks, and optional inputs."""
from __future__ import annotations
import argparse
import ast
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REQUIRED_NOTEBOOKS = ['01_modeling_clustering.ipynb', '01.1_supplementary_stability_and_glucose_ablation.ipynb', '02_naive_bayes_followup.ipynb', '02.1_naive_bayes_followup_binary_xgboost_no_glucose_fixed.ipynb', '02.2_naive_bayes_followup_raw_xgboost_no_glucose_no_missing_categories.ipynb', '02_clusters_fusionados.ipynb', '02_clusters_fusionados_sin_GLU_SUERO.ipynb', '03_naive_bayes_filtrado_vs_sin_filtro_no_cv.ipynb', '04_naive_bayes_best_bins.ipynb']
REQUIRED_PATHS = [
    "src/ensanut_hba1c/__init__.py",
    "src/ensanut_hba1c/modeling.py",
    "src/ensanut_hba1c/io.py",
    "src/ensanut_hba1c/paths.py",
    "src/ensanut_hba1c/config.py",
    "src/ensanut_hba1c/dimensionality.py",
    "src/ensanut_hba1c/clustering.py",
    "src/ensanut_hba1c/reduction_pipeline.py",
    "src/ensanut_hba1c/bayes/analysis.py",
    "src/ensanut_hba1c/bayes/heatmaps.py",
    "src/ensanut_hba1c/bayes/hierarchy.py",
    "src/ensanut_hba1c/workflows/nb_evidence.py",
    "src/ensanut_hba1c/workflows/nb_feature_selection.py",
    "src/ensanut_hba1c/workflows/cluster_dataset.py",
    "src/ensanut_hba1c/workflows/supervised_heatmap.py",
    "src/ensanut_hba1c/workflows/progress.py",
    "src/ensanut_hba1c/workflows/xgboost_engine.py",
    "src/ensanut_hba1c/workflows/xgboost_reporting.py",
    "data/input", "results",
] + [f"notebooks/{name}" for name in REQUIRED_NOTEBOOKS]

def validate_source():
    errors=[]
    for path in sorted((PROJECT_ROOT/"src").rglob("*.py")):
        try: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as e: errors.append(f"{path.relative_to(PROJECT_ROOT)}: {type(e).__name__}: {e}")
    return errors

def validate_notebooks():
    errors=[]
    for name in REQUIRED_NOTEBOOKS:
        path=PROJECT_ROOT/"notebooks"/name
        try: nb=json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{name}: invalid JSON: {e}")
            continue
        for i,cell in enumerate(nb.get("cells",[])):
            if cell.get("cell_type")!="code": continue
            source="".join(cell.get("source",[]))
            if not source.strip(): continue
            try: compile(source, f"{name}:cell-{i}", "exec")
            except SyntaxError as e: errors.append(f"{name}, cell {i}: {e}")
    return errors

def validate_data(require_data):
    errors=[]
    input_dir=PROJECT_ROOT/"data"/"input"
    required=[input_dir/"ENSANUT_2024_mx.csv", input_dir/"ENSANUT_2024.csv"]
    missing=[p for p in required if not p.is_file()]
    if missing and require_data:
        errors.append("Missing input files: "+", ".join(str(p.relative_to(PROJECT_ROOT)) for p in missing))
    if not missing:
        import pandas as pd
        data=pd.read_csv(required[0], nrows=5, low_memory=False)
        dictionary=pd.read_csv(required[1], nrows=5, low_memory=False)
        missing_core=sorted({"FOLIO_I","FOLIO_INT","HB1AC"}.difference(data.columns))
        if missing_core: errors.append("Analytical dataset is missing core columns: "+", ".join(missing_core))
        if not any(c in dictionary.columns for c in ("var","variable","Variable","name")):
            errors.append("Dictionary does not contain a recognized variable-code column.")
    return errors

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--require-data", action="store_true")
    args=parser.parse_args()
    errors=[]
    missing=[x for x in REQUIRED_PATHS if not (PROJECT_ROOT/x).exists()]
    if missing: errors.append("Missing project paths: "+", ".join(missing))
    errors += validate_source() + validate_notebooks() + validate_data(args.require_data)
    if errors:
        print("PROJECT VALIDATION FAILED")
        for e in errors: print("-",e)
        return 1
    print("PROJECT VALIDATION PASSED")
    print("Project root:", PROJECT_ROOT)
    print("Notebooks checked:", len(REQUIRED_NOTEBOOKS))
    print("Input directory:", PROJECT_ROOT/"data"/"input")
    if not args.require_data:
        print("Data presence was not required for this validation run.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
