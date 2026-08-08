#!/usr/bin/env python3
"""Execute the canonical Notebook 01 -> Notebook 02 workflow only."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
EXECUTED_DIR = PROJECT_ROOT / "results" / "executed_notebooks"
NOTEBOOKS = [
    NOTEBOOK_DIR / "01_modeling_clustering.ipynb",
    NOTEBOOK_DIR / "02_naive_bayes_followup.ipynb",
]


def execute_notebook(path: Path, timeout: int) -> Path:
    print(f"\nExecuting {path.name}")
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    try:
        client.execute(cwd=str(PROJECT_ROOT))
    except CellExecutionError:
        failed_path = EXECUTED_DIR / f"FAILED_{path.name}"
        EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, failed_path)
        print("Execution failed. Partial notebook saved to:", failed_path)
        raise
    output_path = EXECUTED_DIR / path.name
    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    print("Executed notebook saved to:", output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=86400, help="Maximum seconds allowed per cell.")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--notebook", choices=["1", "2", "all"], default="all")
    args = parser.parse_args()

    if not args.skip_validation:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_project.py"), "--require-data"],
            check=True,
            cwd=PROJECT_ROOT,
        )

    selected = NOTEBOOKS
    if args.notebook == "1":
        selected = NOTEBOOKS[:1]
    elif args.notebook == "2":
        selected = NOTEBOOKS[1:]

    for notebook_path in selected:
        execute_notebook(notebook_path, timeout=args.timeout)
    print("\nCanonical pipeline execution completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
