"""Input loading, audit exports and Notebook-1 to Notebook-2 handoff."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


_CLUSTER_FILENAMES = (
    "cluster_dataset_for_naive_bayes.pkl",
    "cluster_dataset_for_naive_bayes.csv",
    "complete_cluster_dataset.pkl",
    "complete_cluster_dataset.csv",
)
_DICTIONARY_FILENAMES = (
    "data_dictionary_for_naive_bayes.pkl",
    "data_dictionary_for_naive_bayes.csv",
    "ENSANUT_2024.csv",
    "data_dictionary.csv",
)
_SEARCH_PRUNE_DIRS = {
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "site-packages",
}


def read_csv_robust(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"The input file does not exist: {path}")
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(path, low_memory=False, encoding=encoding, nrows=nrows)
            return frame.loc[:, ~frame.columns.duplicated()].copy()
        except UnicodeDecodeError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def load_ensanut_inputs(
    input_dir: str | Path,
    data_filename: str = "ENSANUT_2024_mx.csv",
    dictionary_filename: str = "ENSANUT_2024.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir)
    return (
        read_csv_robust(input_dir / data_filename),
        read_csv_robust(input_dir / dictionary_filename),
    )


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def export_notebook_handoff(
    cluster_df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    cluster_column: str = "Cluster_SHAP",
    source_notebook: str = "01_modeling_clustering.ipynb",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cluster_column not in cluster_df:
        raise KeyError(f"Missing required column: {cluster_column}")

    cluster_pickle = output_dir / "cluster_dataset_for_naive_bayes.pkl"
    cluster_csv = output_dir / "cluster_dataset_for_naive_bayes.csv"
    dictionary_pickle = output_dir / "data_dictionary_for_naive_bayes.pkl"
    dictionary_csv = output_dir / "data_dictionary_for_naive_bayes.csv"

    cluster_df.reset_index(drop=True).to_pickle(cluster_pickle)
    cluster_df.reset_index(drop=True).to_csv(cluster_csv, index=False)
    dictionary_df.reset_index(drop=True).to_pickle(dictionary_pickle)
    dictionary_df.reset_index(drop=True).to_csv(dictionary_csv, index=False)

    counts = (
        cluster_df[cluster_column]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis(cluster_column)
        .reset_index(name="n")
    )
    counts.to_csv(output_dir / "microcluster_counts.csv", index=False)

    metadata = {
        "source_notebook": source_notebook,
        "cluster_column": cluster_column,
        "n_participants": int(len(cluster_df)),
        "n_microclusters": int(cluster_df[cluster_column].nunique(dropna=True)),
        "participant_columns": int(cluster_df.shape[1]),
        "row_order_preserved": True,
        "files": {
            "cluster_pickle": cluster_pickle.name,
            "cluster_csv": cluster_csv.name,
            "dictionary_pickle": dictionary_pickle.name,
            "dictionary_csv": dictionary_csv.name,
        },
    }
    metadata_path = write_json(metadata, output_dir / "handoff_metadata.json")
    return {
        "cluster_pickle": cluster_pickle,
        "cluster_csv": cluster_csv,
        "dictionary_pickle": dictionary_pickle,
        "dictionary_csv": dictionary_csv,
        "metadata": metadata_path,
    }


def _deduplicate_paths(paths: Iterable[str | Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        path = Path(item).expanduser()
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path.absolute())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path).reset_index(drop=True)
    if path.suffix.lower() == ".csv":
        return read_csv_robust(path).reset_index(drop=True)
    raise ValueError(f"Unsupported handoff format: {path}")


def _read_first_available_table(
    paths: list[Path],
    *,
    required_any_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame | None, Path | None, list[str]]:
    errors: list[str] = []
    for path in _deduplicate_paths(paths):
        if not path.is_file():
            continue
        try:
            frame = _read_table(path)
        except Exception as error:  # pragma: no cover - depends on corrupted user files
            errors.append(f"{path}: {type(error).__name__}: {error}")
            continue
        if required_any_columns and not any(column in frame.columns for column in required_any_columns):
            errors.append(
                f"{path}: does not contain any expected column {list(required_any_columns)}"
            )
            continue
        return frame, path, errors
    return None, None, errors


def _walk_named_files(
    roots: Iterable[str | Path],
    filenames: tuple[str, ...],
    *,
    max_depth: int = 5,
) -> list[Path]:
    """Find known handoff filenames without traversing environments indefinitely."""
    matches: list[Path] = []
    wanted = set(filenames)
    for raw_root in _deduplicate_paths(roots):
        root = raw_root.expanduser()
        if root.is_file():
            if root.name in wanted:
                matches.append(root)
            continue
        if not root.is_dir():
            continue
        root = root.resolve()
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            try:
                depth = len(current_path.relative_to(root).parts)
            except ValueError:
                continue
            dirs[:] = [
                name
                for name in dirs
                if name not in _SEARCH_PRUNE_DIRS and not name.startswith(".")
            ]
            if depth >= max_depth:
                dirs[:] = []
            for filename in files:
                if filename in wanted:
                    matches.append(current_path / filename)
    # Newest files first within the user-specified root order.
    return sorted(
        _deduplicate_paths(matches),
        key=lambda path: path.stat().st_mtime if path.is_file() else -1,
        reverse=True,
    )


def discover_notebook_handoff_candidates(
    search_roots: Iterable[str | Path],
    *,
    max_depth: int = 5,
) -> dict[str, list[Path]]:
    """Discover handoff outputs in sibling/older ENSANUT project folders."""
    return {
        "cluster": _walk_named_files(
            search_roots,
            _CLUSTER_FILENAMES,
            max_depth=max_depth,
        ),
        "dictionary": _walk_named_files(
            search_roots,
            _DICTIONARY_FILENAMES,
            max_depth=max_depth,
        ),
    }


def _infer_project_root_from_file(path: Path) -> Path | None:
    """Infer the nearest ENSANUT project root from an output/input file path."""
    for parent in [path.parent, *path.parents]:
        if (parent / "results").is_dir() or (
            (parent / "src" / "ensanut_hba1c").is_dir()
            and (parent / "data" / "input").is_dir()
        ):
            return parent
    return None


def _dictionary_candidates_near_cluster(path: Path | None) -> list[Path]:
    if path is None:
        return []
    project_root = _infer_project_root_from_file(path)
    if project_root is None:
        return []
    return [
        project_root
        / "results"
        / "01_modeling_clustering"
        / "handoff_to_notebook2"
        / "data_dictionary_for_naive_bayes.pkl",
        project_root
        / "results"
        / "01_modeling_clustering"
        / "handoff_to_notebook2"
        / "data_dictionary_for_naive_bayes.csv",
        project_root / "data" / "input" / "ENSANUT_2024.csv",
        project_root / "data" / "input" / "data_dictionary.csv",
    ]


def load_notebook_handoff(
    handoff_dir: str | Path,
    *,
    fallback_cluster_paths: list[str | Path] | tuple[str | Path, ...] = (),
    fallback_dictionary_paths: list[str | Path] | tuple[str | Path, ...] = (),
    auto_discover_search_roots: list[str | Path] | tuple[str | Path, ...] = (),
    auto_discover_max_depth: int = 5,
    persist_recovered_handoff: bool = False,
    cluster_column: str = "Cluster_SHAP",
    cluster_column_aliases: tuple[str, ...] = (
        "Cluster_SHAP_original",
        "cluster_shap",
        "cluster",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load Notebook-1 outputs and optionally recover them from older folders.

    Resolution order:
    1. Current project's handoff pickle/CSV.
    2. Explicit fallback paths from the notebook.
    3. Known handoff filenames discovered under ``auto_discover_search_roots``.

    If ``persist_recovered_handoff`` is true and an external file is recovered,
    canonical pickle/CSV copies are written into the current project's handoff
    directory, so subsequent runs no longer depend on the old project folder.
    """
    handoff_dir = Path(handoff_dir)
    current_cluster_candidates = [
        handoff_dir / "cluster_dataset_for_naive_bayes.pkl",
        handoff_dir / "cluster_dataset_for_naive_bayes.csv",
    ]
    current_dictionary_candidates = [
        handoff_dir / "data_dictionary_for_naive_bayes.pkl",
        handoff_dir / "data_dictionary_for_naive_bayes.csv",
    ]
    cluster_candidates = [
        *current_cluster_candidates,
        *(Path(path) for path in fallback_cluster_paths),
    ]
    explicit_dictionary_candidates = [
        *current_dictionary_candidates,
        *(Path(path) for path in fallback_dictionary_paths),
    ]

    discovered = {"cluster": [], "dictionary": []}
    if auto_discover_search_roots:
        discovered = discover_notebook_handoff_candidates(
            auto_discover_search_roots,
            max_depth=auto_discover_max_depth,
        )
        cluster_candidates.extend(discovered["cluster"])

    expected_cluster_columns = (cluster_column, *cluster_column_aliases)
    cluster_df, cluster_source, cluster_errors = _read_first_available_table(
        cluster_candidates,
        required_any_columns=expected_cluster_columns,
    )

    # Prefer the dictionary from the same project as the recovered cluster.
    # This avoids pairing a cluster dataset with a newer but unrelated dictionary.
    dictionary_candidates = [
        *explicit_dictionary_candidates,
        *_dictionary_candidates_near_cluster(cluster_source),
        *discovered["dictionary"],
    ]
    dictionary_df, dictionary_source, dictionary_errors = _read_first_available_table(
        dictionary_candidates,
        required_any_columns=("var", "variable", "Variable", "name"),
    )

    missing_groups: list[str] = []
    if cluster_df is None:
        missing_groups.append(
            "dataset with Cluster_SHAP; checked: "
            + ", ".join(str(path) for path in _deduplicate_paths(cluster_candidates))
        )
    if dictionary_df is None:
        missing_groups.append(
            "ENSANUT dictionary; checked: "
            + ", ".join(str(path) for path in _deduplicate_paths(dictionary_candidates))
        )
    if missing_groups:
        details = [*cluster_errors, *dictionary_errors]
        detail_text = ""
        if details:
            detail_text = " Rejected files: " + " | ".join(details[:8])
        raise FileNotFoundError(
            "Notebook 2 could not be initialized. No usable handoff exists "
            "in this project and none was found in the search paths. "
            "Run Notebook 1 through the cell that prints "
            "'Automatic handoff ready for Notebook 2', or define the editable external paths "
            "at the beginning of Notebook 2. Missing: "
            + " | ".join(missing_groups)
            + detail_text
        )

    assert cluster_df is not None and dictionary_df is not None
    source_cluster_column = cluster_column
    if cluster_column not in cluster_df.columns:
        source_cluster_column = next(
            alias for alias in cluster_column_aliases if alias in cluster_df.columns
        )
        cluster_df = cluster_df.rename(columns={source_cluster_column: cluster_column})

    metadata_path = handoff_dir / "handoff_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "source_notebook": "01_modeling_clustering.ipynb",
            "cluster_column": cluster_column,
            "source_cluster_column": source_cluster_column,
            "n_participants": int(len(cluster_df)),
            "n_microclusters": int(cluster_df[cluster_column].nunique(dropna=True)),
            "participant_columns": int(cluster_df.shape[1]),
            "row_order_preserved": True,
            "metadata_reconstructed": True,
        }

    current_cluster_resolved = {
        str(path.resolve()) for path in current_cluster_candidates if path.exists()
    }
    current_dictionary_resolved = {
        str(path.resolve()) for path in current_dictionary_candidates if path.exists()
    }
    recovered_external = (
        cluster_source is not None
        and str(cluster_source.resolve()) not in current_cluster_resolved
    ) or (
        dictionary_source is not None
        and str(dictionary_source.resolve()) not in current_dictionary_resolved
    )

    materialized_paths: dict[str, Path] | None = None
    original_metadata_path = metadata_path if metadata_path.is_file() else None
    if persist_recovered_handoff and recovered_external:
        materialized_paths = export_notebook_handoff(
            cluster_df,
            dictionary_df,
            handoff_dir,
            cluster_column=cluster_column,
            source_notebook="02_naive_bayes_followup.ipynb (recovered external handoff)",
        )
        recovered_metadata = json.loads(
            materialized_paths["metadata"].read_text(encoding="utf-8")
        )
        recovered_metadata.update(
            {
                "recovered_external_handoff": True,
                "recovered_cluster_source": str(cluster_source),
                "recovered_dictionary_source": str(dictionary_source),
                "source_cluster_column": source_cluster_column,
            }
        )
        write_json(recovered_metadata, materialized_paths["metadata"])
        metadata = recovered_metadata
        metadata_path = materialized_paths["metadata"]

    sources: dict[str, Any] = {
        "cluster": cluster_source,
        "dictionary": dictionary_source,
        "metadata": metadata_path if metadata_path.is_file() else original_metadata_path,
        "recovered_external": recovered_external,
        "materialized_handoff": materialized_paths,
        "discovered_cluster_candidates": discovered["cluster"],
        "discovered_dictionary_candidates": discovered["dictionary"],
        "discarded_errors": [*cluster_errors, *dictionary_errors],
    }
    return cluster_df, dictionary_df, metadata, sources
