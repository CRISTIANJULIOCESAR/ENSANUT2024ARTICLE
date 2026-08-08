"""Orchestrate PCA, UMAP2D, PHATE2D and Leiden from one UMAP graph_."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .clustering import run_leiden, umap_graph_to_igraph
from .config import ReductionPipelineConfig
from .dimensionality import (
    embed_shared_umap_graph,
    fit_pca20,
    fit_phate_from_umap_graph,
    fit_shared_umap_graph,
)


def build_representation_table(
    pca_scores: np.ndarray,
    umap2d: np.ndarray,
    phate2d: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Combine PCA views, graph-derived layouts and graph-derived clusters."""
    n = len(labels)
    arrays = [pca_scores, umap2d, phate2d]
    if any(np.asarray(array).shape[0] != n for array in arrays):
        raise ValueError("The representations do not preserve the same participant order.")
    if np.asarray(umap2d).shape[1] < 2 or np.asarray(phate2d).shape[1] < 2:
        raise ValueError("UMAP2D and PHATE2D must contain at least two columns.")

    result = pd.DataFrame({"Cluster_SHAP": np.asarray(labels, dtype=int)})
    for index in range(pca_scores.shape[1]):
        result[f"PCA_{index + 1}"] = pca_scores[:, index]
    result["UMAP_VISUAL_1"] = umap2d[:, 0]
    result["UMAP_VISUAL_2"] = umap2d[:, 1]
    result["UMAP_GRAPH_PHATE_1"] = phate2d[:, 0]
    result["UMAP_GRAPH_PHATE_2"] = phate2d[:, 1]
    return result


def _assert_same_graph_fingerprint(expected: str, branches: dict[str, str]) -> None:
    mismatches = {name: value for name, value in branches.items() if value != expected}
    if mismatches:
        raise RuntimeError(
            "Not all branches received the same internal UMAP graph_. "
            f"Expected={expected}; mismatches={mismatches}"
        )


def run_reduction_and_clustering(
    shap_matrix: np.ndarray,
    config: ReductionPipelineConfig,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run all views and Leiden from exactly one graph.

    SHAP -> configured PCA -> one shared UMAP internal fuzzy-kNN graph_
      - PCA1D = PC1
      - PCA2D = PC1/PC2
      - shared graph_ -> UMAP2D visualization
      - shared graph_ -> PHATE2D visualization
      - shared graph_ -> Leiden clustering

    There is no UMAP representation for clustering. Leiden operates directly
    on graph vertices, fuzzy edges and edge weights.
    """
    pca = fit_pca20(shap_matrix, config.pca)
    shared_graph = fit_shared_umap_graph(pca["scores"], config.shared_umap_graph)

    umap2d = embed_shared_umap_graph(
        pca["scores"], shared_graph["graph"], shared_graph["model"], config.umap2d
    )
    phate = fit_phate_from_umap_graph(shared_graph["graph"], config.phate)
    leiden_graph = umap_graph_to_igraph(shared_graph["graph"])
    leiden = run_leiden(leiden_graph["graph"], config.leiden)

    expected_fingerprint = shared_graph["graph_fingerprint"]
    _assert_same_graph_fingerprint(expected_fingerprint, {
        "UMAP2D": umap2d["graph_fingerprint"],
        "PHATE2D": phate["graph_fingerprint"],
        "Leiden": leiden_graph["graph_fingerprint"],
    })

    table = build_representation_table(
        pca["scores"],
        umap2d["coordinates"],
        phate["phate2d"],
        leiden["labels"],
    )
    results = {
        "pca": pca,
        "shared_umap_graph": shared_graph,
        "umap2d": umap2d,
        "phate": phate,
        "leiden_graph": leiden_graph,
        "leiden": leiden,
        "table": table,
    }
    if output_dir is not None:
        export_reduction_results(results, output_dir, config)
    return results


def export_reduction_results(
    results: dict[str, Any],
    output_dir: str | Path,
    config: ReductionPipelineConfig,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "pca_scores.npy", results["pca"]["scores"])
    # Legacy alias retained for downstream code created before PCA became configurable.
    np.save(output / "pca20_scores.npy", results["pca"]["scores"])
    np.save(output / "pca1d.npy", results["pca"]["pca1d"])
    np.save(output / "pca2d.npy", results["pca"]["pca2d"])
    np.save(output / "umap2d_from_shared_graph.npy", results["umap2d"]["coordinates"])
    np.save(output / "phate2d_from_shared_graph.npy", results["phate"]["phate2d"])
    np.save(output / "leiden_labels_from_shared_graph.npy", results["leiden"]["labels"])
    sparse.save_npz(
        output / "shared_umap_internal_graph.npz",
        results["shared_umap_graph"]["graph"],
    )
    results["table"].to_csv(output / "all_representations_and_leiden.csv", index=False)

    variance = np.asarray(results["pca"]["explained_variance_ratio"])
    variance_table = pd.DataFrame({
        "component": np.arange(1, len(variance) + 1),
        "explained_variance_ratio": variance,
        "explained_variance_percentage": 100 * variance,
        "cumulative_explained_variance_ratio": np.cumsum(variance),
        "cumulative_explained_variance_percentage": 100 * np.cumsum(variance),
    })
    variance_table.to_csv(output / "pca_explained_variance.csv", index=False)
    # Legacy alias retained for compatibility.
    variance_table.to_csv(output / "pca20_explained_variance.csv", index=False)

    graph = results["shared_umap_graph"]["graph"]
    fingerprint = results["shared_umap_graph"]["graph_fingerprint"]
    pd.DataFrame([{
        "graph_fingerprint_sha256": fingerprint,
        "n_nodes": graph.shape[0],
        "undirected_edges": results["leiden_graph"]["n_edges"],
        "nonzero_entries": graph.nnz,
        "density": graph.nnz / (graph.shape[0] * graph.shape[1]),
        "minimum_nonzero_weight": float(graph.data.min()),
        "maximum_weight": float(graph.data.max()),
        "symmetric": bool((graph != graph.T).nnz == 0),
        "source_space": f"PCA{results['pca']['effective_components']}",
        "graph_type": "one UMAP internal fuzzy-kNN graph_",
        "used_for_umap2d": True,
        "used_for_phate2d": True,
        "used_for_leiden": True,
        "leiden_input": "shared_graph_directly",
        "clustering_embedding_created": False,
        "second_knn_graph_created": False,
    }]).to_csv(output / "shared_umap_graph_audit.csv", index=False)

    manifest = config.to_dict()
    manifest["effective"] = {
        "pca_components": results["pca"]["effective_components"],
        "shared_umap_graph_neighbors": results["shared_umap_graph"]["effective_neighbors"],
        "shared_graph_fingerprint_sha256": fingerprint,
        "umap2d_components": results["umap2d"]["effective_components"],
        "leiden_input": "shared UMAP internal graph_ directly",
        "leiden_graph_edges": results["leiden_graph"]["n_edges"],
        "leiden_clusters": results["leiden"]["n_clusters"],
        "leiden_modularity": results["leiden"]["modularity"],
        "phate_landmarks": results["phate"]["effective_landmarks"],
        "phate_selected_t": results["phate"]["selected_t"],
    }
    (output / "method_parameters.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
