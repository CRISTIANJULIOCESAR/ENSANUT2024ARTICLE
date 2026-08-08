"""Leiden clustering directly on the shared internal UMAP fuzzy graph."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from scipy import sparse

from .config import KNNGraphConfig, LeidenConfig
from .dimensionality import graph_fingerprint


def umap_graph_to_igraph(umap_graph: sparse.spmatrix) -> dict[str, Any]:
    """Convert one UMAP fuzzy adjacency matrix to igraph without new kNN.

    Every nonzero upper-triangular fuzzy edge is retained with its original
    UMAP membership strength as the Leiden edge weight.
    """
    import igraph as ig

    graph_csr = sparse.csr_matrix(umap_graph, dtype=np.float64)
    graph_csr = graph_csr.maximum(graph_csr.T).tocsr()
    graph_csr.setdiag(0.0)
    graph_csr.eliminate_zeros()
    graph_csr.sort_indices()
    upper = sparse.triu(graph_csr, k=1, format="coo")
    if upper.nnz == 0:
        raise RuntimeError("El graph_ interno de UMAP no contiene aristas para Leiden.")
    edges = list(zip(upper.row.astype(int), upper.col.astype(int)))
    graph = ig.Graph(n=graph_csr.shape[0], edges=edges, directed=False)
    graph.es["weight"] = upper.data.astype(float).tolist()
    return {
        "graph": graph,
        "n_edges": int(graph.ecount()),
        "graph_fingerprint": graph_fingerprint(graph_csr),
        "source": "shared UMAP internal fuzzy graph_; direct shared graph; no clustering embedding or second kNN",
    }


def run_leiden(graph, config: LeidenConfig) -> dict[str, Any]:
    import leidenalg as la

    partition_name = str(config.partition_type).lower()
    if partition_name in {"rbconfiguration", "rb", "rb_configuration"}:
        partition_class = la.RBConfigurationVertexPartition
        kwargs = {"resolution_parameter": float(config.resolution)}
    elif partition_name == "cpm":
        partition_class = la.CPMVertexPartition
        kwargs = {"resolution_parameter": float(config.resolution)}
    elif partition_name in {"modularity", "modularityvertexpartition"}:
        partition_class = la.ModularityVertexPartition
        kwargs = {}
    else:
        raise ValueError(f"partition_type no reconocido: {config.partition_type}")

    partition = la.find_partition(
        graph,
        partition_class,
        weights=graph.es["weight"],
        n_iterations=int(config.n_iterations),
        max_comm_size=int(config.max_comm_size),
        seed=int(config.seed),
        **kwargs,
    )
    labels = np.asarray(partition.membership, dtype=int)
    modularity = float(graph.modularity(labels.tolist(), weights=graph.es["weight"]))
    return {
        "partition": partition,
        "labels": labels,
        "modularity": modularity,
        "n_clusters": int(np.unique(labels).size),
        "config": asdict(config),
    }


# ---------------------------------------------------------------------------
# Legacy compatibility only. It is deliberately excluded from the organized
# pipeline because it would create a second, different neighborhood graph.
# ---------------------------------------------------------------------------
def build_knn_igraph(data: np.ndarray, config: KNNGraphConfig) -> dict[str, Any]:
    raise RuntimeError(
        "build_knn_igraph is disabled in the organized workflow. "
        "Leiden must use umap_graph_to_igraph(shared_umap_graph)."
    )
