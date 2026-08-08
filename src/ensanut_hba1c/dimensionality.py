"""PCA and graph-derived views from one internal UMAP fuzzy-kNN graph.

The workflow is graph-first:

    SHAP -> configured PCA -> one UMAP internal graph_
                         |-> UMAP2D visualization
                         |-> PHATE2D visualization
                         `-> Leiden clustering (directly on the graph)

PCA1D and PCA2D are slices of the configured PCA representation. UMAP2D is optimized from the shared graph;
PHATE2D consumes the same graph as a precomputed adjacency. No clustering
embedding and no second nearest-neighbor graph are created.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

from .config import PCAConfig, PHATEConfig, UMAPEmbeddingConfig, UMAPGraphConfig


def _validate_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if array.shape[0] < 4 or array.shape[1] < 2:
        raise ValueError(f"{name} requires at least 4 rows and 2 columns.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contiene NaN o infinitos.")
    return array


def _clean_shared_graph(graph: sparse.spmatrix, n_samples: int) -> sparse.csr_matrix:
    cleaned = sparse.csr_matrix(graph, dtype=np.float64)
    # UMAP graph_ is a fuzzy simplicial set and should already be symmetric.
    # maximum() only removes minor sparse-format asymmetries without inventing
    # any new neighborhood calculation.
    cleaned = cleaned.maximum(cleaned.T).tocsr()
    cleaned.setdiag(0.0)
    cleaned.eliminate_zeros()
    cleaned.sort_indices()
    if cleaned.shape != (n_samples, n_samples) or cleaned.nnz == 0:
        raise RuntimeError("The shared internal graph_ is empty or has an invalid shape.")
    if not np.isfinite(cleaned.data).all() or np.any(cleaned.data < 0):
        raise RuntimeError("The shared internal graph_ contains invalid weights.")
    return cleaned


def graph_fingerprint(graph: sparse.spmatrix) -> str:
    """Stable SHA-256 identifier proving that every branch uses one graph."""
    csr = sparse.csr_matrix(graph, dtype=np.float64)
    csr.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
    digest.update(csr.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(csr.indices.astype(np.int64, copy=False).tobytes())
    digest.update(csr.data.astype(np.float64, copy=False).tobytes())
    return digest.hexdigest()


def fit_pca20(matrix: np.ndarray, config: PCAConfig) -> dict[str, Any]:
    """Fit one PCA model and expose PCA1D/PCA2D as slices of that model."""
    array = _validate_matrix(matrix, "SHAP")
    n_samples, n_features = array.shape
    effective_components = max(2, min(int(config.n_components), n_features, n_samples - 1))
    model = PCA(
        n_components=effective_components,
        copy=bool(config.copy),
        whiten=bool(config.whiten),
        svd_solver=str(config.svd_solver),
        tol=float(config.tol),
        iterated_power=config.iterated_power,
        n_oversamples=int(config.n_oversamples),
        power_iteration_normalizer=str(config.power_iteration_normalizer),
        random_state=int(config.random_state),
    )
    scores = np.asarray(model.fit_transform(array), dtype=float)
    return {
        "model": model,
        "scores": scores,
        "pca1d": scores[:, :1].copy(),
        "pca2d": scores[:, :2].copy(),
        "effective_components": effective_components,
        "explained_variance_ratio": np.asarray(model.explained_variance_ratio_, dtype=float),
        "config": asdict(config),
    }


def fit_shared_umap_graph(
    pca_scores: np.ndarray,
    config: UMAPGraphConfig,
) -> dict[str, Any]:
    """Construct UMAP's internal fuzzy-kNN graph once from the configured PCA space.

    ``transform_mode='graph'`` tells UMAP to stop after constructing the
    high-dimensional fuzzy simplicial set. No low-dimensional coordinates are used to build this graph.
    """
    import umap.umap_ as umap

    scores = _validate_matrix(pca_scores, "PCA scores")
    effective_neighbors = max(2, min(int(config.n_neighbors), len(scores) - 1))
    kwargs: dict[str, Any] = dict(
        n_components=2,  # irrelevant in transform_mode='graph'
        n_neighbors=effective_neighbors,
        metric=str(config.metric),
        metric_kwds=config.metric_kwds,
        local_connectivity=float(config.local_connectivity),
        set_op_mix_ratio=float(config.set_op_mix_ratio),
        angular_rp_forest=bool(config.angular_rp_forest),
        force_approximation_algorithm=bool(config.force_approximation_algorithm),
        unique=bool(config.unique),
        low_memory=bool(config.low_memory),
        random_state=int(config.random_state),
        transform_seed=int(config.transform_seed),
        n_jobs=int(config.n_jobs),
        transform_mode="graph",
        verbose=bool(config.verbose),
    )
    if config.disconnection_distance is not None:
        kwargs["disconnection_distance"] = float(config.disconnection_distance)

    model = umap.UMAP(**kwargs)
    model.fit(scores)
    if not hasattr(model, "graph_"):
        raise RuntimeError("UMAP did not expose graph_ after graph construction.")
    graph = _clean_shared_graph(model.graph_, len(scores))
    fingerprint = graph_fingerprint(graph)
    return {
        "model": model,
        "graph": graph,
        "graph_fingerprint": fingerprint,
        "effective_neighbors": effective_neighbors,
        "config": asdict(config),
        "source_space": f"PCA{scores.shape[1]}",
        "graph_origin": f"one UMAP internal fuzzy-kNN graph_ built once from PCA{scores.shape[1]}",
    }


def embed_shared_umap_graph(
    pca_scores: np.ndarray,
    shared_graph: sparse.spmatrix,
    graph_model: Any,
    config: UMAPEmbeddingConfig,
) -> dict[str, Any]:
    """Optimize a UMAP layout from the existing graph without rebuilding kNN."""
    import umap.distances as umap_distances
    import umap.umap_ as umap

    scores = _validate_matrix(pca_scores, "PCA scores")
    graph = _clean_shared_graph(shared_graph, len(scores))
    effective_components = max(2, min(int(config.n_components), len(scores) - 2))
    a, b = umap.find_ab_params(float(config.spread), float(config.min_dist))

    output_metric_name = str(config.output_metric)
    if output_metric_name not in umap_distances.named_distances_with_gradients:
        raise ValueError(
            "output_metric must have a gradient in umap.distances; "
            f"unrecognized metric: {output_metric_name}"
        )
    output_distance = umap_distances.named_distances_with_gradients[output_metric_name]
    output_metric_kwds = config.output_metric_kwds or {}

    embedding, aux_data = umap.simplicial_set_embedding(
        scores,
        graph.copy(),
        effective_components,
        float(config.learning_rate),
        float(a),
        float(b),
        float(config.repulsion_strength),
        int(config.negative_sample_rate),
        config.n_epochs,
        str(config.init),
        check_random_state(int(config.random_state)),
        graph_model._input_distance_func,
        graph_model._metric_kwds,
        False,  # densMAP is intentionally disabled for graph-reuse layouts
        {},
        False,
        output_distance,
        output_metric_kwds,
        output_metric_name in ("euclidean", "l2"),
        False,  # deterministic path because a random_state is fixed
        bool(config.verbose),
    )
    coordinates = np.asarray(embedding, dtype=float)
    if coordinates.shape != (len(scores), effective_components):
        raise RuntimeError(f"Forma UMAP inesperada: {coordinates.shape}")

    disconnected = np.asarray(graph.sum(axis=1)).ravel() == 0
    if disconnected.any():
        coordinates[disconnected] = np.nan

    return {
        "coordinates": coordinates,
        "aux_data": aux_data,
        "effective_components": effective_components,
        "config": asdict(config),
        "graph_fingerprint": graph_fingerprint(graph),
        "input": "the shared UMAP internal graph_; no kNN graph was recomputed",
    }


def fit_phate_from_umap_graph(
    umap_graph: sparse.spmatrix,
    config: PHATEConfig,
) -> dict[str, Any]:
    """Fit PHATE to the same precomputed fuzzy affinity used by UMAP2D/Leiden.

    No nearest-neighbor search is performed here. ``knn_dist`` is fixed to a
    precomputed affinity interpretation, and PHATE constructs only its diffusion
    operator, potential and final MDS layout from the supplied UMAP graph.
    """
    import graphtools
    import phate

    affinity = sparse.csr_matrix(umap_graph, dtype=np.float64)
    affinity = affinity.maximum(affinity.T).tocsr()
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    affinity.sort_indices()
    n_samples = affinity.shape[0]
    if affinity.shape[1] != n_samples or n_samples < 4 or affinity.nnz == 0:
        raise ValueError("The UMAP affinity must be square, non-empty, and contain at least 4 nodes.")
    if str(config.knn_dist) not in {"precomputed", "precomputed_affinity"}:
        raise ValueError(
            "PHATE_CONFIG.knn_dist must be 'precomputed_affinity' (or 'precomputed') "
            "to reuse the same UMAP graph_ without calculating new neighbors."
        )
    if not (-1.0 <= float(config.gamma) <= 1.0):
        raise ValueError("PHATE_CONFIG.gamma must be between -1 and 1.")
    if int(config.t_max) < 1:
        raise ValueError("PHATE_CONFIG.t_max must be >= 1.")

    # This Graph wraps the exact shared adjacency. It does not run kNN.
    graph = graphtools.Graph(affinity, precomputed="adjacency")
    effective_components = max(2, min(int(config.n_components), n_samples - 1))
    if config.n_landmark is None or int(config.n_landmark) >= n_samples:
        effective_landmarks = None
    else:
        effective_landmarks = max(1, int(config.n_landmark))

    operator = phate.PHATE(
        n_components=effective_components,
        knn=max(1, int(config.knn)),
        decay=None if config.decay is None else int(config.decay),
        n_landmark=effective_landmarks,
        t=config.t,
        gamma=float(config.gamma),
        n_pca=None if config.n_pca is None else int(config.n_pca),
        mds_solver=str(config.mds_solver),
        knn_dist=str(config.knn_dist),
        knn_max=None if config.knn_max is None else int(config.knn_max),
        mds_dist=str(config.mds_dist),
        mds=str(config.mds),
        n_jobs=int(config.n_jobs),
        random_state=int(config.random_state),
        random_landmarking=bool(config.random_landmarking),
        verbose=int(config.verbose),
        **dict(config.graph_kwargs),
    )
    operator.fit(graph)
    embedding = np.asarray(
        operator.transform(
            t_max=int(config.t_max),
            plot_optimal_t=bool(config.plot_optimal_t),
        ),
        dtype=float,
    )
    if embedding.shape != (n_samples, effective_components):
        raise RuntimeError(f"Forma PHATE inesperada: {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise RuntimeError("PHATE produjo coordenadas no finitas.")

    selected_t = operator.optimal_t if str(config.t).lower() == "auto" else config.t
    return {
        "operator": operator,
        "graphtools_graph": graph,
        "embedding": embedding,
        "phate2d": embedding[:, :2].copy(),
        "effective_components": effective_components,
        "effective_landmarks": effective_landmarks,
        "selected_t": selected_t,
        "config": asdict(config),
        "graph_fingerprint": graph_fingerprint(affinity),
        "input": "the same shared UMAP internal graph_ adjacency; no kNN was recomputed",
    }

