"""Typed configuration objects for the ENSANUT HbA1c workflow.

The dimensionality workflow builds one fuzzy-kNN graph from the configured PCA space. Leiden,
UMAP2D and PHATE2D all consume that exact graph. No separate UMAP clustering representation and
no post-UMAP kNN graph are constructed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PCAConfig:
    n_components: int = 50
    whiten: bool = False
    svd_solver: str = "randomized"
    tol: float = 0.0
    iterated_power: int | str = "auto"
    n_oversamples: int = 10
    power_iteration_normalizer: str = "auto"
    copy: bool = True
    random_state: int = 42


@dataclass(slots=True)
class UMAPGraphConfig:
    """Parameters that construct the single fuzzy-kNN graph from the configured PCA space.

    These parameters determine the graph used by Leiden, UMAP2D and PHATE2D.
    Layout parameters such as ``min_dist`` are intentionally excluded because
    they do not change the high-dimensional graph.
    """

    n_neighbors: int = 30
    metric: str = "euclidean"
    metric_kwds: dict[str, Any] | None = None
    local_connectivity: float = 10.0
    set_op_mix_ratio: float = 1.0
    angular_rp_forest: bool = False
    force_approximation_algorithm: bool = False
    unique: bool = False
    low_memory: bool = True
    disconnection_distance: float | None = None
    random_state: int = 42
    transform_seed: int = 42
    n_jobs: int = 1
    verbose: bool = False


@dataclass(slots=True)
class UMAPEmbeddingConfig:
    """UMAP2D layout parameters applied to the already constructed graph."""

    n_components: int = 2
    min_dist: float = 1.0
    spread: float = 1.0
    learning_rate: float = 1.0
    init: str = "spectral"
    n_epochs: int | None = None
    repulsion_strength: float = 1.0
    negative_sample_rate: int = 5
    output_metric: str = "euclidean"
    output_metric_kwds: dict[str, Any] | None = None
    random_state: int = 42
    verbose: bool = False


# Compatibility alias for external code that imported UMAPConfig.
UMAPConfig = UMAPEmbeddingConfig


@dataclass(slots=True)
class KNNGraphConfig:
    """Legacy compatibility class; not used by the organized pipeline.

    Leiden must consume the shared internal UMAP graph directly.
    """

    n_neighbors: int = 20
    metric: str = "euclidean"
    algorithm: str = "auto"
    leaf_size: int = 30
    p: float = 2.0
    n_jobs: int = 1
    weight_mode: str = "inverse_distance"
    gaussian_sigma: float | None = None
    symmetrize: str = "union_max"
    distance_epsilon: float = 1e-12


@dataclass(slots=True)
class LeidenConfig:
    partition_type: str = "RBConfiguration"  # RBConfiguration | CPM | Modularity
    resolution: float = 0.2
    n_iterations: int = -1
    max_comm_size: int = 0
    seed: int = 42


@dataclass(slots=True)
class PHATEConfig:
    """PHATE 2.0 parameters for embedding the shared UMAP fuzzy graph.

    The input is already the single fuzzy-kNN graph built from the configured PCA space. Therefore
    PHATE never calculates a second neighborhood graph. Native PHATE graph
    parameters are still exposed for transparency and compatibility with its
    estimator API, but ``knn_dist`` must remain ``precomputed_affinity`` in this
    workflow.
    """

    # Output and PHATE-native graph controls.
    n_components: int = 2
    knn: int = 30
    decay: int | None = 40
    n_landmark: int | None = 2000

    # Diffusion-potential controls.
    t: int | str = 10
    gamma: float = 1.0
    n_pca: int | None = None

    # Distance and MDS controls.
    mds_solver: str = "sgd"
    knn_dist: str = "precomputed_affinity"
    knn_max: int | None = None
    mds_dist: str = "euclidean"
    mds: str = "metric"

    # Execution and landmark controls.
    n_jobs: int = 1
    random_state: int = 42
    random_landmarking: bool = False
    verbose: int = 1

    # PHATE.transform controls; relevant when t="auto".
    t_max: int = 100
    plot_optimal_t: bool = False

    # Advanced graphtools.Graph keyword arguments accepted by PHATE.
    graph_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"thresh": 1e-4}
    )


@dataclass(slots=True)
class ReductionPipelineConfig:
    pca: PCAConfig = field(default_factory=PCAConfig)
    shared_umap_graph: UMAPGraphConfig = field(default_factory=UMAPGraphConfig)
    umap2d: UMAPEmbeddingConfig = field(
        default_factory=lambda: UMAPEmbeddingConfig(
            n_components=2,
            min_dist=1.0,
            spread=1.0,
        )
    )
    leiden: LeidenConfig = field(default_factory=LeidenConfig)
    phate: PHATEConfig = field(default_factory=PHATEConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
