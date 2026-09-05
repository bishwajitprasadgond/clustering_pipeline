"""Public API for the clustering pipeline package.

Re-exports the pieces callers need so they can write
``from SCRIPTS import ClusteringPipeline, ClusteringConfig`` instead of
reaching into the ``clustering_pipeline`` submodule directly.
"""

from .clustering_pipeline import (
    ClusteringAlgorithm,
    ClusteringConfig,
    ClusteringConfigError,
    ClusteringDataError,
    ClusteringPipeline,
    ClusteringResult,
)

__all__ = [
    "ClusteringAlgorithm",
    "ClusteringConfig",
    "ClusteringConfigError",
    "ClusteringDataError",
    "ClusteringPipeline",
    "ClusteringResult",
]
