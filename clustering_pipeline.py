"""Run a clustering algorithm over a feature dataset and compute the metrics
expected by the ``dev_prediction_store`` Hive tables: ``clustering_runs``,
``cluster_metrics`` and ``cluster_assignments``.

This module does not talk to Hive/Impala directly. It takes a pandas
DataFrame of features in, runs the requested clustering algorithm, and
hands back three pandas DataFrames whose columns match the three tables
column-for-column and in the same order, so the caller can load them with
whatever mechanism is already used for that warehouse (Spark, impyla,
``INSERT`` statements, an ORC writer, etc.).

Requires: pandas, numpy, scipy, scikit-learn (>=1.3, for ``sklearn.cluster.HDBSCAN``).

Example
-------
>>> config = ClusteringConfig(
...     algorithm=ClusteringAlgorithm.KMEANS,
...     model_name="cust-segmentation-kmeans",
...     model_id="mdl-km-8821",
...     dataset_name="customer_feature_embeddings",
...     dataset_version="v2.4",
...     id_column="customer_id",
...     n_clusters=5,
...     random_seed=42,
... )
>>> result = ClusteringPipeline(config).run(features_df)
>>> result.run              # one row -> dev_prediction_store.clustering_runs
>>> result.cluster_metrics  # one row per cluster -> dev_prediction_store.cluster_metrics
>>> result.assignments      # one row per record -> dev_prediction_store.cluster_assignments
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, pairwise_distances, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# sklearn's own convention for "this point did not join a cluster" (DBSCAN, HDBSCAN).
NOISE_LABEL = -1
NOISE_CLUSTER_ID = "noise"

# Algorithms that don't produce iterative convergence info get this instead,
# matching the convention already used for this warehouse's historical runs.
NON_ITERATIVE_CONVERGENCE_ITERATIONS = 1


class ClusteringAlgorithm(str, Enum):
    """Supported clustering algorithms."""

    KMEANS = "kmeans"
    MINIBATCH_KMEANS = "minibatch_kmeans"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    GMM = "gmm"
    AGGLOMERATIVE = "agglomerative"


# Algorithms that require an explicit target cluster count up front.
_FIXED_K_ALGORITHMS = {
    ClusteringAlgorithm.KMEANS,
    ClusteringAlgorithm.MINIBATCH_KMEANS,
    ClusteringAlgorithm.GMM,
    ClusteringAlgorithm.AGGLOMERATIVE,
}

_DEFAULT_MODEL_BACKEND = {
    ClusteringAlgorithm.KMEANS: "scikit-learn",
    ClusteringAlgorithm.MINIBATCH_KMEANS: "scikit-learn",
    ClusteringAlgorithm.DBSCAN: "scikit-learn",
    ClusteringAlgorithm.HDBSCAN: "scikit-learn",
    ClusteringAlgorithm.GMM: "scikit-learn",
    ClusteringAlgorithm.AGGLOMERATIVE: "scikit-learn",
}

_SUPPORTED_DISTANCE_METRICS = {"euclidean", "cosine", "manhattan", "mahalanobis"}


class ClusteringConfigError(ValueError):
    """Raised when a ``ClusteringConfig`` is invalid or internally inconsistent."""


class ClusteringDataError(ValueError):
    """Raised when the input DataFrame can't be clustered as given."""


@dataclass
class ClusteringConfig:
    """Settings for a single clustering run.

    Parameters
    ----------
    algorithm
        Which clustering algorithm to run.
    model_name, model_id
        Identify the model in ``clustering_runs`` / ``cluster_metrics`` /
        ``cluster_assignments``. Caller-defined, not generated here.
    dataset_name, dataset_version
        Identify the input dataset for lineage purposes.
    id_column
        Column in the input DataFrame that uniquely identifies each record.
    feature_columns
        Columns to cluster on. Defaults to every column except ``id_column``.
    distance_metric
        One of "euclidean", "cosine", "manhattan", "mahalanobis". Used for
        every post-hoc distance and quality metric. Note that K-Means,
        Mini-Batch K-Means and the Gaussian Mixture Model always fit
        internally using their own fixed notion of distance regardless of
        this setting; it only changes how *reported* distances are computed.
    n_clusters
        Required for kmeans / minibatch_kmeans / gmm / agglomerative.
        Must be left as None for dbscan / hdbscan, which discover the
        cluster count themselves.
    random_seed
        Passed as ``random_state`` to algorithms that support it
        (kmeans, minibatch_kmeans, gmm). Recorded in the output either way.
    algorithm_params
        Extra keyword arguments forwarded to the underlying scikit-learn
        estimator, e.g. ``{"eps": 0.5, "min_samples": 10}`` for DBSCAN.
    outlier_zscore_threshold
        A non-noise point is flagged as a statistical outlier when its
        distance to its own cluster's centroid is more than this many
        standard deviations above that cluster's mean distance.
    representative_points_count
        How many record IDs closest to each centroid to store as that
        cluster's representative points.
    metric_sample_size
        Silhouette scoring is O(n^2); above this many records it is computed
        on a random subsample instead of the full dataset.
    cluster_labels
        Optional human-readable name per cluster label, e.g.
        ``{0: "High-Value Loyalists"}``. Clusters without an entry fall back
        to "Cluster <n>" (or "Noise" for the noise cluster).
    model_backend
        Overrides the auto-derived backend string (defaults to
        "scikit-learn" for every algorithm here).
    """

    algorithm: ClusteringAlgorithm
    model_name: str
    model_id: str
    dataset_name: str
    dataset_version: str
    id_column: str
    feature_columns: list[str] | None = None
    distance_metric: str = "euclidean"
    n_clusters: int | None = None
    random_seed: int | None = None
    algorithm_params: dict = field(default_factory=dict)
    outlier_zscore_threshold: float = 2.0
    representative_points_count: int = 3
    metric_sample_size: int = 5000
    cluster_labels: dict[int, str] | None = None
    model_backend: str | None = None


@dataclass
class ClusteringResult:
    """The three output tables produced by a single ``ClusteringPipeline.run`` call."""

    run: pd.DataFrame
    cluster_metrics: pd.DataFrame
    assignments: pd.DataFrame


@dataclass
class _FitOutcome:
    """Everything a per-algorithm fit method needs to report back."""

    labels: np.ndarray
    centroids: dict[int, np.ndarray]
    probabilities: np.ndarray
    convergence_iterations: int
    status: str


@dataclass
class _QualityMetrics:
    """Run-level cluster quality metrics, plus the per-cluster silhouette breakdown."""

    silhouette_score: float | None
    calinski_harabasz_score: float | None
    davies_bouldin_score: float | None
    dunn_index: float | None
    intra_cluster_distance: float | None
    inter_cluster_distance: float | None
    per_cluster_silhouette: dict[int, float]


_RUN_COLUMNS = [
    "uuid", "model_name", "model_id", "clustering_timestamp",
    "dataset_name", "dataset_version",
    "clustering_algorithm", "distance_metric", "model_backend",
    "embedding_dimension", "total_records",
    "requested_cluster_count", "actual_cluster_count",
    "random_seed",
    "clustering_time_ms", "total_execution_time_ms",
    "preprocessing_time_ms", "inference_time_ms",
    "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", "dunn_index",
    "intra_cluster_distance", "inter_cluster_distance",
    "min_cluster_size", "max_cluster_size", "mean_cluster_size", "std_cluster_size",
    "noise_points", "noise_ratio_pct",
    "outlier_count", "outlier_ratio_pct",
    "convergence_iterations",
    "status", "error_message", "recommended_action",
]

_CLUSTER_METRICS_COLUMNS = [
    "uuid", "run_uuid", "model_name", "model_id", "clustering_timestamp",
    "cluster_id", "cluster_label",
    "cluster_size", "cluster_percentage",
    "centroid", "cluster_radius",
    "intra_cluster_distance", "nearest_cluster_distance", "silhouette_score",
    "min_distance", "max_distance", "mean_distance", "std_distance",
    "median_distance", "p25_distance", "p50_distance", "p75_distance", "p95_distance", "p99_distance",
    "outlier_count", "outlier_percentage",
    "representative_points",
    "cluster_status", "cluster_metadata",
]

_ASSIGNMENTS_COLUMNS = [
    "uuid", "run_uuid", "model_name", "model_id", "assignment_timestamp",
    "record_id", "cluster_id",
    "cluster_distance", "cluster_probability",
    "is_noise", "is_outlier",
]


class ClusteringPipeline:
    """Fits one clustering algorithm and builds the three warehouse output tables.

    A pipeline instance is meant to be used for a single ``run()`` call; it
    caches a couple of values (like the inverse covariance matrix used for
    Mahalanobis distance) on ``self`` for the duration of that call.
    """

    def __init__(self, config: ClusteringConfig) -> None:
        self._validate_config(config)
        self.config = config
        self._inverse_covariance: np.ndarray | None = None

    def run(self, data: pd.DataFrame) -> ClusteringResult:
        """Cluster ``data`` and compute every metric the output tables need.

        Parameters
        ----------
        data
            One row per record, with an ID column and one or more numeric
            feature columns, as described by ``self.config``.

        Returns
        -------
        ClusteringResult
            The three output DataFrames. If the clustering algorithm itself
            fails at runtime, ``result.run`` still has exactly one row with
            ``status="FAILED"`` and ``error_message`` set, and the other two
            frames are returned empty (correct columns, zero rows) so a
            batch of many runs can tolerate one bad run without crashing.

        Raises
        ------
        ClusteringDataError
            If ``data`` doesn't match ``self.config`` (missing columns,
            empty frame, non-numeric features, NaN/inf values, duplicate
            IDs). These are caller mistakes, not runtime failures, so they
            raise immediately rather than producing a FAILED row.
        """
        run_uuid = str(uuid.uuid4())
        clustering_timestamp = datetime.now(timezone.utc)
        total_start = time.perf_counter()

        record_ids, feature_matrix, preprocessing_time_ms = self._prepare_features(data)

        try:
            scaled_matrix, inference_time_ms, outcome = self._fit(feature_matrix)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any estimator failure becomes a FAILED run row.
            logger.exception("Clustering run %s failed during model fit", run_uuid)
            return self._failed_result(run_uuid, clustering_timestamp, len(data), str(exc))

        clustering_time_ms = preprocessing_time_ms + inference_time_ms

        cluster_distances = self._compute_cluster_distances(scaled_matrix, outcome.labels, outcome.centroids)
        outlier_mask = self._flag_outliers(outcome.labels, cluster_distances)
        quality = self._compute_quality_metrics(scaled_matrix, outcome.labels, outcome.centroids, cluster_distances)

        run_row = self._build_run_row(
            run_uuid=run_uuid,
            clustering_timestamp=clustering_timestamp,
            total_records=len(data),
            embedding_dimension=feature_matrix.shape[1],
            preprocessing_time_ms=preprocessing_time_ms,
            inference_time_ms=inference_time_ms,
            clustering_time_ms=clustering_time_ms,
            total_execution_time_ms=(time.perf_counter() - total_start) * 1000,
            outcome=outcome,
            outlier_mask=outlier_mask,
            quality=quality,
        )

        cluster_metrics_rows = self._build_cluster_metrics_rows(
            run_uuid=run_uuid,
            clustering_timestamp=clustering_timestamp,
            record_ids=record_ids,
            labels=outcome.labels,
            centroids=outcome.centroids,
            cluster_distances=cluster_distances,
            outlier_mask=outlier_mask,
            per_cluster_silhouette=quality.per_cluster_silhouette,
        )

        assignment_rows = self._build_assignment_rows(
            run_uuid=run_uuid,
            record_ids=record_ids,
            labels=outcome.labels,
            cluster_distances=cluster_distances,
            probabilities=outcome.probabilities,
            outlier_mask=outlier_mask,
        )

        return ClusteringResult(
            run=pd.DataFrame([run_row], columns=_RUN_COLUMNS),
            cluster_metrics=pd.DataFrame(cluster_metrics_rows, columns=_CLUSTER_METRICS_COLUMNS),
            assignments=pd.DataFrame(assignment_rows, columns=_ASSIGNMENTS_COLUMNS),
        )

    # -- configuration and input validation -----------------------------------

    @staticmethod
    def _validate_config(config: ClusteringConfig) -> None:
        if config.distance_metric not in _SUPPORTED_DISTANCE_METRICS:
            raise ClusteringConfigError(
                f"Unsupported distance_metric {config.distance_metric!r}; "
                f"expected one of {sorted(_SUPPORTED_DISTANCE_METRICS)}"
            )

        needs_k = config.algorithm in _FIXED_K_ALGORITHMS
        if needs_k and (config.n_clusters is None or config.n_clusters < 2):
            raise ClusteringConfigError(
                f"{config.algorithm.value} requires n_clusters >= 2, got {config.n_clusters!r}"
            )
        if not needs_k and config.n_clusters is not None:
            raise ClusteringConfigError(
                f"{config.algorithm.value} discovers its own cluster count; "
                f"n_clusters must be left as None, got {config.n_clusters!r}"
            )

        # scikit-learn's AgglomerativeClustering only supports Mahalanobis distance
        # through a callable metric with precomputed parameters, which this pipeline
        # doesn't wire up, so reject it explicitly rather than let sklearn fail deep
        # inside the fit call with a confusing error.
        if config.algorithm == ClusteringAlgorithm.AGGLOMERATIVE and config.distance_metric == "mahalanobis":
            raise ClusteringConfigError("Agglomerative clustering does not support the mahalanobis distance metric")

    def _prepare_features(self, data: pd.DataFrame) -> tuple[list[str], np.ndarray, float]:
        start = time.perf_counter()
        config = self.config

        if data.empty:
            raise ClusteringDataError("Input DataFrame is empty; nothing to cluster")
        if config.id_column not in data.columns:
            raise ClusteringDataError(f"id_column {config.id_column!r} not found in input DataFrame")

        feature_columns = config.feature_columns or [c for c in data.columns if c != config.id_column]
        missing = [c for c in feature_columns if c not in data.columns]
        if missing:
            raise ClusteringDataError(f"feature_columns not found in input DataFrame: {missing}")
        if not feature_columns:
            raise ClusteringDataError("No feature columns to cluster on")

        if data[config.id_column].duplicated().any():
            raise ClusteringDataError(f"id_column {config.id_column!r} contains duplicate values")

        features = data[feature_columns]
        non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(features[c])]
        if non_numeric:
            raise ClusteringDataError(f"feature_columns must be numeric, got non-numeric: {non_numeric}")

        feature_matrix = features.to_numpy(dtype=float)
        if not np.isfinite(feature_matrix).all():
            raise ClusteringDataError("feature_columns contain NaN or infinite values")

        record_ids = data[config.id_column].astype(str).tolist()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return record_ids, feature_matrix, elapsed_ms

    # -- model fitting ----------------------------------------------------------

    def _fit(self, feature_matrix: np.ndarray) -> tuple[np.ndarray, float, _FitOutcome]:
        scaled_matrix = StandardScaler().fit_transform(feature_matrix)
        if self.config.distance_metric == "mahalanobis":
            # pinv rather than inv: guards against a singular covariance matrix
            # when features are collinear, which is common in embedding data.
            self._inverse_covariance = np.linalg.pinv(np.cov(scaled_matrix, rowvar=False))

        fit_methods = {
            ClusteringAlgorithm.KMEANS: self._fit_kmeans,
            ClusteringAlgorithm.MINIBATCH_KMEANS: self._fit_minibatch_kmeans,
            ClusteringAlgorithm.DBSCAN: self._fit_dbscan,
            ClusteringAlgorithm.HDBSCAN: self._fit_hdbscan,
            ClusteringAlgorithm.GMM: self._fit_gmm,
            ClusteringAlgorithm.AGGLOMERATIVE: self._fit_agglomerative,
        }

        start = time.perf_counter()
        outcome = fit_methods[self.config.algorithm](scaled_matrix)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return scaled_matrix, elapsed_ms, outcome

    def _fit_kmeans(self, X: np.ndarray) -> _FitOutcome:
        model = KMeans(n_clusters=self.config.n_clusters, random_state=self.config.random_seed, **self.config.algorithm_params)
        labels = model.fit_predict(X)
        centroids = {i: model.cluster_centers_[i] for i in range(model.cluster_centers_.shape[0])}
        probabilities = self._centroid_distance_probabilities(X, labels, centroids)
        return _FitOutcome(labels, centroids, probabilities, int(model.n_iter_), "SUCCESS")

    def _fit_minibatch_kmeans(self, X: np.ndarray) -> _FitOutcome:
        model = MiniBatchKMeans(n_clusters=self.config.n_clusters, random_state=self.config.random_seed, **self.config.algorithm_params)
        labels = model.fit_predict(X)
        centroids = {i: model.cluster_centers_[i] for i in range(model.cluster_centers_.shape[0])}
        probabilities = self._centroid_distance_probabilities(X, labels, centroids)
        return _FitOutcome(labels, centroids, probabilities, int(model.n_iter_), "SUCCESS")

    def _fit_dbscan(self, X: np.ndarray) -> _FitOutcome:
        params = dict(self.config.algorithm_params)
        params.setdefault("metric", self.config.distance_metric)
        if self.config.distance_metric == "mahalanobis":
            params.setdefault("metric_params", {"VI": self._inverse_covariance})

        model = DBSCAN(**params)
        labels = model.fit_predict(X)
        centroids = self._centroids_from_members(X, labels)
        probabilities = np.where(labels == NOISE_LABEL, 0.0, 1.0)
        return _FitOutcome(labels, centroids, probabilities, NON_ITERATIVE_CONVERGENCE_ITERATIONS, "SUCCESS")

    def _fit_hdbscan(self, X: np.ndarray) -> _FitOutcome:
        params = dict(self.config.algorithm_params)
        params.setdefault("metric", self.config.distance_metric)
        if self.config.distance_metric == "mahalanobis":
            params.setdefault("metric_params", {"VI": self._inverse_covariance})

        model = HDBSCAN(**params)
        labels = model.fit_predict(X)
        centroids = self._centroids_from_members(X, labels)
        return _FitOutcome(labels, centroids, model.probabilities_, NON_ITERATIVE_CONVERGENCE_ITERATIONS, "SUCCESS")

    def _fit_gmm(self, X: np.ndarray) -> _FitOutcome:
        model = GaussianMixture(n_components=self.config.n_clusters, random_state=self.config.random_seed, **self.config.algorithm_params)
        labels = model.fit_predict(X)
        probabilities = model.predict_proba(X).max(axis=1)
        centroids = {i: model.means_[i] for i in range(model.means_.shape[0])}
        status = "CONVERGED" if model.converged_ else "SUCCESS"
        return _FitOutcome(labels, centroids, probabilities, int(model.n_iter_), status)

    def _fit_agglomerative(self, X: np.ndarray) -> _FitOutcome:
        params = dict(self.config.algorithm_params)
        params.setdefault("metric", self.config.distance_metric)
        # ward linkage (sklearn's default) only supports euclidean distance;
        # fall back to average linkage for any other configured metric.
        if self.config.distance_metric != "euclidean":
            params.setdefault("linkage", "average")

        model = AgglomerativeClustering(n_clusters=self.config.n_clusters, **params)
        labels = model.fit_predict(X)
        centroids = self._centroids_from_members(X, labels)
        probabilities = self._centroid_distance_probabilities(X, labels, centroids)
        return _FitOutcome(labels, centroids, probabilities, NON_ITERATIVE_CONVERGENCE_ITERATIONS, "SUCCESS")

    @staticmethod
    def _centroids_from_members(X: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
        """Mean feature vector per cluster, for algorithms with no native centroid."""
        return {label: X[labels == label].mean(axis=0) for label in set(labels) if label != NOISE_LABEL}

    # -- distance helpers ---------------------------------------------------------

    def _metric_for(self, n_features: int) -> str:
        # Mahalanobis distance needs a full covariance matrix, which is
        # unreliable with very few features; fall back to euclidean rather
        # than risk a near-singular covariance matrix silently distorting
        # every reported distance.
        if self.config.distance_metric == "mahalanobis" and n_features < 2:
            return "euclidean"
        return self.config.distance_metric

    def _pairwise_distances(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """``pairwise_distances`` with the configured metric, wiring up Mahalanobis automatically."""
        metric = self._metric_for(X.shape[1])
        if metric == "mahalanobis":
            return pairwise_distances(X, Y, metric=metric, VI=self._inverse_covariance)
        return pairwise_distances(X, Y, metric=metric)

    def _centroid_distance_probabilities(self, X: np.ndarray, labels: np.ndarray, centroids: dict[int, np.ndarray]) -> np.ndarray:
        """Soft-assignment "probability" for algorithms with no native one.

        Centroid-based algorithms (K-Means family, Agglomerative) only ever
        produce a hard label, so this approximates a probability from how
        much closer a point is to its own centroid than to the others, via a
        softmax over negative distances. It is not a calibrated probability,
        just a normalized confidence score for the ``cluster_probability``
        column.
        """
        centroid_labels = sorted(centroids)
        centroid_matrix = np.stack([centroids[label] for label in centroid_labels])
        distances = self._pairwise_distances(X, centroid_matrix)

        negative_distances = -distances
        shifted = negative_distances - negative_distances.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        softmax = weights / weights.sum(axis=1, keepdims=True)

        label_to_column = {label: i for i, label in enumerate(centroid_labels)}
        own_columns = np.array([label_to_column[label] for label in labels])
        return softmax[np.arange(len(labels)), own_columns]

    def _compute_cluster_distances(self, X: np.ndarray, labels: np.ndarray, centroids: dict[int, np.ndarray]) -> np.ndarray:
        """Distance from each record to a centroid, one distance per record.

        For a clustered record, this is the distance to its own cluster's
        centroid. Noise records (DBSCAN/HDBSCAN, label -1) have no centroid
        of their own, so they get the distance to the *nearest* real
        cluster instead, which is the more useful signal for a noise point
        (how close did it come to joining a cluster). Stays NaN if there
        are no real clusters at all to measure against.
        """
        distances = np.full(len(labels), np.nan)
        if not centroids:
            return distances

        for label, centroid in centroids.items():
            member_mask = labels == label
            distances[member_mask] = self._pairwise_distances(X[member_mask], centroid.reshape(1, -1)).ravel()

        noise_mask = labels == NOISE_LABEL
        if noise_mask.any():
            centroid_matrix = np.stack(list(centroids.values()))
            distances[noise_mask] = self._pairwise_distances(X[noise_mask], centroid_matrix).min(axis=1)

        return distances

    def _flag_outliers(self, labels: np.ndarray, cluster_distances: np.ndarray) -> np.ndarray:
        """A record is an outlier if it's noise, or unusually far from its own centroid.

        The z-score check only applies within a record's own cluster, and
        noise points are always outliers regardless of distance. This
        mirrors the warehouse's existing convention of tracking
        ``noise_ratio_pct`` and ``outlier_ratio_pct`` as two separate
        signals: noise is a density concept specific to DBSCAN/HDBSCAN,
        while "outlier" is a statistical distance concept that applies to
        every algorithm.
        """
        is_outlier = labels == NOISE_LABEL
        for label in set(labels):
            if label == NOISE_LABEL:
                continue
            member_mask = labels == label
            member_distances = cluster_distances[member_mask]
            std = member_distances.std()
            if std == 0:
                continue
            z_scores = (member_distances - member_distances.mean()) / std
            is_outlier[member_mask] = z_scores > self.config.outlier_zscore_threshold
        return is_outlier

    # -- run-level quality metrics ------------------------------------------------

    def _compute_quality_metrics(
        self, X: np.ndarray, labels: np.ndarray, centroids: dict[int, np.ndarray], cluster_distances: np.ndarray
    ) -> _QualityMetrics:
        non_noise_mask = labels != NOISE_LABEL
        non_noise_labels = labels[non_noise_mask]
        cluster_count = len(set(non_noise_labels))

        if cluster_count < 2:
            logger.warning(
                "Only %d real cluster(s) found; silhouette/Calinski-Harabasz/"
                "Davies-Bouldin/Dunn index require at least 2 and will be left null",
                cluster_count,
            )
            return _QualityMetrics(None, None, None, None, None, None, {})

        # Calinski-Harabasz and Davies-Bouldin are inherently variance/centroid based
        # in scikit-learn's implementation; neither accepts a metric argument, so they
        # always reflect euclidean geometry regardless of the configured distance_metric.
        calinski = float(calinski_harabasz_score(X[non_noise_mask], non_noise_labels))
        davies_bouldin = float(davies_bouldin_score(X[non_noise_mask], non_noise_labels))

        silhouette_score, per_cluster_silhouette = self._silhouette_scores(X[non_noise_mask], non_noise_labels)

        centroid_labels = sorted(centroids)
        centroid_matrix = np.stack([centroids[label] for label in centroid_labels])
        centroid_distances = self._pairwise_distances(centroid_matrix)
        off_diagonal = centroid_distances[~np.eye(len(centroid_labels), dtype=bool)]
        inter_cluster_distance = float(off_diagonal.mean())

        per_cluster_mean_distance = {label: float(cluster_distances[labels == label].mean()) for label in centroid_labels}
        intra_cluster_distance = float(np.mean(list(per_cluster_mean_distance.values())))

        # Dunn index, computed from centroids rather than all pairwise points:
        # min distance between any two clusters, over the least tightly packed
        # cluster's average spread. The textbook definition uses full pairwise
        # point distances, which is O(n^2) and impractical at the record counts
        # this pipeline is meant for; this centroid-based variant is a standard,
        # much cheaper approximation that preserves the same "separation over
        # compactness" interpretation.
        min_inter_centroid_distance = float(off_diagonal.min())
        max_intra_cluster_spread = max(per_cluster_mean_distance.values())
        dunn_index = min_inter_centroid_distance / max_intra_cluster_spread if max_intra_cluster_spread > 0 else None

        return _QualityMetrics(
            silhouette_score=silhouette_score,
            calinski_harabasz_score=calinski,
            davies_bouldin_score=davies_bouldin,
            dunn_index=dunn_index,
            intra_cluster_distance=intra_cluster_distance,
            inter_cluster_distance=inter_cluster_distance,
            per_cluster_silhouette=per_cluster_silhouette,
        )

    def _silhouette_scores(self, X: np.ndarray, labels: np.ndarray) -> tuple[float, dict[int, float]]:
        """Overall and per-cluster silhouette, sharing one (possibly subsampled) pass.

        ``silhouette_samples`` is O(n^2); above ``metric_sample_size`` records
        we compute it on a random subsample instead so a 100k-row run doesn't
        require billions of distance calculations. The subsample is drawn
        without stratification by design: silhouette is a global geometric
        property, so a uniform random sample is representative, and forcing
        exact per-cluster quotas would bias it toward small clusters.
        """
        rng = np.random.default_rng(self.config.random_seed)
        if len(X) > self.config.metric_sample_size:
            sample_idx = rng.choice(len(X), size=self.config.metric_sample_size, replace=False)
        else:
            sample_idx = np.arange(len(X))

        metric = self._metric_for(X.shape[1])
        metric_kwargs = {"VI": self._inverse_covariance} if metric == "mahalanobis" else {}
        sample_values = silhouette_samples(X[sample_idx], labels[sample_idx], metric=metric, **metric_kwargs)
        sample_labels = labels[sample_idx]

        overall = float(sample_values.mean())
        per_cluster = {label: float(sample_values[sample_labels == label].mean()) for label in set(sample_labels)}
        return overall, per_cluster

    # -- building the output rows --------------------------------------------------

    def _build_run_row(
        self,
        run_uuid: str,
        clustering_timestamp: datetime,
        total_records: int,
        embedding_dimension: int,
        preprocessing_time_ms: float,
        inference_time_ms: float,
        clustering_time_ms: float,
        total_execution_time_ms: float,
        outcome: _FitOutcome,
        outlier_mask: np.ndarray,
        quality: _QualityMetrics,
    ) -> dict:
        config = self.config
        labels = outcome.labels
        non_noise_labels = labels[labels != NOISE_LABEL]

        cluster_sizes = pd.Series(non_noise_labels).value_counts()
        noise_points = int((labels == NOISE_LABEL).sum())
        # Outliers are counted only among non-noise points; see _flag_outliers.
        outlier_count = int(outlier_mask[labels != NOISE_LABEL].sum())

        return {
            "uuid": run_uuid,
            "model_name": config.model_name,
            "model_id": config.model_id,
            "clustering_timestamp": clustering_timestamp,
            "dataset_name": config.dataset_name,
            "dataset_version": config.dataset_version,
            "clustering_algorithm": config.algorithm.value,
            "distance_metric": config.distance_metric,
            "model_backend": config.model_backend or _DEFAULT_MODEL_BACKEND[config.algorithm],
            "embedding_dimension": embedding_dimension,
            "total_records": total_records,
            "requested_cluster_count": config.n_clusters or 0,
            "actual_cluster_count": int(cluster_sizes.size),
            "random_seed": config.random_seed,
            "clustering_time_ms": clustering_time_ms,
            "total_execution_time_ms": total_execution_time_ms,
            "preprocessing_time_ms": preprocessing_time_ms,
            "inference_time_ms": inference_time_ms,
            "silhouette_score": quality.silhouette_score,
            "calinski_harabasz_score": quality.calinski_harabasz_score,
            "davies_bouldin_score": quality.davies_bouldin_score,
            "dunn_index": quality.dunn_index,
            "intra_cluster_distance": quality.intra_cluster_distance,
            "inter_cluster_distance": quality.inter_cluster_distance,
            "min_cluster_size": int(cluster_sizes.min()) if not cluster_sizes.empty else 0,
            "max_cluster_size": int(cluster_sizes.max()) if not cluster_sizes.empty else 0,
            "mean_cluster_size": float(cluster_sizes.mean()) if not cluster_sizes.empty else 0.0,
            "std_cluster_size": float(cluster_sizes.std()) if cluster_sizes.size > 1 else 0.0,
            "noise_points": noise_points,
            "noise_ratio_pct": round(100 * noise_points / total_records, 2),
            "outlier_count": outlier_count,
            "outlier_ratio_pct": round(100 * outlier_count / total_records, 2),
            "convergence_iterations": outcome.convergence_iterations,
            "status": outcome.status,
            "error_message": None,
            "recommended_action": self._recommend_action(quality.silhouette_score, noise_points / total_records),
        }

    @staticmethod
    def _recommend_action(silhouette_score: float | None, noise_ratio: float) -> str:
        """A small rule of thumb, not a substitute for a human reviewing the run."""
        if silhouette_score is None:
            return "Investigate input features or algorithm parameters; fewer than 2 clusters were formed"
        if noise_ratio > 0.08:
            return "Tune density parameters (eps / min_samples / min_cluster_size); noise ratio is high"
        if silhouette_score >= 0.6:
            return "Deploy to production"
        if silhouette_score >= 0.5:
            return "Promote as challenger model"
        if silhouette_score >= 0.35:
            return "Re-evaluate cluster count and feature set"
        return "Do not promote; clustering quality is too low"

    def _build_cluster_metrics_rows(
        self,
        run_uuid: str,
        clustering_timestamp: datetime,
        record_ids: list[str],
        labels: np.ndarray,
        centroids: dict[int, np.ndarray],
        cluster_distances: np.ndarray,
        outlier_mask: np.ndarray,
        per_cluster_silhouette: dict[int, float],
    ) -> list[dict]:
        config = self.config
        total_records = len(labels)
        record_id_array = np.array(record_ids)
        centroid_labels = sorted(centroids)
        centroid_matrix = np.stack([centroids[label] for label in centroid_labels]) if centroid_labels else None

        rows = []
        for position, label in enumerate(centroid_labels):
            member_mask = labels == label
            member_distances = cluster_distances[member_mask]
            member_ids = record_id_array[member_mask]
            member_outlier_count = int(outlier_mask[member_mask].sum())

            other_centroids = np.delete(centroid_matrix, position, axis=0)
            nearest_cluster_distance = (
                float(self._pairwise_distances(centroids[label].reshape(1, -1), other_centroids).min())
                if len(other_centroids)
                else None
            )

            closest_order = np.argsort(member_distances)[: config.representative_points_count]

            rows.append({
                "uuid": str(uuid.uuid4()),
                "run_uuid": run_uuid,
                "model_name": config.model_name,
                "model_id": config.model_id,
                "clustering_timestamp": clustering_timestamp,
                "cluster_id": f"cluster_{label}",
                "cluster_label": self._cluster_label_for(label),
                "cluster_size": int(member_mask.sum()),
                "cluster_percentage": round(100 * member_mask.sum() / total_records, 2),
                "centroid": json.dumps(np.round(centroids[label], 3).tolist()),
                "cluster_radius": float(member_distances.max()),
                "intra_cluster_distance": self._intra_cluster_distance_proxy(member_distances),
                "nearest_cluster_distance": nearest_cluster_distance,
                "silhouette_score": per_cluster_silhouette.get(label),
                "min_distance": float(member_distances.min()),
                "max_distance": float(member_distances.max()),
                "mean_distance": float(member_distances.mean()),
                "std_distance": float(member_distances.std()),
                "median_distance": float(np.median(member_distances)),
                "p25_distance": float(np.percentile(member_distances, 25)),
                "p50_distance": float(np.percentile(member_distances, 50)),
                "p75_distance": float(np.percentile(member_distances, 75)),
                "p95_distance": float(np.percentile(member_distances, 95)),
                "p99_distance": float(np.percentile(member_distances, 99)),
                "outlier_count": member_outlier_count,
                "outlier_percentage": round(100 * member_outlier_count / member_mask.sum(), 2),
                "representative_points": member_ids[closest_order].tolist(),
                "cluster_status": self._cluster_status(per_cluster_silhouette.get(label)),
                "cluster_metadata": json.dumps({"member_count": int(member_mask.sum())}),
            })

        noise_mask = labels == NOISE_LABEL
        if noise_mask.any():
            noise_ids = record_id_array[noise_mask]
            rows.append({
                "uuid": str(uuid.uuid4()),
                "run_uuid": run_uuid,
                "model_name": config.model_name,
                "model_id": config.model_id,
                "clustering_timestamp": clustering_timestamp,
                "cluster_id": NOISE_CLUSTER_ID,
                "cluster_label": self._cluster_label_for(None),
                "cluster_size": int(noise_mask.sum()),
                "cluster_percentage": round(100 * noise_mask.sum() / total_records, 2),
                "centroid": None,
                "cluster_radius": None,
                "intra_cluster_distance": None,
                "nearest_cluster_distance": None,
                "silhouette_score": None,
                "min_distance": None,
                "max_distance": None,
                "mean_distance": None,
                "std_distance": None,
                "median_distance": None,
                "p25_distance": None,
                "p50_distance": None,
                "p75_distance": None,
                "p95_distance": None,
                "p99_distance": None,
                "outlier_count": int(noise_mask.sum()),
                "outlier_percentage": 100.0,
                "representative_points": noise_ids[: config.representative_points_count].tolist(),
                "cluster_status": "NOISE",
                "cluster_metadata": json.dumps({"member_count": int(noise_mask.sum())}),
            })

        return rows

    @staticmethod
    def _intra_cluster_distance_proxy(member_distances: np.ndarray) -> float:
        """Cheap proxy for the average pairwise distance between members of a cluster.

        The true average pairwise point-to-point distance within a cluster
        is O(m^2) per cluster, too expensive to justify at the record counts
        this pipeline targets. For points spread roughly evenly around a
        centroid, doubling the average distance-to-centroid approximates the
        expected distance between two random members reasonably well.
        """
        return float(member_distances.mean() * 2)

    def _cluster_label_for(self, label: int | None) -> str:
        if label is None:
            return "Noise"
        if self.config.cluster_labels and label in self.config.cluster_labels:
            return self.config.cluster_labels[label]
        return f"Cluster {label}"

    @staticmethod
    def _cluster_status(silhouette_score: float | None) -> str:
        if silhouette_score is None:
            return "UNKNOWN"
        if silhouette_score >= 0.6:
            return "OPTIMAL"
        if silhouette_score >= 0.4:
            return "STABLE"
        return "DIVERGENT"

    def _build_assignment_rows(
        self,
        run_uuid: str,
        record_ids: list[str],
        labels: np.ndarray,
        cluster_distances: np.ndarray,
        probabilities: np.ndarray,
        outlier_mask: np.ndarray,
    ) -> list[dict]:
        config = self.config
        assignment_timestamp = datetime.now(timezone.utc)

        rows = []
        for record_id, label, distance, probability, is_outlier in zip(
            record_ids, labels, cluster_distances, probabilities, outlier_mask
        ):
            is_noise = bool(label == NOISE_LABEL)
            rows.append({
                "uuid": str(uuid.uuid4()),
                "run_uuid": run_uuid,
                "model_name": config.model_name,
                "model_id": config.model_id,
                "assignment_timestamp": assignment_timestamp,
                "record_id": record_id,
                "cluster_id": NOISE_CLUSTER_ID if is_noise else f"cluster_{label}",
                "cluster_distance": None if np.isnan(distance) else float(distance),
                "cluster_probability": float(probability),
                "is_noise": is_noise,
                "is_outlier": bool(is_outlier),
            })
        return rows

    @staticmethod
    def _failed_result(run_uuid: str, clustering_timestamp: datetime, total_records: int, error_message: str) -> ClusteringResult:
        run_row = {column: None for column in _RUN_COLUMNS}
        run_row.update({
            "uuid": run_uuid,
            "clustering_timestamp": clustering_timestamp,
            "total_records": total_records,
            "status": "FAILED",
            "error_message": error_message,
            "recommended_action": "Investigate pipeline failure before retrying",
        })
        return ClusteringResult(
            run=pd.DataFrame([run_row], columns=_RUN_COLUMNS),
            cluster_metrics=pd.DataFrame(columns=_CLUSTER_METRICS_COLUMNS),
            assignments=pd.DataFrame(columns=_ASSIGNMENTS_COLUMNS),
        )
