# Clustering Pipeline

A Python library module that runs a clustering algorithm over a feature dataset
and computes every metric required by three data warehouse tables:
`clustering_runs`, `cluster_metrics`, and `cluster_assignments`. It takes a
pandas DataFrame of features as input and returns three pandas DataFrames as
output, one per table, with columns in the same name and order as the target
schema. The caller is responsible for loading the returned DataFrames into
Hive/Impala (or any other destination) using whatever mechanism is already in
use for that warehouse.

This document describes the module's architecture, its public API, every
supported algorithm and distance metric, and the exact formula or procedure
behind each computed metric.

## Table of Contents

1. [Overview](#overview)
2. [Repository Contents](#repository-contents)
3. [Requirements](#requirements)
4. [Quick Start](#quick-start)
5. [Architecture and Pipeline Stages](#architecture-and-pipeline-stages)
6. [Public API Reference](#public-api-reference)
   - [ClusteringAlgorithm](#clusteringalgorithm)
   - [ClusteringConfig](#clusteringconfig)
   - [ClusteringResult](#clusteringresult)
   - [ClusteringPipeline](#clusteringpipeline)
7. [Supported Algorithms](#supported-algorithms)
8. [Distance Metrics](#distance-metrics)
9. [Metrics and Formulas](#metrics-and-formulas)
   - [Silhouette Score](#silhouette-score)
   - [Calinski-Harabasz Score](#calinski-harabasz-score)
   - [Davies-Bouldin Score](#davies-bouldin-score)
   - [Dunn Index](#dunn-index)
   - [Intra-Cluster Distance](#intra-cluster-distance)
   - [Inter-Cluster Distance](#inter-cluster-distance)
   - [Cluster Probability](#cluster-probability)
   - [Outlier Detection (Z-Score)](#outlier-detection-z-score)
   - [Noise versus Outliers](#noise-versus-outliers)
   - [Distance Percentiles and Cluster Radius](#distance-percentiles-and-cluster-radius)
   - [Recommended Action](#recommended-action)
   - [Cluster Status](#cluster-status)
10. [Output Schema Reference](#output-schema-reference)
    - [clustering_runs](#clustering_runs)
    - [cluster_metrics](#cluster_metrics)
    - [cluster_assignments](#cluster_assignments)
11. [Error Handling](#error-handling)
12. [Design Notes and Limitations](#design-notes-and-limitations)
13. [Demo Notebook](#demo-notebook)

## Overview

The module does not connect to Hive, Impala, or any warehouse directly. Its
only job is to:

1. Validate a configuration and an input feature DataFrame.
2. Fit one of six supported clustering algorithms.
3. Compute run-level quality metrics, per-cluster statistics, and per-record
   assignment details.
4. Return three pandas DataFrames whose columns match the three target
   tables exactly, so a caller can hand them to any existing load mechanism
   (Spark, `impyla`, `INSERT` statements, an ORC writer, and so on).

A single `ClusteringPipeline` instance performs exactly one clustering run.
Configuration errors and malformed input raise exceptions immediately.
Failures that occur inside the clustering algorithm itself (for example, an
estimator rejecting a parameter) are caught and converted into a single
`clustering_runs` row with `status = "FAILED"`, so a batch of many runs can
tolerate one bad run without crashing.

## Repository Contents

| File | Purpose |
|---|---|
| `clustering_pipeline.py` | The library module: all public classes, the clustering logic, and every metric computation described in this document. |
| `__init__.py` | Re-exports the public API (`ClusteringAlgorithm`, `ClusteringConfig`, `ClusteringResult`, `ClusteringPipeline`, `ClusteringConfigError`, `ClusteringDataError`) so the directory can be imported as a package. |
| `clustering_pipeline_demo.ipynb` | A runnable Jupyter notebook that imports the module directly, builds a synthetic dataset, and demonstrates all three output tables for several algorithms. |
| `README.md` | This document. |

## Requirements

| Library | Purpose |
|---|---|
| `pandas` | Input and output DataFrames. |
| `numpy` | Numeric arrays, distance and statistics computation. |
| `scipy` | Available for numeric support used by `scikit-learn`. |
| `scikit-learn` (>= 1.3) | All clustering estimators and quality metrics. Version 1.3 or later is required because `sklearn.cluster.HDBSCAN` (the native scikit-learn implementation, not the separate `hdbscan` package) was introduced in that release. |

No other third-party dependency is required. There is no command-line
interface; the module is meant to be imported.

## Quick Start

```python
import pandas as pd
from clustering_pipeline import ClusteringAlgorithm, ClusteringConfig, ClusteringPipeline

features_df = pd.read_csv("customer_feature_embeddings.csv")

config = ClusteringConfig(
    algorithm=ClusteringAlgorithm.KMEANS,
    model_name="cust-segmentation-kmeans",
    model_id="mdl-km-8821",
    dataset_name="customer_feature_embeddings",
    dataset_version="v2.4",
    id_column="customer_id",
    n_clusters=5,
    random_seed=42,
)

result = ClusteringPipeline(config).run(features_df)

result.run              # one row -> clustering_runs
result.cluster_metrics  # one row per cluster (+ one "noise" row if applicable) -> cluster_metrics
result.assignments      # one row per input record -> cluster_assignments
```

## Architecture and Pipeline Stages

`ClusteringPipeline.run()` executes the following stages in order:

| Stage | Method | Description |
|---|---|---|
| 1 | `_validate_config` | Runs once, at construction time. Rejects an unsupported `distance_metric`, a missing or unexpected `n_clusters`, and the unsupported agglomerative + Mahalanobis combination. |
| 2 | `_prepare_features` | Validates the input DataFrame (non-empty, required columns present, no duplicate IDs, numeric features, no `NaN`/`inf`), then extracts the feature matrix and record IDs. |
| 3 | `_fit` | Standardizes features with `StandardScaler`, computes the inverse covariance matrix once if the Mahalanobis metric is configured, then dispatches to the selected algorithm's fit method. |
| 4 | `_compute_cluster_distances` | Computes, for every record, the distance to its own cluster's centroid (or the nearest real centroid, for noise records). |
| 5 | `_flag_outliers` | Flags statistical outliers using a per-cluster z-score test; all noise records are always flagged. |
| 6 | `_compute_quality_metrics` | Computes silhouette, Calinski-Harabasz, Davies-Bouldin, the Dunn index, and the intra-/inter-cluster distances. |
| 7 | `_build_run_row` | Assembles the single `clustering_runs` row: sizes, ratios, timings, and the recommended action. |
| 8 | `_build_cluster_metrics_rows` | Assembles one `cluster_metrics` row per real cluster, plus one `noise` row if any noise records exist. |
| 9 | `_build_assignment_rows` | Assembles one `cluster_assignments` row per input record. |

All feature standardization and every downstream distance and centroid
computation operate on the standardized feature matrix (mean 0, standard
deviation 1 per feature), not on the original feature scale. See
[Design Notes and Limitations](#design-notes-and-limitations).

## Public API Reference

### ClusteringAlgorithm

An enumeration of the six supported algorithms. Each member's value is the
string stored in the `clustering_algorithm` output column.

| Member | Value |
|---|---|
| `KMEANS` | `"kmeans"` |
| `MINIBATCH_KMEANS` | `"minibatch_kmeans"` |
| `DBSCAN` | `"dbscan"` |
| `HDBSCAN` | `"hdbscan"` |
| `GMM` | `"gmm"` |
| `AGGLOMERATIVE` | `"agglomerative"` |

### ClusteringConfig

A dataclass describing a single clustering run.

| Field | Type | Default | Description |
|---|---|---|---|
| `algorithm` | `ClusteringAlgorithm` | required | Which algorithm to run. |
| `model_name` | `str` | required | Caller-defined model name, recorded in every output row. |
| `model_id` | `str` | required | Caller-defined model identifier, recorded in every output row. |
| `dataset_name` | `str` | required | Name of the input dataset, for lineage. |
| `dataset_version` | `str` | required | Version of the input dataset, for lineage. |
| `id_column` | `str` | required | Column in the input DataFrame that uniquely identifies each record. |
| `feature_columns` | `list[str] \| None` | `None` | Columns to cluster on. When `None`, every column except `id_column` is used. |
| `distance_metric` | `str` | `"euclidean"` | One of `euclidean`, `cosine`, `manhattan`, `mahalanobis`. Used for every post-hoc distance and quality metric. See [Distance Metrics](#distance-metrics) for the important caveat that some algorithms ignore this during fitting. |
| `n_clusters` | `int \| None` | `None` | Required (and must be at least 2) for K-Means, Mini-Batch K-Means, GMM, and Agglomerative. Must be left as `None` for DBSCAN and HDBSCAN, which discover their own cluster count. |
| `random_seed` | `int \| None` | `None` | Passed as `random_state` to algorithms that support it; recorded in the output regardless. |
| `algorithm_params` | `dict` | `{}` | Extra keyword arguments forwarded directly to the underlying scikit-learn estimator, for example `{"eps": 0.5, "min_samples": 10}` for DBSCAN. |
| `outlier_zscore_threshold` | `float` | `2.0` | A non-noise record is flagged as a statistical outlier when its distance to its own cluster's centroid exceeds this many standard deviations above that cluster's mean distance. |
| `representative_points_count` | `int` | `3` | Number of record IDs closest to each centroid stored as that cluster's representative points. |
| `metric_sample_size` | `int` | `5000` | Above this many records, silhouette scoring is computed on a random subsample rather than the full dataset (silhouette is an O(n squared) computation). |
| `cluster_labels` | `dict[int, str] \| None` | `None` | Optional human-readable name per cluster label, for example `{0: "High-Value Loyalists"}`. Clusters without an entry fall back to `"Cluster <n>"`. |
| `model_backend` | `str \| None` | `None` | Overrides the auto-derived backend string. Defaults to `"scikit-learn"` for every algorithm. |

### ClusteringResult

A dataclass holding the three output DataFrames.

| Field | Type | Description |
|---|---|---|
| `run` | `pandas.DataFrame` | Exactly one row, matching the `clustering_runs` schema. |
| `cluster_metrics` | `pandas.DataFrame` | One row per real cluster, plus one `noise` row if any noise records exist, matching the `cluster_metrics` schema. |
| `assignments` | `pandas.DataFrame` | One row per input record, matching the `cluster_assignments` schema. |

### ClusteringPipeline

| Member | Description |
|---|---|
| `__init__(config: ClusteringConfig)` | Validates `config` immediately and raises `ClusteringConfigError` on failure. |
| `run(data: pandas.DataFrame) -> ClusteringResult` | Runs the full pipeline described in [Architecture and Pipeline Stages](#architecture-and-pipeline-stages) and returns the three output DataFrames. |

## Supported Algorithms

| Algorithm | scikit-learn estimator | Requires `n_clusters` | Native centroid | Native probability | Convergence info |
|---|---|---|---|---|---|
| K-Means | `KMeans` | Yes | Yes (`cluster_centers_`) | No (approximated, see [Cluster Probability](#cluster-probability)) | `n_iter_` |
| Mini-Batch K-Means | `MiniBatchKMeans` | Yes | Yes (`cluster_centers_`) | No (approximated) | `n_iter_` |
| DBSCAN | `DBSCAN` | No (discovered) | No (approximated as the mean feature vector per cluster) | No (1.0 for a clustered point, 0.0 for noise) | Fixed at 1 (non-iterative) |
| HDBSCAN | `HDBSCAN` | No (discovered) | No (approximated as the mean feature vector per cluster) | Yes (`probabilities_`) | Fixed at 1 (non-iterative) |
| Gaussian Mixture Model | `GaussianMixture` | Yes | Yes (`means_`) | Yes (`predict_proba`) | `n_iter_` |
| Agglomerative / Hierarchical | `AgglomerativeClustering` | Yes | No (approximated as the mean feature vector per cluster) | No (approximated) | Fixed at 1 (non-iterative) |

Algorithms without a native centroid (DBSCAN, HDBSCAN, Agglomerative) have
one computed as the arithmetic mean of the standardized feature vectors of
every record assigned to that cluster:

```
centroid_k = mean(x_i for every record i where label(i) = k)
```

## Distance Metrics

`distance_metric` controls every post-hoc distance and quality computation
performed after the algorithm produces its labels. For two feature vectors
`x` and `y` with components `x_1 .. x_n`:

| Metric | Formula |
|---|---|
| Euclidean | `d(x, y) = sqrt( sum_i (x_i - y_i)^2 )` |
| Manhattan | `d(x, y) = sum_i abs(x_i - y_i)` |
| Cosine | `d(x, y) = 1 - ( x . y ) / ( norm(x) * norm(y) )` |
| Mahalanobis | `d(x, y) = sqrt( (x - y)^T * S_inv * (x - y) )`, where `S_inv` is the pseudo-inverse of the covariance matrix of the standardized feature matrix |

The Mahalanobis inverse covariance matrix is computed once per run (via
`numpy.linalg.pinv`, a pseudo-inverse rather than a plain inverse, so a
singular or near-singular covariance matrix does not raise an error) and
reused everywhere a Mahalanobis distance is needed.

Important caveats:

- K-Means, Mini-Batch K-Means, and the Gaussian Mixture Model always fit
  internally using their own fixed notion of distance, regardless of
  `distance_metric`. The configured metric only changes how distances are
  *reported* afterward, not how the algorithm itself partitions the data.
- DBSCAN and HDBSCAN pass the configured metric straight through to
  scikit-learn, so it does affect how those two actually cluster.
- Agglomerative clustering also passes the metric through, but scikit-learn's
  default `ward` linkage only supports Euclidean distance; the pipeline
  automatically falls back to `average` linkage for any other configured
  metric.
- Agglomerative clustering does not support the Mahalanobis metric in this
  pipeline at all; `ClusteringConfigError` is raised at configuration time
  rather than letting the estimator fail deep inside `fit()`.
- If Mahalanobis distance is requested with fewer than 2 feature columns,
  the pipeline silently falls back to Euclidean distance, since a covariance
  matrix over a single feature is not a meaningful basis for this distance.
- Calinski-Harabasz and Davies-Bouldin are inherently variance/centroid based
  in scikit-learn's implementation and do not accept a metric argument at
  all; they always reflect Euclidean geometry regardless of the configured
  `distance_metric`.

## Metrics and Formulas

Unless stated otherwise, every metric below is computed on the standardized
feature matrix, after noise records (label `-1`) have been excluded.

### Silhouette Score

For a record `i` in cluster `A`, let `a(i)` be its mean distance to every
other record in `A`, and `b(i)` be the lowest mean distance from `i` to any
other cluster's records:

```
s(i) = ( b(i) - a(i) ) / max( a(i), b(i) )
```

`s(i)` ranges from -1 (poorly matched) to 1 (well matched). The run-level
`silhouette_score` is the mean of `s(i)` over all non-noise records; the
per-cluster silhouette stored in `cluster_metrics` is the mean of `s(i)`
restricted to that cluster's members.

Because this computation is O(n squared), any run with more than
`metric_sample_size` records (5,000 by default) computes it on a uniform
random subsample of that size instead of the full dataset. The subsample is
drawn without per-cluster stratification: silhouette is treated as a global
geometric property, so a plain random sample is representative, and forcing
exact per-cluster quotas would bias the result toward small clusters.

Left `null` (with a logged warning) when fewer than 2 real clusters were
formed, since silhouette is undefined for a single cluster.

### Calinski-Harabasz Score

Also known as the Variance Ratio Criterion. For `k` clusters over `n`
records, let `B_k` be the between-cluster dispersion matrix and `W_k` the
within-cluster dispersion matrix:

```
CH = ( trace(B_k) / (k - 1) ) / ( trace(W_k) / (n - k) )
```

Higher values indicate denser, better-separated clusters. There is no
theoretical upper bound. Computed with `sklearn.metrics.calinski_harabasz_score`.

### Davies-Bouldin Score

For each cluster `i`, let `s_i` be the average distance of its members to
its centroid (cluster scatter), and `d(c_i, c_j)` the distance between
centroids `i` and `j`:

```
DB = (1 / k) * sum_i  max_(j != i) [ ( s_i + s_j ) / d(c_i, c_j) ]
```

Lower values indicate better separation; 0 is the best possible score.
Computed with `sklearn.metrics.davies_bouldin_score`.

### Dunn Index

The textbook Dunn index is:

```
D = min_(i != j) d(c_i, c_j)  /  max_k diam(cluster_k)
```

where `diam(cluster_k)` is the largest pairwise distance between any two
members of cluster `k`. That definition requires all pairwise point
distances within every cluster, an O(n squared) computation that is
impractical at the record counts this pipeline targets. This module instead
uses a centroid-based approximation that preserves the same "separation over
compactness" interpretation at O(n) cost:

```
dunn_index = min_inter_centroid_distance / max_intra_cluster_spread
```

where:

```
min_inter_centroid_distance = min_(i != j) d(centroid_i, centroid_j)
max_intra_cluster_spread     = max_k ( mean distance of cluster k's members to centroid_k )
```

Higher values indicate better-separated, more compact clusters. `null` when
fewer than 2 real clusters exist, or when every cluster's spread is exactly
zero.

### Intra-Cluster Distance

This name is used for two related but distinct quantities in the two output
tables; they are not interchangeable.

Run-level (`clustering_runs.intra_cluster_distance`): the mean, across all
real clusters, of each cluster's mean distance from its members to its own
centroid:

```
intra_cluster_distance = mean_k ( mean distance of cluster k's members to centroid_k )
```

Per-cluster (`cluster_metrics.intra_cluster_distance`): an approximation of
the average pairwise distance *between members* of a single cluster (as
opposed to the distance to the centroid). Computing the true value is
O(m squared) per cluster of size `m`. For points spread roughly evenly around
a centroid, doubling the mean distance-to-centroid approximates the expected
distance between two random members of the same cluster reasonably well:

```
intra_cluster_distance_k = 2 * mean(distance of cluster k's members to centroid_k)
```

### Inter-Cluster Distance

The mean pairwise distance between every pair of distinct cluster centroids:

```
inter_cluster_distance = mean_(i != j) d(centroid_i, centroid_j)
```

Reported once per run, at the run level only (not per cluster; the
per-cluster equivalent is `nearest_cluster_distance`, described in
[Output Schema Reference](#output-schema-reference)).

### Cluster Probability

Reported per record in `cluster_assignments.cluster_probability`. Its source
depends on the algorithm:

| Algorithm | Source |
|---|---|
| Gaussian Mixture Model | `predict_proba(x).max()`, the model's own posterior responsibility for the most likely component. |
| HDBSCAN | `probabilities_`, the model's own cluster membership strength. |
| DBSCAN | `1.0` for a clustered record, `0.0` for a noise record. |
| K-Means, Mini-Batch K-Means, Agglomerative | Approximated (see below); these algorithms produce only a hard label. |

For the algorithms without a native probability, a softmax over the negative
distances from the record to every cluster's centroid is used:

```
p(x, k) = exp( -d(x, centroid_k) ) / sum_j exp( -d(x, centroid_j) )
```

`cluster_probability` is then `p(x, k)` for the record's own assigned cluster
`k`. This is not a calibrated probability in any statistical sense; it is a
normalized confidence score that is 1 when a point sits exactly on its own
centroid and far from every other, and approaches `1 / (number of clusters)`
when a point is equidistant from all of them.

### Outlier Detection (Z-Score)

Applied independently within each real cluster (never across clusters, and
never to noise records, which are always outliers regardless of distance).
For a cluster with member distances-to-centroid `d_1 .. d_m`, mean `mu`, and
standard deviation `sigma`:

```
z_i = ( d_i - mu ) / sigma
outlier_i = z_i > outlier_zscore_threshold
```

The default `outlier_zscore_threshold` is `2.0`. If a cluster's standard
deviation is exactly 0 (every member is equidistant from the centroid), no
member of that cluster is flagged, since the z-score is undefined.

### Noise versus Outliers

These are two independent concepts, tracked separately, and the schema's
`noise_points`/`noise_ratio_pct` and `outlier_count`/`outlier_ratio_pct`
columns are never derived from one another:

| Concept | Applies to | Meaning |
|---|---|---|
| Noise | DBSCAN, HDBSCAN only | A density-based concept native to those algorithms: a record too sparse to belong to any cluster (`label = -1`). |
| Outlier | Every algorithm | A statistical concept: a record whose distance from its own cluster's centroid is unusually large (z-score based), or any noise record. |

Because of this, `outlier_count` is computed only among non-noise records; a
run's outlier ratio can be smaller than, larger than, or unrelated to its
noise ratio, and for K-Means, Mini-Batch K-Means, GMM, and Agglomerative,
`noise_points` is always 0 while `outlier_count` can still be nonzero.

### Distance Percentiles and Cluster Radius

Computed per cluster, over that cluster's member distances-to-centroid
(`d_1 .. d_m`):

| Field | Definition |
|---|---|
| `min_distance` | `min(d_1 .. d_m)` |
| `max_distance` / `cluster_radius` | `max(d_1 .. d_m)` (both columns store the same value) |
| `mean_distance` | `mean(d_1 .. d_m)` |
| `std_distance` | Sample standard deviation of `d_1 .. d_m` |
| `median_distance` / `p50_distance` | 50th percentile |
| `p25_distance`, `p75_distance`, `p95_distance`, `p99_distance` | Percentiles via `numpy.percentile` |

### Recommended Action

A rule-of-thumb string, not a substitute for a human reviewing the run.
Rules are evaluated in order; the first match determines the result:

| Order | Condition | Recommended Action |
|---|---|---|
| 1 | Fewer than 2 real clusters were formed (`silhouette_score` is `null`) | "Investigate input features or algorithm parameters; fewer than 2 clusters were formed" |
| 2 | `noise_points / total_records > 0.08` | "Tune density parameters (eps / min_samples / min_cluster_size); noise ratio is high" |
| 3 | `silhouette_score >= 0.6` | "Deploy to production" |
| 4 | `silhouette_score >= 0.5` | "Promote as challenger model" |
| 5 | `silhouette_score >= 0.35` | "Re-evaluate cluster count and feature set" |
| 6 | Otherwise | "Do not promote; clustering quality is too low" |

Note that rule 2 is checked before the silhouette thresholds, so a run with
an excellent silhouette score but a high noise ratio is still flagged for
density-parameter tuning rather than for deployment.

### Cluster Status

Assigned per cluster, from that cluster's own silhouette score:

| Condition | Status |
|---|---|
| Silhouette score unavailable | `UNKNOWN` |
| Silhouette score >= 0.6 | `OPTIMAL` |
| Silhouette score >= 0.4 | `STABLE` |
| Otherwise | `DIVERGENT` |
| The dedicated noise row | `NOISE` (fixed, not derived from silhouette) |

## Output Schema Reference

### clustering_runs

One row per call to `run()`.

| Column | Description |
|---|---|
| `uuid` | Unique identifier for this run. |
| `model_name`, `model_id` | Caller-supplied identifiers from `ClusteringConfig`. |
| `clustering_timestamp` | UTC timestamp captured at the start of `run()`. |
| `dataset_name`, `dataset_version` | Caller-supplied dataset identifiers. |
| `clustering_algorithm` | The `ClusteringAlgorithm` value used. |
| `distance_metric` | The configured distance metric. |
| `model_backend` | `"scikit-learn"` unless overridden. |
| `embedding_dimension` | Number of feature columns clustered on. |
| `total_records` | Number of input rows. |
| `requested_cluster_count` | The configured `n_clusters`, or 0 for algorithms that discover their own count. |
| `actual_cluster_count` | Number of distinct real (non-noise) clusters found. |
| `random_seed` | The configured random seed. |
| `clustering_time_ms` | `preprocessing_time_ms + inference_time_ms`. |
| `total_execution_time_ms` | Wall-clock time for the entire `run()` call. |
| `preprocessing_time_ms` | Time spent validating input and building the feature matrix. |
| `inference_time_ms` | Time spent inside the estimator's fit call. |
| `silhouette_score` | See [Silhouette Score](#silhouette-score). |
| `calinski_harabasz_score` | See [Calinski-Harabasz Score](#calinski-harabasz-score). |
| `davies_bouldin_score` | See [Davies-Bouldin Score](#davies-bouldin-score). |
| `dunn_index` | See [Dunn Index](#dunn-index). |
| `intra_cluster_distance` | Run-level definition; see [Intra-Cluster Distance](#intra-cluster-distance). |
| `inter_cluster_distance` | See [Inter-Cluster Distance](#inter-cluster-distance). |
| `min_cluster_size`, `max_cluster_size`, `mean_cluster_size`, `std_cluster_size` | Summary statistics over real cluster sizes. |
| `noise_points`, `noise_ratio_pct` | Count and percentage of records with `label = -1`. |
| `outlier_count`, `outlier_ratio_pct` | Count and percentage of non-noise records flagged by the z-score test. |
| `convergence_iterations` | Iterations reported by the estimator, or 1 for non-iterative algorithms. |
| `status` | `"SUCCESS"`, `"CONVERGED"` (GMM only, when `converged_` is true), or `"FAILED"`. |
| `error_message` | `null` on success; the caught exception's message on failure. |
| `recommended_action` | See [Recommended Action](#recommended-action). |

### cluster_metrics

One row per real cluster, plus one additional row with `cluster_id = "noise"`
if the run produced any noise records.

| Column | Description |
|---|---|
| `uuid` | Unique identifier for this cluster row. |
| `run_uuid` | Foreign key to the parent `clustering_runs` row. |
| `model_name`, `model_id`, `clustering_timestamp` | Copied from the run. |
| `cluster_id` | `"cluster_<label>"` for a real cluster, `"noise"` for the noise row. |
| `cluster_label` | Human-readable name: the caller's `cluster_labels` override, `"Cluster <n>"` by default, or `"Noise"`. |
| `cluster_size` | Number of records in this cluster. |
| `cluster_percentage` | `100 * cluster_size / total_records`. |
| `centroid` | JSON array of the centroid's coordinates (standardized feature space, rounded to 3 decimals); `null` for the noise row. |
| `cluster_radius` | See [Distance Percentiles and Cluster Radius](#distance-percentiles-and-cluster-radius). |
| `intra_cluster_distance` | Per-cluster definition; see [Intra-Cluster Distance](#intra-cluster-distance). |
| `nearest_cluster_distance` | Distance from this cluster's centroid to the nearest other cluster's centroid. |
| `silhouette_score` | This cluster's mean per-record silhouette value. |
| `min_distance`, `max_distance`, `mean_distance`, `std_distance`, `median_distance`, `p25_distance`, `p50_distance`, `p75_distance`, `p95_distance`, `p99_distance` | See [Distance Percentiles and Cluster Radius](#distance-percentiles-and-cluster-radius). |
| `outlier_count`, `outlier_percentage` | Outliers within this cluster; always `100.0` percent for the noise row. |
| `representative_points` | The `representative_points_count` record IDs closest to the centroid. |
| `cluster_status` | See [Cluster Status](#cluster-status). |
| `cluster_metadata` | JSON object, currently `{"member_count": <cluster_size>}`. |

### cluster_assignments

One row per input record.

| Column | Description |
|---|---|
| `uuid` | Unique identifier for this assignment row. |
| `run_uuid` | Foreign key to the parent `clustering_runs` row. |
| `model_name`, `model_id` | Copied from the run. |
| `assignment_timestamp` | UTC timestamp captured when the assignment rows were built. |
| `record_id` | The value from the input DataFrame's `id_column`. |
| `cluster_id` | `"cluster_<label>"`, or `"noise"` for a noise record. |
| `cluster_distance` | Distance to the record's own centroid (or nearest centroid, if noise); `null` only when no real clusters exist at all. |
| `cluster_probability` | See [Cluster Probability](#cluster-probability). |
| `is_noise` | `true` if `label = -1`. |
| `is_outlier` | See [Outlier Detection (Z-Score)](#outlier-detection-z-score). |

## Error Handling

| Exception | Base class | Raised when |
|---|---|---|
| `ClusteringConfigError` | `ValueError` | An unsupported `distance_metric`; a fixed-k algorithm missing (or given an invalid) `n_clusters`; a discovery algorithm (DBSCAN/HDBSCAN) given a non-`None` `n_clusters`; Agglomerative clustering configured with the Mahalanobis metric. Raised at `ClusteringPipeline.__init__`, before any data is touched. |
| `ClusteringDataError` | `ValueError` | An empty input DataFrame; a missing `id_column` or feature column; duplicate values in `id_column`; non-numeric feature columns; `NaN` or infinite values in the feature columns. Raised at the start of `run()`, before fitting begins. |

Both are caller/input mistakes and are always raised immediately, never
swallowed.

Runtime failures that occur *while fitting the model* (for example, an
invalid `algorithm_params` value that the estimator itself rejects) are
caught inside `run()` and logged, and `run()` returns a `ClusteringResult`
whose `run` DataFrame has exactly one row with `status = "FAILED"` and
`error_message` set to the exception's message. `cluster_metrics` and
`assignments` are returned as empty DataFrames with the correct columns.
This deliberately does not raise, so that a batch of many runs can continue
past one failed run.

## Design Notes and Limitations

- All standardization and every downstream centroid, distance, and quality
  computation operate on the standardized feature matrix produced by
  `StandardScaler` (mean 0, standard deviation 1 per feature), not on the
  original feature units. This includes the `centroid` field stored in
  `cluster_metrics`. A caller that needs centroids in the original feature
  scale must inverse-transform them separately.
- Calinski-Harabasz and Davies-Bouldin always reflect Euclidean geometry,
  regardless of the configured `distance_metric`, because scikit-learn's
  implementations do not accept a metric argument.
- K-Means, Mini-Batch K-Means, and the Gaussian Mixture Model fit using their
  own fixed internal notion of distance; `distance_metric` only affects
  metrics computed after fitting, not the clustering result itself.
- The Dunn index and the per-cluster `intra_cluster_distance` are both
  cheaper approximations of their textbook, fully pairwise definitions,
  chosen to keep the pipeline at O(n) or O(n * k) cost rather than O(n
  squared) at the record counts it targets. See
  [Dunn Index](#dunn-index) and [Intra-Cluster Distance](#intra-cluster-distance).
- Silhouette scoring is subsampled above `metric_sample_size` records for the
  same reason.
- A single-cluster result (fewer than 2 real clusters found) is treated as a
  valid business outcome, not an error: silhouette, Calinski-Harabasz,
  Davies-Bouldin, and the Dunn index are left `null` with a logged warning,
  rather than raising.
- `recommended_action` and `cluster_status` are simple rule-based heuristics
  meant as a starting point for a human reviewer, not a substitute for one.

## Demo Notebook

`clustering_pipeline_demo.ipynb` imports `clustering_pipeline` directly (no
package installation required, provided the notebook is run from this
directory) and walks through:

1. Generating a synthetic dataset with `sklearn.datasets.make_blobs`.
2. Running K-Means and inspecting all three output DataFrames.
3. Visualizing the resulting clusters on a 2-D PCA projection.
4. Running DBSCAN to show the noise row and noise/outlier distinction in
   practice.
5. Running all six algorithms against the same dataset and comparing their
   run-level quality metrics side by side.

Open it with Jupyter and run all cells, or view the already-computed output
saved in the file.
