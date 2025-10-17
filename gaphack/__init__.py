"""
gapHACk: Gap-Optimized Hierarchical Agglomerative Clustering

A Python package for DNA barcode clustering that optimizes for the barcode gap
between intra-species and inter-species genetic distances.
"""

__version__ = "0.6.0"

from .core import GapOptimizedClustering
from .target_clustering import TargetModeClustering
from .distance_providers import (
    DistanceProvider,
    MSACachedDistanceProvider,
    PrecomputedDistanceProvider,
    MSAAlignmentError
)
from .utils import (
    calculate_distance_matrix,
    load_sequences_from_fasta,
    save_clusters_to_file,
    format_cluster_output,
    validate_sequences
)
from .analyze import (
    calculate_intra_cluster_distances,
    calculate_inter_cluster_distances,
    calculate_percentiles,
    calculate_barcode_gap_metrics,
    create_histogram,
    create_combined_histogram,
    format_analysis_report
)

__all__ = [
    "GapOptimizedClustering",
    "TargetModeClustering",
    "DistanceProvider",
    "MSACachedDistanceProvider",
    "PrecomputedDistanceProvider",
    "MSAAlignmentError",
    "calculate_distance_matrix",
    "load_sequences_from_fasta",
    "save_clusters_to_file",
    "format_cluster_output",
    "validate_sequences",
    "calculate_intra_cluster_distances",
    "calculate_inter_cluster_distances",
    "calculate_percentiles",
    "calculate_barcode_gap_metrics",
    "create_histogram",
    "create_combined_histogram",
    "format_analysis_report"
]