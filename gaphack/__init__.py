"""
gapHACk: Gap-Optimized Hierarchical Agglomerative Clustering

A Python package for DNA barcode clustering that optimizes for the barcode gap
between intra-species and inter-species genetic distances.
"""

__version__ = "0.1.1"

from .core import GapOptimizedClustering
from .utils import (
    calculate_distance_matrix,
    load_sequences_from_fasta,
    save_clusters_to_file,
    format_cluster_output,
    validate_sequences
)

__all__ = [
    "GapOptimizedClustering",
    "calculate_distance_matrix",
    "load_sequences_from_fasta",
    "save_clusters_to_file",
    "format_cluster_output",
    "validate_sequences"
]