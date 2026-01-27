"""
BLAST result analysis for gapHACk.

This module analyzes BLAST results to identify sequences that cluster with
the query using gap-optimized clustering.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

from .target_clustering import TargetModeClustering
from .distance_providers import MSACachedDistanceProvider
from .utils import MSAAlignmentError

logger = logging.getLogger(__name__)


@dataclass
class SequenceResult:
    """Result for a single sequence in the BLAST analysis."""
    index: int  # 0-based position in input (index 0 is the query)
    id: str
    in_query_cluster: bool
    distance_to_query: Optional[float]


@dataclass
class BlastAnalysisResult:
    """Complete result from BLAST analysis."""
    # Query information
    query_id: str
    query_length: int

    # Summary statistics
    total_sequences: int
    query_cluster_size: int
    barcode_gap_found: bool
    gap_size: Optional[float]  # Inter min - intra max (in distance units)
    gap_size_percent: Optional[float]  # Gap size as percentage

    # Intra-cluster distance distribution (within query cluster)
    intra_cluster_distance: Dict[str, Optional[float]]  # min, p5, median, p95, max

    # Distance to nearest sequence outside query cluster
    nearest_other_distance: Optional[float]

    # Per-sequence results
    sequences: List[SequenceResult]

    # Diagnostic information
    method: str
    min_split: float
    max_lump: float
    normalization_length: int
    distance_metric: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def round_distance(val: Optional[float]) -> Optional[float]:
            """Round distance to 4 decimal places (sufficient for 1 nt resolution)."""
            return round(val, 4) if val is not None else None

        # Round intra_cluster_distance values
        intra_rounded = {
            k: round_distance(v) for k, v in self.intra_cluster_distance.items()
        }

        return {
            "query": {
                "id": self.query_id,
                "length": self.query_length
            },
            "summary": {
                "total_sequences": self.total_sequences,
                "query_cluster_size": self.query_cluster_size,
                "barcode_gap_found": self.barcode_gap_found,
                "gap_size": round_distance(self.gap_size),
                "gap_size_percent": round(self.gap_size_percent, 2) if self.gap_size_percent is not None else None,
                "intra_cluster_distance": intra_rounded,
                "nearest_other_distance": round_distance(self.nearest_other_distance)
            },
            "sequences": [
                {
                    "index": seq.index,
                    "id": seq.id,
                    "in_query_cluster": seq.in_query_cluster,
                    "distance_to_query": round_distance(seq.distance_to_query)
                }
                for seq in self.sequences
            ],
            "diagnostics": {
                "method": self.method,
                "min_split": self.min_split,
                "max_lump": self.max_lump,
                "normalization_length": self.normalization_length,
                "distance_metric": self.distance_metric,
                "warnings": self.warnings
            }
        }


class BlastAnalyzer:
    """
    Analyze BLAST results to identify sequences in the query's cluster.

    Uses gap-optimized target mode clustering to determine which sequences
    from a BLAST search cluster with the query sequence.
    """

    def __init__(self,
                 min_split: float = 0.005,
                 max_lump: float = 0.02,
                 target_percentile: int = 95,
                 show_progress: bool = True):
        """
        Initialize the BLAST analyzer.

        Args:
            min_split: Minimum distance to split clusters (sequences closer are lumped)
            max_lump: Maximum distance to lump clusters (sequences farther are split)
            target_percentile: Percentile for gap optimization (default: 95)
            show_progress: Whether to show progress bars
        """
        self.min_split = min_split
        self.max_lump = max_lump
        self.target_percentile = target_percentile
        self.show_progress = show_progress

    def analyze(self,
                sequences: List[str],
                headers: List[str]) -> BlastAnalysisResult:
        """
        Analyze sequences where the first sequence is the query.

        Args:
            sequences: List of DNA sequences (first is query)
            headers: List of sequence headers/IDs (first is query)

        Returns:
            BlastAnalysisResult with classification and metrics
        """
        if len(sequences) < 2:
            return self._create_insufficient_data_result(sequences, headers)

        query_seq = sequences[0]
        query_id = headers[0]
        query_length = len(query_seq)

        warnings = []

        # Create MSA-based distance provider
        try:
            logger.info(f"Creating MSA for {len(sequences)} sequences...")
            distance_provider = MSACachedDistanceProvider(sequences, headers)
        except MSAAlignmentError as e:
            logger.error(f"MSA alignment failed: {e}")
            return self._create_alignment_error_result(sequences, headers, str(e))

        # Run target mode clustering with query as seed
        logger.info("Running target mode clustering...")
        clustering = TargetModeClustering(
            min_split=self.min_split,
            max_lump=self.max_lump,
            target_percentile=self.target_percentile,
            show_progress=self.show_progress
        )

        target_cluster, remaining, metrics = clustering.cluster(
            distance_provider,
            target_indices=[0],  # Query is first sequence
            sequences=sequences
        )

        # Extract distances to query for all sequences
        distances_to_query = self._extract_distances_to_query(distance_provider, len(sequences))

        # Determine query cluster set (sequences in target cluster)
        query_cluster_set = set(target_cluster)

        # Compute gap metrics
        gap_metrics = self._compute_gap_metrics(
            query_cluster_set,
            set(remaining),
            distances_to_query,
            metrics
        )

        # Classify each sequence
        sequence_results = self._classify_sequences(
            headers,
            query_cluster_set,
            distances_to_query
        )

        # Check for warnings
        if len(target_cluster) == 1:
            warnings.append("Query has no close matches")

        # Check for NaN distances
        nan_count = sum(1 for d in distances_to_query.values() if d is None or (isinstance(d, float) and np.isnan(d)))
        if nan_count > 0:
            warnings.append(f"{nan_count} sequences had insufficient overlap for distance calculation")

        return BlastAnalysisResult(
            query_id=query_id,
            query_length=query_length,
            total_sequences=len(sequences),
            query_cluster_size=len(target_cluster),
            barcode_gap_found=gap_metrics['gap_found'],
            gap_size=gap_metrics['gap_size'],
            gap_size_percent=gap_metrics['gap_size_percent'],
            intra_cluster_distance=gap_metrics['intra_cluster_distance'],
            nearest_other_distance=gap_metrics['nearest_other_distance'],
            sequences=sequence_results,
            method="gap-optimized-target-clustering",
            min_split=self.min_split,
            max_lump=self.max_lump,
            normalization_length=distance_provider.normalization_length,
            distance_metric="MycoBLAST-adjusted (homopolymer-normalized, indel-normalized)",
            warnings=warnings
        )

    def _extract_distances_to_query(self,
                                     distance_provider: MSACachedDistanceProvider,
                                     n: int) -> Dict[int, Optional[float]]:
        """Extract distances from all sequences to the query (index 0)."""
        distances = {}
        for i in range(n):
            if i == 0:
                distances[i] = 0.0
            else:
                dist = distance_provider.get_distance(0, i)
                distances[i] = dist if not np.isnan(dist) else None
        return distances

    def _compute_gap_metrics(self,
                             query_cluster_set: set,
                             remaining_set: set,
                             distances_to_query: Dict[int, Optional[float]],
                             clustering_metrics: Dict) -> Dict[str, Any]:
        """Compute barcode gap metrics from clustering results."""

        # Get intra-cluster distances (query cluster members to query)
        intra_distances = []
        for idx in query_cluster_set:
            if idx == 0:
                continue  # Skip query itself
            dist = distances_to_query.get(idx)
            if dist is not None and not np.isnan(dist):
                intra_distances.append(dist)

        # Get inter-cluster distances (sequences outside query cluster to query)
        inter_distances = []
        for idx in remaining_set:
            dist = distances_to_query.get(idx)
            if dist is not None and not np.isnan(dist):
                inter_distances.append(dist)

        # Compute intra-cluster statistics
        if intra_distances:
            intra_stats = {
                "min": float(np.min(intra_distances)),
                "p5": float(np.percentile(intra_distances, 5)),
                "median": float(np.median(intra_distances)),
                "p95": float(np.percentile(intra_distances, 95)),
                "max": float(np.max(intra_distances))
            }
            intra_max = intra_stats["max"]
        else:
            intra_stats = {"min": None, "p5": None, "median": None, "p95": None, "max": None}
            intra_max = 0.0

        # Compute nearest distance outside query cluster
        if inter_distances:
            nearest_other = float(np.min(inter_distances))
        else:
            nearest_other = None

        # Calculate gap
        if nearest_other is not None and intra_max is not None:
            gap_size = nearest_other - intra_max
            gap_found = gap_size > 0
            gap_size_percent = gap_size * 100  # Convert to percentage
        else:
            gap_size = None
            gap_found = False
            gap_size_percent = None

        return {
            "gap_found": gap_found,
            "gap_size": gap_size,
            "gap_size_percent": gap_size_percent,
            "intra_cluster_distance": intra_stats,
            "nearest_other_distance": nearest_other
        }

    def _classify_sequences(self,
                            headers: List[str],
                            query_cluster_set: set,
                            distances_to_query: Dict[int, Optional[float]]) -> List[SequenceResult]:
        """Classify each sequence based on clustering results."""
        results = []

        for idx, header in enumerate(headers):
            in_cluster = idx in query_cluster_set
            distance = distances_to_query.get(idx)

            results.append(SequenceResult(
                index=idx,
                id=header,
                in_query_cluster=in_cluster,
                distance_to_query=distance
            ))

        return results

    def _create_insufficient_data_result(self,
                                          sequences: List[str],
                                          headers: List[str]) -> BlastAnalysisResult:
        """Create result when there's insufficient data for analysis."""
        query_id = headers[0] if headers else "unknown"
        query_length = len(sequences[0]) if sequences else 0

        sequence_results = []
        if headers:
            sequence_results.append(SequenceResult(
                index=0,
                id=headers[0],
                in_query_cluster=True,
                distance_to_query=0.0
            ))

        return BlastAnalysisResult(
            query_id=query_id,
            query_length=query_length,
            total_sequences=len(sequences),
            query_cluster_size=1 if sequences else 0,
            barcode_gap_found=False,
            gap_size=None,
            gap_size_percent=None,
            intra_cluster_distance={"min": None, "p5": None, "median": None, "p95": None, "max": None},
            nearest_other_distance=None,
            sequences=sequence_results,
            method="gap-optimized-target-clustering",
            min_split=self.min_split,
            max_lump=self.max_lump,
            normalization_length=query_length,
            distance_metric="MycoBLAST-adjusted (homopolymer-normalized, indel-normalized)",
            warnings=["Insufficient data - need at least 2 sequences for analysis"]
        )

    def _create_alignment_error_result(self,
                                        sequences: List[str],
                                        headers: List[str],
                                        error_msg: str) -> BlastAnalysisResult:
        """Create result when MSA alignment fails."""
        query_id = headers[0] if headers else "unknown"
        query_length = len(sequences[0]) if sequences else 0

        sequence_results = []
        for i, header in enumerate(headers):
            sequence_results.append(SequenceResult(
                index=i,
                id=header,
                in_query_cluster=(i == 0),  # Only query is in cluster
                distance_to_query=0.0 if i == 0 else None
            ))

        return BlastAnalysisResult(
            query_id=query_id,
            query_length=query_length,
            total_sequences=len(sequences),
            query_cluster_size=1,
            barcode_gap_found=False,
            gap_size=None,
            gap_size_percent=None,
            intra_cluster_distance={"min": None, "p5": None, "median": None, "p95": None, "max": None},
            nearest_other_distance=None,
            sequences=sequence_results,
            method="gap-optimized-target-clustering",
            min_split=self.min_split,
            max_lump=self.max_lump,
            normalization_length=query_length,
            distance_metric="MycoBLAST-adjusted (homopolymer-normalized, indel-normalized)",
            warnings=[f"MSA alignment failed: {error_msg}"]
        )


def format_text_output(result: BlastAnalysisResult) -> str:
    """Format analysis result as human-readable text."""
    lines = []

    lines.append("=" * 60)
    lines.append("BLAST Analysis Results")
    lines.append("=" * 60)
    lines.append("")

    # Query info
    lines.append(f"Query: {result.query_id}")
    lines.append(f"Query length: {result.query_length} bp")
    lines.append("")

    # Summary
    lines.append("-" * 40)
    lines.append("Summary")
    lines.append("-" * 40)
    lines.append(f"Total sequences: {result.total_sequences}")
    lines.append(f"Query cluster size: {result.query_cluster_size}")
    lines.append(f"Barcode gap found: {'Yes' if result.barcode_gap_found else 'No'}")

    if result.gap_size is not None:
        lines.append(f"Gap size: {result.gap_size:.4f} ({result.gap_size_percent:.2f}%)")

    if result.nearest_other_distance is not None:
        lines.append(f"Nearest outside cluster: {result.nearest_other_distance:.4f}")

    intra = result.intra_cluster_distance
    if intra.get('median') is not None:
        lines.append(f"Intra-cluster distances: min={intra['min']:.4f}, median={intra['median']:.4f}, max={intra['max']:.4f}")

    lines.append("")

    # Sequence classifications
    lines.append("-" * 40)
    lines.append("Sequence Classifications")
    lines.append("-" * 40)
    lines.append(f"{'Idx':<5} {'ID':<30} {'In Cluster':<12} {'Distance':<10}")
    lines.append("-" * 57)

    for seq in result.sequences:
        in_cluster = "Yes" if seq.in_query_cluster else "No"
        dist = f"{seq.distance_to_query:.4f}" if seq.distance_to_query is not None else "N/A"
        lines.append(f"{seq.index:<5} {seq.id:<30} {in_cluster:<12} {dist:<10}")

    lines.append("")

    # Diagnostics
    lines.append("-" * 40)
    lines.append("Diagnostics")
    lines.append("-" * 40)
    lines.append(f"Method: {result.method}")
    lines.append(f"min_split: {result.min_split}")
    lines.append(f"max_lump: {result.max_lump}")
    lines.append(f"Normalization length: {result.normalization_length} bp")
    lines.append(f"Distance metric: {result.distance_metric}")

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    lines.append("")

    return "\n".join(lines)


def format_tsv_output(result: BlastAnalysisResult) -> str:
    """Format analysis result as TSV (tab-separated values)."""
    lines = []

    # Header
    lines.append("index\tid\tin_query_cluster\tdistance_to_query")

    # Data rows
    for seq in result.sequences:
        in_cluster = "true" if seq.in_query_cluster else "false"
        dist = f"{seq.distance_to_query:.4f}" if seq.distance_to_query is not None else ""
        lines.append(f"{seq.index}\t{seq.id}\t{in_cluster}\t{dist}")

    return "\n".join(lines)
