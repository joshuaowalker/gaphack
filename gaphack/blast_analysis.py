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
    identity_to_query: Optional[float]  # Pairwise identity % (comparable to BLAST)
    identity_to_query_normalized: Optional[float]  # Normalized identity % (used by clustering)
    identity_to_medoid_normalized: Optional[float]  # Identity % to query cluster medoid
    identity_to_nearest_non_member_normalized: Optional[float]  # For members: identity % to nearest non-member


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
    gap_size_percent: Optional[float]  # Gap size as percentage

    # Query cluster medoid (most representative sequence)
    medoid_id: Optional[str]  # Header of the medoid sequence
    medoid_index: Optional[int]  # 0-based index of the medoid in input

    # Intra-cluster identity distribution (within query cluster)
    intra_cluster_identity: Dict[str, Optional[float]]  # min, p5, median, p95, max (in %)

    # Identity to nearest sequence outside query cluster
    nearest_non_member_identity: Optional[float]

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
        def round_identity(val: Optional[float]) -> Optional[float]:
            """Round identity to 2 decimal places (e.g., 99.63%)."""
            return round(val, 2) if val is not None else None

        # Round intra_cluster_identity values
        intra_rounded = {
            k: round_identity(v) for k, v in self.intra_cluster_identity.items()
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
                "gap_size_percent": round(self.gap_size_percent, 2) if self.gap_size_percent is not None else None,
                "medoid_id": self.medoid_id,
                "medoid_index": self.medoid_index,
                "intra_cluster_identity": intra_rounded,
                "nearest_non_member_identity": round_identity(self.nearest_non_member_identity)
            },
            "sequences": [
                {
                    "index": seq.index,
                    "id": seq.id,
                    "in_query_cluster": seq.in_query_cluster,
                    "identity_to_query": round_identity(seq.identity_to_query),
                    "identity_to_query_normalized": round_identity(seq.identity_to_query_normalized),
                    "identity_to_medoid_normalized": round_identity(seq.identity_to_medoid_normalized),
                    "identity_to_nearest_non_member_normalized": round_identity(seq.identity_to_nearest_non_member_normalized)
                }
                for seq in self.sequences
            ],
            "diagnostics": {
                "method": self.method,
                "min_split": self.min_split,
                "max_lump": self.max_lump,
                "normalization_length": self.normalization_length,
                "identity_metric": self.distance_metric,
                "warnings": self.warnings
            }
        }


def _distance_to_identity(distance: Optional[float]) -> Optional[float]:
    """Convert distance (0-1) to identity percentage (0-100).

    Identity = (1 - distance) * 100
    e.g., distance 0.0037 → identity 99.63%
    """
    if distance is None or np.isnan(distance):
        return None
    return (1.0 - distance) * 100.0


class BlastAnalyzer:
    """
    Analyze BLAST results to identify sequences in the query's cluster.

    Uses gap-optimized target mode clustering to determine which sequences
    from a BLAST search cluster with the query sequence.
    """

    def __init__(self,
                 min_split: float = 0.005,
                 max_lump: float = 0.02,
                 target_percentile: int = 100,
                 show_progress: bool = True):
        """
        Initialize the BLAST analyzer.

        Args:
            min_split: Minimum distance to split clusters (sequences closer are lumped)
            max_lump: Maximum distance to lump clusters (sequences farther are split)
            target_percentile: Percentile for gap optimization (default: 100 for complete linkage)
            show_progress: Whether to show progress bars

        Raises:
            ValueError: If min_split or max_lump are negative, or if min_split >= max_lump
        """
        if min_split < 0:
            raise ValueError("min_split must be non-negative")
        if max_lump < 0:
            raise ValueError("max_lump must be non-negative")
        if min_split >= max_lump:
            raise ValueError("min_split must be less than max_lump")

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

        # Extract identities to query for all sequences
        identities_to_query = self._extract_identities_to_query(distance_provider, len(sequences))

        # Determine query cluster set (sequences in target cluster)
        query_cluster_set = set(target_cluster)

        # Find the medoid of the query cluster
        medoid_idx = self._find_query_cluster_medoid(
            distance_provider,
            list(query_cluster_set)
        )
        medoid_id = headers[medoid_idx]

        # Extract identities to medoid for all sequences
        identities_to_medoid = self._extract_identities_to_medoid(
            distance_provider,
            medoid_idx,
            len(sequences)
        )

        # Compute identities to nearest non-member for cluster members
        remaining_set = set(remaining)
        identities_to_nearest_non_member = self._compute_identities_to_nearest_non_member(
            distance_provider,
            query_cluster_set,
            remaining_set,
            len(sequences)
        )

        # Compute gap metrics
        gap_metrics = self._compute_gap_metrics(
            query_cluster_set,
            remaining_set,
            identities_to_query,
            metrics
        )

        # Classify each sequence
        sequence_results = self._classify_sequences(
            headers,
            query_cluster_set,
            identities_to_query,
            identities_to_medoid,
            identities_to_nearest_non_member
        )

        # Check for warnings
        if len(target_cluster) == 1:
            warnings.append("Query has no close matches")

        # Check for NaN identities (check the pairwise identity)
        nan_count = sum(1 for d in identities_to_query.values()
                        if d.get('pairwise') is None or
                        (isinstance(d.get('pairwise'), float) and np.isnan(d.get('pairwise'))))
        if nan_count > 0:
            warnings.append(f"{nan_count} sequences had insufficient overlap for distance calculation")

        return BlastAnalysisResult(
            query_id=query_id,
            query_length=query_length,
            total_sequences=len(sequences),
            query_cluster_size=len(target_cluster),
            barcode_gap_found=gap_metrics['gap_found'],
            gap_size_percent=gap_metrics['gap_size_percent'],
            medoid_id=medoid_id,
            medoid_index=medoid_idx,
            intra_cluster_identity=gap_metrics['intra_cluster_identity'],
            nearest_non_member_identity=gap_metrics['nearest_non_member_identity'],
            sequences=sequence_results,
            method="gap-optimized-target-clustering",
            min_split=self.min_split,
            max_lump=self.max_lump,
            normalization_length=distance_provider.normalization_length,
            distance_metric="MycoBLAST-adjusted (homopolymer-normalized, indel-normalized)",
            warnings=warnings
        )

    def _extract_identities_to_query(self,
                                      distance_provider: MSACachedDistanceProvider,
                                      n: int) -> Dict[int, Dict[str, Optional[float]]]:
        """Extract identities from all sequences to the query (index 0).

        Returns:
            Dict mapping sequence index to dict with 'pairwise' and 'normalized' identity %
        """
        identities = {}
        for i in range(n):
            if i == 0:
                identities[i] = {'pairwise': 100.0, 'normalized': 100.0}
            else:
                result = distance_provider.get_distance_detailed(0, i)
                if result.is_valid:
                    identities[i] = {
                        'pairwise': _distance_to_identity(result.distance_pairwise),
                        'normalized': _distance_to_identity(result.distance_normalized)
                    }
                else:
                    identities[i] = {'pairwise': None, 'normalized': None}
        return identities

    def _find_query_cluster_medoid(self,
                                    distance_provider: MSACachedDistanceProvider,
                                    query_cluster_indices: List[int]) -> int:
        """Find the medoid of the query cluster.

        The medoid is the sequence with minimum total distance to all other members.

        Args:
            distance_provider: MSA-based distance provider
            query_cluster_indices: List of sequence indices in the query cluster

        Returns:
            Index of the medoid sequence
        """
        if len(query_cluster_indices) == 0:
            raise ValueError("Cannot find medoid of empty cluster")

        if len(query_cluster_indices) == 1:
            return query_cluster_indices[0]

        min_total_distance = float('inf')
        medoid_idx = query_cluster_indices[0]

        for candidate_idx in query_cluster_indices:
            total_distance = 0.0
            for other_idx in query_cluster_indices:
                if candidate_idx != other_idx:
                    distance = distance_provider.get_distance(candidate_idx, other_idx)
                    if not np.isnan(distance):
                        total_distance += distance

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                medoid_idx = candidate_idx

        return medoid_idx

    def _extract_identities_to_medoid(self,
                                       distance_provider: MSACachedDistanceProvider,
                                       medoid_idx: int,
                                       n: int) -> Dict[int, Optional[float]]:
        """Extract normalized identities from all sequences to the medoid.

        Args:
            distance_provider: MSA-based distance provider
            medoid_idx: Index of the medoid sequence
            n: Total number of sequences

        Returns:
            Dict mapping sequence index to normalized identity % to medoid
        """
        identities = {}
        for i in range(n):
            if i == medoid_idx:
                identities[i] = 100.0
            else:
                result = distance_provider.get_distance_detailed(medoid_idx, i)
                if result.is_valid:
                    identities[i] = _distance_to_identity(result.distance_normalized)
                else:
                    identities[i] = None
        return identities

    def _compute_identities_to_nearest_non_member(self,
                                                   distance_provider: MSACachedDistanceProvider,
                                                   query_cluster_set: set,
                                                   non_member_set: set,
                                                   n: int) -> Dict[int, Optional[float]]:
        """Compute identity to nearest non-member for each cluster member.

        Args:
            distance_provider: MSA-based distance provider
            query_cluster_set: Set of indices in the query cluster
            non_member_set: Set of indices NOT in the query cluster
            n: Total number of sequences

        Returns:
            Dict mapping sequence index to identity % to nearest non-member.
            Only cluster members have values; non-members get None.
        """
        identities = {}

        # Non-members get None
        for idx in range(n):
            if idx not in query_cluster_set:
                identities[idx] = None
                continue

            # For cluster members, find the nearest non-member (highest identity)
            if not non_member_set:
                # No non-members exist
                identities[idx] = None
                continue

            min_distance = float('inf')
            for non_member_idx in non_member_set:
                result = distance_provider.get_distance_detailed(idx, non_member_idx)
                if result.is_valid and result.distance_normalized < min_distance:
                    min_distance = result.distance_normalized

            if min_distance == float('inf'):
                identities[idx] = None
            else:
                identities[idx] = _distance_to_identity(min_distance)

        return identities

    def _compute_gap_metrics(self,
                             query_cluster_set: set,
                             remaining_set: set,
                             identities_to_query: Dict[int, Dict[str, Optional[float]]],
                             clustering_metrics: Dict) -> Dict[str, Any]:
        """Compute barcode gap metrics from clustering results.

        Uses normalized identities since that's what the clustering algorithm uses.
        Gap is computed as: min intra-cluster identity - max inter-cluster identity.
        A positive gap means the lowest identity within the cluster is higher than
        the highest identity outside the cluster.
        """

        # Get intra-cluster identities (query cluster members to query)
        # Use normalized identities since that's what clustering uses
        intra_identities = []
        for idx in query_cluster_set:
            if idx == 0:
                continue  # Skip query itself
            id_info = identities_to_query.get(idx, {})
            identity = id_info.get('normalized')
            if identity is not None and not np.isnan(identity):
                intra_identities.append(identity)

        # Get inter-cluster identities (sequences outside query cluster to query)
        inter_identities = []
        for idx in remaining_set:
            id_info = identities_to_query.get(idx, {})
            identity = id_info.get('normalized')
            if identity is not None and not np.isnan(identity):
                inter_identities.append(identity)

        # Compute intra-cluster statistics (in identity %)
        if intra_identities:
            intra_stats = {
                "min": float(np.min(intra_identities)),
                "p5": float(np.percentile(intra_identities, 5)),
                "median": float(np.median(intra_identities)),
                "p95": float(np.percentile(intra_identities, 95)),
                "max": float(np.max(intra_identities))
            }
            intra_min = intra_stats["min"]
        else:
            intra_stats = {"min": None, "p5": None, "median": None, "p95": None, "max": None}
            intra_min = 100.0  # Default to 100% if no intra-cluster sequences

        # Compute nearest identity outside query cluster (highest identity = closest)
        if inter_identities:
            nearest_non_member = float(np.max(inter_identities))
        else:
            nearest_non_member = None

        # Calculate gap (in identity percentage points)
        # Gap = min intra identity - max inter identity
        # Positive means clear separation
        if nearest_non_member is not None and intra_min is not None:
            gap_size_percent = intra_min - nearest_non_member
            gap_found = gap_size_percent > 0
        else:
            gap_found = False
            gap_size_percent = None

        return {
            "gap_found": gap_found,
            "gap_size_percent": gap_size_percent,
            "intra_cluster_identity": intra_stats,
            "nearest_non_member_identity": nearest_non_member
        }

    def _classify_sequences(self,
                            headers: List[str],
                            query_cluster_set: set,
                            identities_to_query: Dict[int, Dict[str, Optional[float]]],
                            identities_to_medoid: Dict[int, Optional[float]],
                            identities_to_nearest_non_member: Dict[int, Optional[float]]) -> List[SequenceResult]:
        """Classify each sequence based on clustering results."""
        results = []

        for idx, header in enumerate(headers):
            in_cluster = idx in query_cluster_set
            id_info = identities_to_query.get(idx, {})

            results.append(SequenceResult(
                index=idx,
                id=header,
                in_query_cluster=in_cluster,
                identity_to_query=id_info.get('pairwise'),
                identity_to_query_normalized=id_info.get('normalized'),
                identity_to_medoid_normalized=identities_to_medoid.get(idx),
                identity_to_nearest_non_member_normalized=identities_to_nearest_non_member.get(idx)
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
                identity_to_query=100.0,
                identity_to_query_normalized=100.0,
                identity_to_medoid_normalized=100.0,  # Query is its own medoid
                identity_to_nearest_non_member_normalized=None  # No non-members exist
            ))

        return BlastAnalysisResult(
            query_id=query_id,
            query_length=query_length,
            total_sequences=len(sequences),
            query_cluster_size=1 if sequences else 0,
            barcode_gap_found=False,
            gap_size_percent=None,
            medoid_id=query_id if headers else None,
            medoid_index=0 if headers else None,
            intra_cluster_identity={"min": None, "p5": None, "median": None, "p95": None, "max": None},
            nearest_non_member_identity=None,
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
                identity_to_query=100.0 if i == 0 else None,
                identity_to_query_normalized=100.0 if i == 0 else None,
                identity_to_medoid_normalized=100.0 if i == 0 else None,  # Query is its own medoid
                identity_to_nearest_non_member_normalized=None  # Cannot compute without alignment
            ))

        return BlastAnalysisResult(
            query_id=query_id,
            query_length=query_length,
            total_sequences=len(sequences),
            query_cluster_size=1,
            barcode_gap_found=False,
            gap_size_percent=None,
            medoid_id=query_id if headers else None,
            medoid_index=0 if headers else None,
            intra_cluster_identity={"min": None, "p5": None, "median": None, "p95": None, "max": None},
            nearest_non_member_identity=None,
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

    if result.medoid_id is not None:
        medoid_is_query = " (query)" if result.medoid_index == 0 else ""
        lines.append(f"Cluster medoid: {result.medoid_id} (index {result.medoid_index}){medoid_is_query}")

    if result.gap_size_percent is not None:
        lines.append(f"Gap size: {result.gap_size_percent:.2f}%")

    if result.nearest_non_member_identity is not None:
        lines.append(f"Nearest outside cluster: {result.nearest_non_member_identity:.2f}%")

    intra = result.intra_cluster_identity
    if intra.get('median') is not None:
        lines.append(f"Intra-cluster identity: min={intra['min']:.2f}%, median={intra['median']:.2f}%, max={intra['max']:.2f}%")

    lines.append("")

    # Sequence classifications
    lines.append("-" * 40)
    lines.append("Sequence Classifications")
    lines.append("-" * 40)
    lines.append(f"{'Idx':<5} {'ID':<30} {'Member':<8} {'To Query':<10} {'To Medoid':<10} {'To Non-Mem':<10}")
    lines.append("-" * 85)

    for seq in result.sequences:
        in_cluster = "Yes" if seq.in_query_cluster else "No"
        id_query = f"{seq.identity_to_query_normalized:.2f}%" if seq.identity_to_query_normalized is not None else "N/A"
        id_medoid = f"{seq.identity_to_medoid_normalized:.2f}%" if seq.identity_to_medoid_normalized is not None else "N/A"
        id_non_mem = f"{seq.identity_to_nearest_non_member_normalized:.2f}%" if seq.identity_to_nearest_non_member_normalized is not None else "N/A"
        lines.append(f"{seq.index:<5} {seq.id:<30} {in_cluster:<8} {id_query:<10} {id_medoid:<10} {id_non_mem:<10}")

    lines.append("")

    # Diagnostics
    lines.append("-" * 40)
    lines.append("Diagnostics")
    lines.append("-" * 40)
    lines.append(f"Method: {result.method}")
    lines.append(f"min_split: {result.min_split}")
    lines.append(f"max_lump: {result.max_lump}")
    lines.append(f"Normalization length: {result.normalization_length} bp")
    lines.append(f"Identity metric: {result.distance_metric}")

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
    lines.append("index\tid\tin_query_cluster\tidentity_to_query\tidentity_to_query_normalized\tidentity_to_medoid_normalized\tidentity_to_nearest_non_member_normalized")

    # Data rows
    for seq in result.sequences:
        in_cluster = "true" if seq.in_query_cluster else "false"
        identity = f"{seq.identity_to_query:.2f}" if seq.identity_to_query is not None else ""
        identity_norm = f"{seq.identity_to_query_normalized:.2f}" if seq.identity_to_query_normalized is not None else ""
        identity_medoid = f"{seq.identity_to_medoid_normalized:.2f}" if seq.identity_to_medoid_normalized is not None else ""
        identity_non_mem = f"{seq.identity_to_nearest_non_member_normalized:.2f}" if seq.identity_to_nearest_non_member_normalized is not None else ""
        lines.append(f"{seq.index}\t{seq.id}\t{in_cluster}\t{identity}\t{identity_norm}\t{identity_medoid}\t{identity_non_mem}")

    return "\n".join(lines)
