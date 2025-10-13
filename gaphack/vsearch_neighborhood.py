"""vsearch-based neighborhood finding for gaphack cluster refinement."""

import hashlib
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .neighborhood_finder import NeighborhoodFinder

logger = logging.getLogger(__name__)


@dataclass
class SequenceCandidate:
    """Represents a vsearch candidate for neighborhood discovery."""
    sequence_id: str
    sequence_hash: str
    blast_identity: float  # Keep name for compatibility, contains vsearch identity
    alignment_length: int
    e_value: float  # Not used by vsearch, always 0.0
    bit_score: float  # Not used by vsearch, always 0.0


class VsearchNeighborhoodFinder(NeighborhoodFinder):
    """vsearch-based neighborhood finder for sequence clustering decomposition."""

    def __init__(self, sequences: List[str], headers: List[str],
                 output_dir: Optional[Path] = None):
        """Initialize vsearch database from all sequences.

        Args:
            sequences: List of DNA sequences as strings
            headers: List of sequence headers/identifiers
            output_dir: Output directory for decompose run (vsearch DB goes in {output_dir}/.vsearch/)
        """
        super().__init__(sequences, headers, output_dir)

        # Use output_dir/.vsearch/ if provided, otherwise fall back to temp directory
        if output_dir:
            self.cache_dir = Path(output_dir) / ".vsearch"
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "gaphack_decompose_vsearch_cache"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create sequence lookup: hash -> (sequence, header) for fast lookup
        self.sequence_lookup = {}
        self.header_to_index = {header: i for i, header in enumerate(headers)}

        for i, (seq, header) in enumerate(zip(sequences, headers)):
            seq_hash = self._hash_sequence(seq)
            if seq_hash not in self.sequence_lookup:
                self.sequence_lookup[seq_hash] = []
            self.sequence_lookup[seq_hash].append((seq, header, i))

        # Generate cache key based on sequence set content
        self.cache_key = self._get_cache_key()
        self.vsearch_db_path = self.cache_dir / f"gaphack_decompose_db_{self.cache_key}.fasta"
        self.temp_fasta_path = self.cache_dir / f"gaphack_decompose_seqs_{self.cache_key}.fasta"

        # Create database if not cached
        if not self._is_database_cached():
            self._create_database()

    def _hash_sequence(self, sequence: str) -> str:
        """Generate hash for sequence."""
        return hashlib.sha256(sequence.upper().encode()).hexdigest()[:16]

    def _get_cache_key(self) -> str:
        """Generate cache key based on sequence set content."""
        # Sort sequences by hash for consistent ordering
        seq_hashes = sorted([self._hash_sequence(seq) for seq in self.sequences])

        # Create hash from sequence hashes
        content_hash = hashlib.sha256()
        for seq_hash in seq_hashes:
            content_hash.update(seq_hash.encode())

        return content_hash.hexdigest()[:12]

    def _is_database_cached(self) -> bool:
        """Check if vsearch database file exists."""
        return self.vsearch_db_path.exists()

    def _create_database(self) -> None:
        """Create vsearch database from all sequences."""
        logger.info(f"Creating vsearch database with {len(self.sequences)} sequences")

        # Write sequences to FASTA file (vsearch can use FASTA directly)
        # Use sequence hash as ID for fast lookup, handling collisions
        unique_sequences = {}  # hash -> sequence
        hash_collisions = 0

        with open(self.vsearch_db_path, 'w') as f:
            for i, (seq, header) in enumerate(zip(self.sequences, self.headers)):
                seq_hash = self._hash_sequence(seq)

                if seq_hash not in unique_sequences:
                    # First encounter of this hash - store sequence
                    unique_sequences[seq_hash] = seq
                    record = SeqRecord(
                        Seq(seq),
                        id=seq_hash,
                        description=f"{header}"
                    )
                    SeqIO.write(record, f, "fasta")
                else:
                    # Validate that sequences match for same hash
                    if unique_sequences[seq_hash] != seq:
                        hash_collisions += 1
                        logger.warning(f"Hash collision detected: {seq_hash} "
                                     f"maps to different sequences (header: {header})")
                        # Use hash with suffix to make unique
                        collision_hash = f"{seq_hash}_collision_{hash_collisions}"
                        unique_sequences[collision_hash] = seq

                        # Update lookup table for this sequence
                        if collision_hash not in self.sequence_lookup:
                            self.sequence_lookup[collision_hash] = []
                        self.sequence_lookup[collision_hash].append((seq, header, i))

                        record = SeqRecord(
                            Seq(seq),
                            id=collision_hash,
                            description=f"{header} (collision_resolved)"
                        )
                        SeqIO.write(record, f, "fasta")

        logger.info(f"Deduplicated {len(self.sequences)} sequences to {len(unique_sequences)} unique sequences for vsearch database")
        if hash_collisions > 0:
            logger.warning(f"Resolved {hash_collisions} hash collisions by creating unique sequence IDs")

        # vsearch uses FASTA directly (no separate database indexing step like BLAST)
        logger.info("vsearch database (FASTA) created successfully")

    def find_neighborhood(self, target_headers: List[str], max_hits: int = 1000, e_value_threshold: float = 1e-5,
                          min_identity: Optional[float] = None) -> List[str]:
        """Find neighborhood sequences for given target headers.

        Args:
            target_headers: List of sequence headers to find neighborhoods for
            max_hits: Maximum number of vsearch hits to return per target
            e_value_threshold: Not used by vsearch (kept for interface compatibility)
            min_identity: Minimum percent identity (auto-calculated if None)

        Returns:
            List of sequence headers in the neighborhood (including targets)
        """
        if not self._is_database_cached():
            raise RuntimeError("vsearch database not available")

        if not target_headers:
            return []

        # Get target sequences
        target_sequences = []
        for header in target_headers:
            if header not in self.header_to_index:
                logger.warning(f"Target header '{header}' not found in sequences")
                continue
            idx = self.header_to_index[header]
            target_sequences.append((header, self.sequences[idx]))

        if not target_sequences:
            logger.warning("No valid target sequences found")
            return []

        # Calculate threshold if not provided
        if min_identity is None:
            min_identity = 90.0  # Conservative default for neighborhood discovery
            logger.debug(f"Using default vsearch identity threshold: {min_identity:.1f}%")

        # Run vsearch search for targets
        candidates_by_target = self._get_candidates_for_sequences(target_sequences, max_hits, min_identity)

        # Collect all unique sequence headers from neighborhoods
        neighborhood_headers = set()

        for target_header, candidates in candidates_by_target.items():
            # Always include the target itself
            neighborhood_headers.add(target_header)

            # Add candidate sequences
            for candidate in candidates:
                # Look up all sequences with this hash (handle duplicates)
                sequences_with_hash = self.sequence_lookup.get(candidate.sequence_hash, [])
                for seq, header, idx in sequences_with_hash:
                    neighborhood_headers.add(header)

        neighborhood_list = sorted(neighborhood_headers)
        logger.debug(f"Found neighborhood of {len(neighborhood_list)} sequences for {len(target_headers)} targets")

        return neighborhood_list

    def _get_candidates_for_sequences(self, query_sequences: List[Tuple[str, str]], max_targets: int = 1000,
                                      e_value_threshold: float = 1e-5, min_identity: float = 90.0) -> Dict[str, List[SequenceCandidate]]:
        """Get vsearch candidates for query sequences.

        Note: e_value_threshold parameter is ignored (for compatibility with BLAST interface).
        """
        # Create temporary batch query file
        batch_query_file = self.cache_dir / f"neighborhood_query_{hash(tuple(seq_id for seq_id, _ in query_sequences))}.fasta"

        try:
            # Write query sequences to file
            with open(batch_query_file, 'w') as f:
                for seq_id, sequence in query_sequences:
                    record = SeqRecord(
                        Seq(sequence),
                        id=seq_id,
                        description=""
                    )
                    SeqIO.write(record, f, "fasta")

            # Run vsearch search
            results_by_query = self._run_vsearch_search(batch_query_file, max_targets, min_identity,
                                                        len(query_sequences))

            # Ensure all queries have entries (even if no hits)
            for seq_id, _ in query_sequences:
                if seq_id not in results_by_query:
                    results_by_query[seq_id] = []

            return results_by_query

        finally:
            # Clean up temporary query file
            if batch_query_file.exists():
                batch_query_file.unlink()

    def _run_vsearch_search(self, batch_query_file: Path, max_targets: int, min_identity: float,
                            num_queries: int) -> Dict[str, List[SequenceCandidate]]:
        """Run vsearch search and parse results."""

        # Convert identity from 0-100 to 0.0-1.0 for vsearch
        vsearch_identity = min_identity / 100.0

        # Run vsearch search with batch queries and multi-threading
        num_threads = min(os.cpu_count() or 1, num_queries, 8)  # Cap at 8 threads
        cmd = [
            'vsearch',
            '--usearch_global', str(batch_query_file),
            '--db', str(self.vsearch_db_path),
            '--userout', '/dev/stdout',
            '--userfields', 'query+target+id+alnlen+qcov',
            '--id', str(vsearch_identity),
            '--maxaccepts', str(max_targets),
            '--threads', str(num_threads),
            '--output_no_hits'
        ]

        logger.debug(f"Running vsearch with {num_threads} threads for {num_queries} queries")

        try:
            vsearch_start_time = time.time()
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            vsearch_time = time.time() - vsearch_start_time
            logger.debug(f"vsearch search completed in {vsearch_time:.2f}s")
        except subprocess.CalledProcessError as e:
            logger.error(f"vsearch search failed: {e}")
            logger.error(f"vsearch stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError("vsearch command not found. Please ensure vsearch is installed.")

        # Parse results
        results_by_query = {}
        total_hits = 0

        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 5:
                continue

            total_hits += 1

            query_id = parts[0]
            subject_id = parts[1]  # Sequence hash
            percent_identity = float(parts[2])  # vsearch outputs as percentage (0-100)
            alignment_length = int(parts[3])
            query_coverage = float(parts[4])

            if query_id not in results_by_query:
                results_by_query[query_id] = []

            candidate = SequenceCandidate(
                sequence_id=query_id,
                sequence_hash=subject_id,
                blast_identity=percent_identity,
                alignment_length=alignment_length,
                e_value=0.0,  # vsearch doesn't compute e-value
                bit_score=percent_identity  # Use identity for ranking (higher = better)
            )
            results_by_query[query_id].append(candidate)

        # Sort each query's candidates by percent identity (higher = better)
        for query_id in results_by_query:
            results_by_query[query_id].sort(key=lambda x: x.blast_identity, reverse=True)

        logger.debug(f"vsearch search for {num_queries} queries returned {total_hits} total hits")
        logger.debug(f"vsearch search returned results for {len(results_by_query)} queries")

        return results_by_query

    def cleanup(self) -> None:
        """Remove temporary files and cached database."""
        try:
            # Remove vsearch database (FASTA file)
            if self.vsearch_db_path.exists():
                self.vsearch_db_path.unlink()

            logger.debug("vsearch database cleanup completed")
        except Exception as e:
            logger.warning(f"Error during vsearch database cleanup: {e}")
