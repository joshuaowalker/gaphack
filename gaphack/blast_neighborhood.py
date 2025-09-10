"""BLAST-based neighborhood finding for gaphack-decompose."""

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

logger = logging.getLogger(__name__)


@dataclass
class SequenceCandidate:
    """Represents a BLAST candidate for neighborhood discovery."""
    sequence_id: str
    sequence_hash: str
    blast_identity: float
    alignment_length: int
    e_value: float
    bit_score: float


class BlastNeighborhoodFinder:
    """BLAST-based neighborhood finder for sequence clustering decomposition."""
    
    def __init__(self, sequences: List[str], headers: List[str], cache_dir: Optional[Path] = None):
        """Initialize BLAST database from all sequences.
        
        Args:
            sequences: List of DNA sequences as strings
            headers: List of sequence headers/identifiers
            cache_dir: Directory for caching BLAST databases (default: temp directory)
        """
        if len(sequences) != len(headers):
            raise ValueError("Number of sequences and headers must match")
        
        self.sequences = sequences
        self.headers = headers
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "gaphack_decompose_blast_cache"
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
        self.blast_db_path = self.cache_dir / f"gaphack_decompose_db_{self.cache_key}"
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
        """Check if BLAST database files exist."""
        required_extensions = ['.nhr', '.nin', '.nsq']
        return all((self.blast_db_path.parent / f"{self.blast_db_path.name}{ext}").exists() 
                  for ext in required_extensions)
    
    def _create_database(self) -> None:
        """Create BLAST database from all sequences."""
        logger.info(f"Creating BLAST database with {len(self.sequences)} sequences")
        
        # Write sequences to temporary FASTA file
        # Use sequence hash as ID for fast lookup, handling collisions
        unique_sequences = {}  # hash -> sequence
        hash_collisions = 0
        
        with open(self.temp_fasta_path, 'w') as f:
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
        
        logger.info(f"Deduplicated {len(self.sequences)} sequences to {len(unique_sequences)} unique sequences for BLAST database")
        if hash_collisions > 0:
            logger.warning(f"Resolved {hash_collisions} hash collisions by creating unique sequence IDs")
        
        # Create BLAST database
        cmd = [
            'makeblastdb',
            '-in', str(self.temp_fasta_path),
            '-dbtype', 'nucl',
            '-out', str(self.blast_db_path),
            '-parse_seqids'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("BLAST database created successfully")
            logger.debug(f"makeblastdb output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create BLAST database: {e}")
            logger.error(f"makeblastdb stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError("makeblastdb command not found. Please ensure NCBI BLAST+ is installed.")
    
    def find_neighborhood(self, target_headers: List[str], 
                         max_hits: int = 500,
                         e_value_threshold: float = 1e-5,
                         min_identity: Optional[float] = None,
                         min_query_coverage: float = 50.0,
                         min_alignment_length: int = 200) -> List[str]:
        """Find neighborhood sequences for given target headers.
        
        Args:
            target_headers: List of sequence headers to find neighborhoods for
            max_hits: Maximum number of BLAST hits to return per target
            e_value_threshold: E-value threshold for BLAST search
            min_identity: Minimum percent identity (auto-calculated if None)
            min_query_coverage: Minimum query coverage percentage
            min_alignment_length: Minimum alignment length in base pairs
            
        Returns:
            List of sequence headers in the neighborhood (including targets)
        """
        if not self._is_database_cached():
            raise RuntimeError("BLAST database not available")
        
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
        
        # Calculate threshold if not provided (simple heuristic for now)
        if min_identity is None:
            min_identity = 80.0  # Conservative default for neighborhood discovery
            logger.debug(f"Using default BLAST identity threshold: {min_identity:.1f}%")
        
        # Run BLAST search for targets
        candidates_by_target = self._get_candidates_for_sequences(
            target_sequences, max_hits, e_value_threshold, min_identity,
            min_query_coverage, min_alignment_length
        )
        
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
    
    def _get_candidates_for_sequences(self, query_sequences: List[Tuple[str, str]], 
                                    max_targets: int = 500,
                                    e_value_threshold: float = 1e-5,
                                    min_identity: float = 80.0,
                                    min_query_coverage: float = 50.0,
                                    min_alignment_length: int = 200) -> Dict[str, List[SequenceCandidate]]:
        """Get BLAST candidates for query sequences."""
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
            
            # Run BLAST search
            results_by_query = self._run_blast_search(
                batch_query_file, max_targets, e_value_threshold, min_identity,
                min_query_coverage, min_alignment_length, len(query_sequences)
            )
            
            # Ensure all queries have entries (even if no hits)
            for seq_id, _ in query_sequences:
                if seq_id not in results_by_query:
                    results_by_query[seq_id] = []
            
            return results_by_query
            
        finally:
            # Clean up temporary query file
            if batch_query_file.exists():
                batch_query_file.unlink()
    
    def _run_blast_search(self, batch_query_file: Path, max_targets: int, 
                         e_value_threshold: float, min_identity: float,
                         min_query_coverage: float, min_alignment_length: int,
                         num_queries: int) -> Dict[str, List[SequenceCandidate]]:
        """Run BLAST search and parse results."""
        
        # Run BLAST search with batch queries and multi-threading
        num_threads = min(os.cpu_count() or 1, num_queries, 8)  # Cap at 8 threads
        cmd = [
            'blastn',
            '-query', str(batch_query_file),
            '-db', str(self.blast_db_path),
            '-outfmt', '6 qseqid sseqid qcovs pident length evalue bitscore',
            '-max_target_seqs', str(max_targets),
            '-evalue', str(e_value_threshold),
            '-perc_identity', str(min_identity),
            '-num_threads', str(num_threads)
        ]
        
        logger.debug(f"Running BLAST with {num_threads} threads for {num_queries} queries")
        
        try:
            blast_start_time = time.time()
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            blast_time = time.time() - blast_start_time
            logger.debug(f"BLAST search completed in {blast_time:.2f}s")
        except subprocess.CalledProcessError as e:
            logger.error(f"BLAST search failed: {e}")
            logger.error(f"blastn stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError("blastn command not found. Please ensure NCBI BLAST+ is installed.")
        
        # Parse results
        results_by_query = {}
        total_hits = 0
        filtered_hits = 0
        
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 7:
                continue
            
            total_hits += 1
            
            query_id = parts[0]
            subject_id = parts[1]  # Sequence hash
            query_coverage = float(parts[2])
            percent_identity = float(parts[3])
            alignment_length = int(parts[4])
            e_value = float(parts[5])
            bit_score = float(parts[6])
            
            # Apply additional filtering
            if alignment_length < min_alignment_length:
                filtered_hits += 1
                continue
            
            if query_id not in results_by_query:
                results_by_query[query_id] = []
            
            candidate = SequenceCandidate(
                sequence_id=query_id,
                sequence_hash=subject_id,
                blast_identity=percent_identity,
                alignment_length=alignment_length,
                e_value=e_value,
                bit_score=bit_score
            )
            results_by_query[query_id].append(candidate)
        
        # Sort each query's candidates by bit score
        for query_id in results_by_query:
            results_by_query[query_id].sort(key=lambda x: x.bit_score, reverse=True)
        
        kept_hits = total_hits - filtered_hits
        logger.debug(f"BLAST filtering: {total_hits} total hits → {kept_hits} kept ({filtered_hits} filtered by length)")
        logger.debug(f"BLAST search for {num_queries} queries returned results for {len(results_by_query)} queries")
        
        return results_by_query
    
    def cleanup(self) -> None:
        """Remove temporary files and cached database."""
        try:
            # Remove temporary FASTA file
            if self.temp_fasta_path.exists():
                self.temp_fasta_path.unlink()
            
            # Remove BLAST database files
            blast_extensions = ['.nhr', '.nin', '.nsq', '.ndb', '.not', '.ntf', '.nto']
            for ext in blast_extensions:
                db_file = Path(f"{self.blast_db_path}{ext}")
                if db_file.exists():
                    db_file.unlink()
                    
            logger.debug("BLAST database cleanup completed")
        except Exception as e:
            logger.warning(f"Error during BLAST database cleanup: {e}")