"""
Utility functions for gapHACk.

This module provides helper functions for sequence processing and distance calculations.
"""

import numpy as np
from typing import List, Tuple, Optional, Literal, Dict
from Bio import SeqIO
from Bio.Seq import Seq
import logging
from tqdm import tqdm
import hashlib


def load_sequences_from_fasta(fasta_path: str) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Load sequences from a FASTA file.
    
    Args:
        fasta_path: Path to the FASTA file
        
    Returns:
        Tuple of (sequences, headers, header_mapping) where:
        - sequences is a list of DNA strings
        - headers is a list of sequence identifiers  
        - header_mapping is a dict from sequence ID to full header (ID + description)
    """
    sequences = []
    headers = []
    header_mapping = {}
    
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq).upper())
            seq_id = record.id
            headers.append(seq_id)
            
            # Preserve full header: ID + description if present
            full_header = record.description if record.description else record.id
            header_mapping[seq_id] = full_header
    except Exception as e:
        logging.error(f"Error reading FASTA file: {e}")
        raise
    
    return sequences, headers, header_mapping


def load_sequences_with_deduplication(fasta_path: str) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    Load sequences from a FASTA file with content-based deduplication.

    Args:
        fasta_path: Path to the FASTA file

    Returns:
        Tuple of (unique_sequences, hash_ids, hash_to_headers_map) where:
        - unique_sequences is a list of deduplicated DNA strings
        - hash_ids is a list of hash identifiers for each unique sequence
        - hash_to_headers_map is a dict from hash_id to list of original headers
    """
    sequence_to_hash = {}  # normalized_seq -> hash_id
    unique_sequences = []
    hash_to_headers = {}   # hash_id -> [original_header1, original_header2, ...]

    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            # Normalize sequence (uppercase, strip whitespace)
            normalized_seq = str(record.seq).upper().strip()

            # Create deterministic hash for sequence content
            if normalized_seq not in sequence_to_hash:
                # Use SHA-256 for a robust hash, truncated to 16 chars for readability
                seq_hash = hashlib.sha256(normalized_seq.encode()).hexdigest()[:16]
                hash_id = f"seq_{seq_hash}"

                # Handle hash collisions (extremely unlikely but robust)
                collision_counter = 1
                base_hash_id = hash_id
                while hash_id in hash_to_headers:
                    hash_id = f"{base_hash_id}_{collision_counter}"
                    collision_counter += 1

                sequence_to_hash[normalized_seq] = hash_id
                unique_sequences.append(normalized_seq)
                hash_to_headers[hash_id] = []

            # Add this header to the hash mapping
            hash_id = sequence_to_hash[normalized_seq]

            # Preserve full header: ID + description if present
            full_header = record.description if record.description else record.id
            hash_to_headers[hash_id].append(full_header)

    except Exception as e:
        logging.error(f"Error reading FASTA file: {e}")
        raise

    # Create list of hash_ids in same order as unique_sequences
    hash_ids = [sequence_to_hash[seq] for seq in unique_sequences]

    # Log deduplication statistics
    total_sequences = sum(len(headers) for headers in hash_to_headers.values())
    duplicates = total_sequences - len(unique_sequences)
    if duplicates > 0:
        logging.info(f"Deduplicated {total_sequences} sequences to {len(unique_sequences)} unique sequences "
                    f"({duplicates} duplicates removed)")

    return unique_sequences, hash_ids, hash_to_headers


def calculate_distance_matrix(sequences: List[str], 
                            alignment_method: Literal["adjusted", "traditional"] = "adjusted",
                            end_skip_distance: int = 20,
                            normalize_homopolymers: bool = True,
                            handle_iupac_overlap: bool = True,
                            normalize_indels: bool = True,
                            max_repeat_motif_length: int = 2,
                            show_progress: bool = True) -> np.ndarray:
    """
    Calculate pairwise distance matrix for sequences using adjusted-identity.
    
    Args:
        sequences: List of DNA sequences as strings
        alignment_method: Either "adjusted" (with MycoBLAST adjustments) or "traditional" (raw BLAST-like)
        end_skip_distance: Distance from sequence ends to skip in alignment (for adjusted method)
        normalize_homopolymers: Whether to ignore homopolymer length differences (for adjusted method)
        handle_iupac_overlap: Whether to allow IUPAC ambiguity codes to match via intersection (for adjusted method)
        normalize_indels: Whether to count contiguous indels as single events (for adjusted method)
        max_repeat_motif_length: Maximum length of repeat motifs to detect (for adjusted method)
        show_progress: Whether to show progress bar for distance calculations
        
    Returns:
        Numpy array of pairwise distances (n x n)
    """
    try:
        from adjusted_identity import align_and_score, AdjustmentParams, RAW_ADJUSTMENT_PARAMS
    except ImportError:
        raise ImportError(
            "adjusted-identity package is required. "
            "Install it with: pip install git+https://github.com/joshuaowalker/adjusted-identity.git"
        )
    
    n = len(sequences)
    distance_matrix = np.zeros((n, n))
    
    # Choose alignment parameters based on method
    if alignment_method == "adjusted":
        params = AdjustmentParams(
            end_skip_distance=end_skip_distance,
            normalize_homopolymers=normalize_homopolymers,
            handle_iupac_overlap=handle_iupac_overlap,
            normalize_indels=normalize_indels,
            max_repeat_motif_length=max_repeat_motif_length
        )
        adjustments = []
        if normalize_homopolymers: adjustments.append("homopolymer_norm")
        if handle_iupac_overlap: adjustments.append("iupac_overlap")
        if normalize_indels: adjustments.append("indel_norm")
        adjustments_str = f"({', '.join(adjustments)})" if adjustments else "(no adjustments)"
        logging.debug(f"Using adjusted identity with MycoBLAST adjustments {adjustments_str}, "
                    f"end_skip_distance={end_skip_distance}, max_repeat_motif_length={max_repeat_motif_length}")
    else:  # traditional
        params = RAW_ADJUSTMENT_PARAMS
        logging.info("Using traditional BLAST-like identity calculation")
    
    # Calculate total number of comparisons for progress bar
    total_comparisons = (n * (n - 1)) // 2
    
    # Create progress bar if requested and meaningful
    pbar = None
    if show_progress and total_comparisons > 0:
        pbar = tqdm(total=total_comparisons, 
                    desc="Calculating pairwise distances", 
                    unit=" comparisons")
    
    for i in range(n):
        for j in range(i + 1, n):
            try:
                # Pass shortest sequence first for consistent infix alignment
                seq_i, seq_j = sequences[i], sequences[j]
                if len(seq_i) <= len(seq_j):
                    result = align_and_score(seq_i, seq_j, params)
                else:
                    result = align_and_score(seq_j, seq_i, params)
                dist = 1.0 - result.identity
            except Exception as e:
                logging.warning(f"Alignment failed for sequences {i} and {j}: {e}")
                dist = 1.0  # Maximum distance for failed alignments
            
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
            if pbar:
                pbar.update(1)
    
    if pbar:
        pbar.close()
    
    return distance_matrix




def format_cluster_output(clusters: List[List[int]], 
                         singletons: List[int],
                         headers: Optional[List[str]] = None) -> str:
    """
    Format clustering results for output.
    
    Args:
        clusters: List of clusters, each containing sequence indices
        singletons: List of singleton sequence indices
        headers: Optional list of sequence headers/names
        
    Returns:
        Formatted string representation of clusters
    """
    output_lines = []
    
    # Format clusters
    for i, cluster in enumerate(clusters, 1):
        cluster_members = []
        for idx in cluster:
            if headers:
                cluster_members.append(headers[idx])
            else:
                cluster_members.append(f"seq_{idx}")
        
        output_lines.append(f"Cluster {i} ({len(cluster)} sequences):")
        for member in cluster_members:
            output_lines.append(f"  - {member}")
    
    # Format singletons
    if singletons:
        output_lines.append(f"\nSingletons ({len(singletons)} sequences):")
        for idx in singletons:
            if headers:
                output_lines.append(f"  - {headers[idx]}")
            else:
                output_lines.append(f"  - seq_{idx}")
    
    return "\n".join(output_lines)


def save_clusters_to_file(clusters: List[List[int]], 
                         singletons: List[int],
                         output_path: str,
                         headers: Optional[List[str]] = None,
                         sequences: Optional[List[str]] = None,
                         format: str = "fasta",
                         singleton_label: str = "singleton"):
    """
    Save clustering results to files.
    
    Args:
        clusters: List of clusters, each containing sequence indices
        singletons: List of singleton sequence indices
        output_path: Path to output file (or base path for FASTA format)
        headers: Optional list of sequence headers/names
        sequences: Optional list of sequences (required for FASTA format)
        format: Output format ("fasta", "tsv", or "text")
        singleton_label: Label for singleton sequences ("singleton" or "unclustered")
    """
    if format == "fasta":
        # FASTA format: create one file per cluster and one for singletons
        if sequences is None:
            raise ValueError("Sequences are required for FASTA format output")
        
        from pathlib import Path
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        
        # Get base path without extension
        output_base = Path(output_path)
        if output_base.suffix.lower() == '.fasta' or output_base.suffix.lower() == '.fa':
            output_base = output_base.with_suffix('')
        
        # Calculate padding width based on number of clusters
        total_clusters = len(clusters) + (1 if singletons else 0)  # +1 for singletons file
        pad_width = max(3, len(str(total_clusters)))  # At least 3 digits
        
        # Write each cluster to a separate FASTA file
        for cluster_idx, cluster in enumerate(clusters, 1):
            cluster_name = f"cluster_{str(cluster_idx).zfill(pad_width)}"
            cluster_file = f"{output_base}.{cluster_name}.fasta"
            
            records = []
            for seq_idx in cluster:
                header = headers[seq_idx] if headers else f"seq_{seq_idx}"
                # Add cluster info to header
                record = SeqRecord(
                    Seq(sequences[seq_idx]),
                    id=header,
                    description=f"{cluster_name}"
                )
                records.append(record)
            
            with open(cluster_file, 'w') as f:
                SeqIO.write(records, f, "fasta-2line")
            
            logging.debug(f"Wrote {len(records)} sequences to {cluster_file}")
        
        # Write singletons/unclustered to a separate file
        if singletons:
            singleton_file = f"{output_base}.{singleton_label}s.fasta"
            records = []
            for seq_idx in singletons:
                header = headers[seq_idx] if headers else f"seq_{seq_idx}"
                record = SeqRecord(
                    Seq(sequences[seq_idx]),
                    id=header,
                    description=singleton_label
                )
                records.append(record)
            
            with open(singleton_file, 'w') as f:
                SeqIO.write(records, f, "fasta-2line")
            
            logging.debug(f"Wrote {len(records)} {singleton_label} sequences to {singleton_file}")
    
    elif format == "tsv":
        # Tab-separated format: sequence_id<tab>cluster_id
        with open(output_path, 'w') as f:
            f.write("sequence_id\tcluster_id\n")
            
            # Write clusters
            for cluster_id, cluster in enumerate(clusters, 1):
                for idx in cluster:
                    seq_id = headers[idx] if headers else f"seq_{idx}"
                    f.write(f"{seq_id}\tcluster_{cluster_id}\n")
            
            # Write singletons/unclustered
            for idx in singletons:
                seq_id = headers[idx] if headers else f"seq_{idx}"
                f.write(f"{seq_id}\t{singleton_label}\n")
    
    else:  # text format
        output_text = format_cluster_output(clusters, singletons, headers)
        with open(output_path, 'w') as f:
            f.write(output_text)


def validate_sequences(sequences: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate DNA sequences.
    
    Args:
        sequences: List of DNA sequences
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    # Include IUPAC ambiguous nucleotide codes
    valid_bases = set('ATCGNRYKMSWBDHV-')
    errors = []
    
    if not sequences:
        return False, ["No sequences provided"]
    
    for i, seq in enumerate(sequences):
        if not seq:
            errors.append(f"Sequence {i+1} is empty")
            continue
        
        invalid_chars = set(seq.upper()) - valid_bases
        if invalid_chars:
            errors.append(f"Sequence {i+1} contains invalid characters: {invalid_chars}")
    
    return len(errors) == 0, errors