"""
Utility functions for gapHACk.

This module provides helper functions for sequence processing and distance calculations.
"""

import numpy as np
from typing import List, Tuple, Optional, Literal, Dict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import logging
from tqdm import tqdm
import hashlib
import subprocess
import tempfile
import os


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


def run_spoa_msa(sequences: List[str]) -> Optional[List[str]]:
    """Run SPOA to create multiple sequence alignment.

    Args:
        sequences: List of DNA sequences

    Returns:
        List of aligned sequences (with gaps) in same order as input, or None if SPOA fails
    """
    if not sequences:
        return None

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as temp_input:
        try:
            # Write sequences to temporary file with simple numeric IDs
            seq_records = [
                SeqRecord(Seq(seq), id=str(i), description="")
                for i, seq in enumerate(sequences)
            ]
            SeqIO.write(seq_records, temp_input, "fasta")
            temp_input.flush()

            # Run SPOA with -r 2 for FASTA alignment output
            result = subprocess.run(
                ['spoa', temp_input.name, '-r', '2'],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse aligned sequences from SPOA output
            aligned_sequences = {}
            lines = result.stdout.strip().split('\n')
            current_id = None
            current_seq = []

            for line in lines:
                if line.startswith('>'):
                    if current_id is not None and not current_id.startswith('Consensus'):
                        aligned_sequences[current_id] = ''.join(current_seq)
                    current_id = line[1:].strip()
                    current_seq = []
                elif line.strip():
                    current_seq.append(line.strip())

            # Add the last sequence (skip if consensus)
            if current_id is not None and not current_id.startswith('Consensus'):
                aligned_sequences[current_id] = ''.join(current_seq)

            # Return sequences in original order
            result_sequences = []
            for i in range(len(sequences)):
                if str(i) in aligned_sequences:
                    result_sequences.append(aligned_sequences[str(i)])
                else:
                    logging.warning(f"SPOA did not return alignment for sequence {i}")
                    return None

            return result_sequences

        except subprocess.CalledProcessError as e:
            logging.warning(f"SPOA alignment failed: {e}")
            return None
        except Exception as e:
            logging.warning(f"Error running SPOA: {e}")
            return None
        finally:
            # Clean up temporary file
            if os.path.exists(temp_input.name):
                os.unlink(temp_input.name)


def replace_terminal_gaps(aligned_sequences: List[str]) -> List[str]:
    """Replace leading and trailing '-' gaps with '.' to mark terminal gaps.

    Args:
        aligned_sequences: List of aligned sequences with gap characters

    Returns:
        List of sequences with terminal gaps marked as '.'
    """
    processed = []

    for seq in aligned_sequences:
        seq_list = list(seq)

        # Find first non-gap character
        first_base = None
        for i, char in enumerate(seq_list):
            if char != '-':
                first_base = i
                break

        # Find last non-gap character
        last_base = None
        for i in range(len(seq_list) - 1, -1, -1):
            if seq_list[i] != '-':
                last_base = i
                break

        # Replace leading gaps with '.'
        if first_base is not None:
            for i in range(first_base):
                if seq_list[i] == '-':
                    seq_list[i] = '.'

        # Replace trailing gaps with '.'
        if last_base is not None:
            for i in range(last_base + 1, len(seq_list)):
                if seq_list[i] == '-':
                    seq_list[i] = '.'

        processed.append(''.join(seq_list))

    return processed


# Coverage threshold for alignment trimming
MSA_COVERAGE_THRESHOLD = 0.5


class MSAAlignmentError(Exception):
    """Raised when MSA alignment processing fails."""
    pass


def trim_alignment_by_coverage(aligned_sequences: List[str],
                               coverage_threshold: float = MSA_COVERAGE_THRESHOLD) -> List[str]:
    """Trim alignment to region where sufficient sequences have data.

    Finds the leftmost and rightmost positions where at least
    coverage_threshold fraction of sequences have non-terminal-gap
    characters (bases or internal gaps '-'), then trims all sequences
    to this region. Terminal gaps ('.' characters) indicate missing data.

    This addresses issues where different noisy sequence tails cause
    SPOA to open large gaps, which penalizes pairwise distances between
    sequences with different tail patterns.

    Args:
        aligned_sequences: List of aligned sequences with terminal gaps marked as '.'
        coverage_threshold: Minimum fraction of sequences that must have data at
                          a position for it to be included (default: 0.5)

    Returns:
        Trimmed aligned sequences

    Raises:
        MSAAlignmentError: If no positions meet the coverage threshold
    """
    if not aligned_sequences:
        raise MSAAlignmentError("Cannot trim empty alignment")

    n_sequences = len(aligned_sequences)
    alignment_length = len(aligned_sequences[0])
    min_coverage = int(coverage_threshold * n_sequences)

    # Count sequences with data (non-terminal-gap) at each position
    coverage_counts = []
    for pos in range(alignment_length):
        count = sum(1 for seq in aligned_sequences if seq[pos] != '.')
        coverage_counts.append(count)

    # Find first position that meets threshold
    first_valid = None
    for pos in range(alignment_length):
        if coverage_counts[pos] >= min_coverage:
            first_valid = pos
            break

    # Find last position that meets threshold
    last_valid = None
    for pos in range(alignment_length - 1, -1, -1):
        if coverage_counts[pos] >= min_coverage:
            last_valid = pos
            break

    # Check if we found a valid region
    if first_valid is None or last_valid is None or first_valid > last_valid:
        raise MSAAlignmentError(
            f"No alignment region with >={coverage_threshold:.0%} coverage found. "
            f"Alignment has {n_sequences} sequences with {alignment_length}bp length, "
            f"requiring >={min_coverage} sequences with data at each position. "
            f"This indicates extremely divergent sequences with insufficient overlap."
        )

    # Trim all sequences to the valid region
    trimmed_sequences = [seq[first_valid:last_valid + 1] for seq in aligned_sequences]

    # Log trimming information
    original_length = alignment_length
    trimmed_length = last_valid - first_valid + 1
    left_trim = first_valid
    right_trim = original_length - last_valid - 1

    logging.debug(
        f"Trimmed alignment from {original_length}bp to {trimmed_length}bp "
        f"({original_length - trimmed_length}bp removed: "
        f"{left_trim}bp left, {right_trim}bp right) "
        f"using {coverage_threshold:.0%} coverage threshold"
    )

    return trimmed_sequences


def filter_msa_positions(seq1_aligned: str, seq2_aligned: str) -> Tuple[str, str]:
    """Remove positions unsuitable for pairwise scoring from MSA.

    Removes positions where:
    - Either sequence has a terminal gap ('.') - represents missing data
    - Both sequences have indel gaps ('-') - no information to compare

    Args:
        seq1_aligned: First aligned sequence (with gaps)
        seq2_aligned: Second aligned sequence (with gaps)

    Returns:
        Tuple of (cleaned_seq1, cleaned_seq2) ready for scoring
    """
    cleaned_seq1 = []
    cleaned_seq2 = []

    for i in range(len(seq1_aligned)):
        # Skip if either has terminal gap
        if seq1_aligned[i] == '.' or seq2_aligned[i] == '.':
            continue
        # Skip if both have indel gaps
        if seq1_aligned[i] == '-' and seq2_aligned[i] == '-':
            continue

        cleaned_seq1.append(seq1_aligned[i])
        cleaned_seq2.append(seq2_aligned[i])

    return ''.join(cleaned_seq1), ''.join(cleaned_seq2)


def compute_msa_distance(seq1_aligned: str, seq2_aligned: str,
                         original_len1: Optional[int] = None,
                         original_len2: Optional[int] = None,
                         normalization_length: Optional[int] = None) -> float:
    """Compute distance between two sequences from an MSA.

    Uses standardized MycoBLAST-style adjustment parameters to score a
    pre-existing alignment.

    Distance is normalized using median sequence length from the MSA to ensure
    consistent scaling when comparing sequences of different lengths (e.g.,
    full ITS vs ITS2-only sequences).

    Args:
        seq1_aligned: First aligned sequence (with gaps: '-' and '.')
        seq2_aligned: Second aligned sequence (with gaps: '-' and '.')
        original_len1: Length of first original sequence (before alignment).
                      If not provided, will be calculated from aligned sequence.
        original_len2: Length of second original sequence (before alignment).
                      If not provided, will be calculated from aligned sequence.
        normalization_length: Length to use for distance normalization (typically
                             median sequence length in MSA). If not provided,
                             uses scored_positions from alignment (legacy behavior).

    Returns:
        Distance value (edits / normalization_length), or np.nan if alignment failed
        (insufficient overlap or no scoreable positions)
    """
    # Minimum overlap fraction required for valid alignment
    MIN_OVERLAP_FRACTION = 0.5

    try:
        from adjusted_identity import score_alignment, AdjustmentParams
    except ImportError:
        raise ImportError(
            "adjusted-identity package is required. "
            "Install it with: pip install git+https://github.com/joshuaowalker/adjusted-identity.git"
        )

    # Calculate original sequence lengths if not provided
    # Original length = count of non-gap characters (excluding both '.' and '-')
    if original_len1 is None:
        original_len1 = sum(1 for c in seq1_aligned if c not in ['.', '-'])
    if original_len2 is None:
        original_len2 = sum(1 for c in seq2_aligned if c not in ['.', '-'])

    # Handle empty sequences
    if original_len1 == 0 or original_len2 == 0:
        logging.debug("Empty sequence detected in MSA alignment")
        return np.nan

    # Filter positions unsuitable for scoring
    seq1_clean, seq2_clean = filter_msa_positions(seq1_aligned, seq2_aligned)
    overlap_len = len(seq1_clean)

    # Check minimum overlap - must be ≥50% of the shorter sequence
    min_required_overlap = min(original_len1, original_len2) * MIN_OVERLAP_FRACTION
    if overlap_len < min_required_overlap:
        logging.debug(
            f"Insufficient overlap: {overlap_len}bp < {min_required_overlap:.0f}bp "
            f"(original lengths: {original_len1}bp, {original_len2}bp)"
        )
        return np.nan

    if overlap_len == 0:
        # No positions to score (should be caught by min_required_overlap check, but defensive)
        return np.nan

    # MycoBLAST-style parameters: end_skip=0, all normalizations=True, max_repeat=1
    params = AdjustmentParams(
        end_skip_distance=0,
        normalize_homopolymers=True,
        handle_iupac_overlap=True,
        normalize_indels=True,
        max_repeat_motif_length=1
    )

    try:
        # Score the pre-aligned sequences
        result = score_alignment(seq1_clean, seq2_clean, adjustment_params=params)

        # Calculate distance using normalization length if provided
        if normalization_length is not None:
            # Normalize to median sequence length in MSA
            distance = result.mismatches / normalization_length
        else:
            # Legacy behavior: convert identity to distance
            distance = 1.0 - result.identity

    except Exception as e:
        logging.warning(f"Alignment scoring failed: {e}")
        return np.nan  # Alignment failed

    return distance


def calculate_distance_matrix(sequences: List[str],
                            show_progress: bool = True) -> np.ndarray:
    """
    Calculate pairwise distance matrix for sequences using MSA-based scoring.

    Uses SPOA to create a multiple sequence alignment, then scores all pairwise
    distances based on the shared alignment space. This provides more consistent
    alignment positions compared to independent pairwise alignments.

    Uses standardized MycoBLAST-style adjustment parameters for all alignments.

    Args:
        sequences: List of DNA sequences as strings
        show_progress: Whether to show progress bar for distance calculations

    Returns:
        Numpy array of pairwise distances (n x n)

    Raises:
        RuntimeError: If SPOA fails to create multiple sequence alignment
    """
    n = len(sequences)
    distance_matrix = np.zeros((n, n))

    # Use SPOA for MSA-based distance calculation
    logging.info("Creating MSA using SPOA for distance calculation...")
    aligned_sequences = run_spoa_msa(sequences)

    if aligned_sequences is None:
        raise RuntimeError(
            f"Failed to create multiple sequence alignment for {n} sequences using SPOA. "
            "This could be due to: extremely divergent sequences, SPOA subprocess error, "
            "or incomplete alignment output. Please check that SPOA is installed and sequences are valid."
        )

    # Successfully created MSA
    logging.info(f"SPOA alignment successful for {n} sequences")

    # Replace terminal gaps with '.'
    aligned_sequences = replace_terminal_gaps(aligned_sequences)

    # Trim alignment to region with sufficient coverage
    aligned_sequences = trim_alignment_by_coverage(aligned_sequences)

    # Compute median sequence length for distance normalization
    sequence_lengths = [len(seq) for seq in sequences]
    normalization_length = int(np.median(sequence_lengths))
    logging.info(
        f"MSA normalization length (median): {normalization_length}bp "
        f"(range: {min(sequence_lengths)}-{max(sequence_lengths)}bp)"
    )

    # Calculate total number of comparisons for progress bar
    total_comparisons = (n * (n - 1)) // 2

    # Create progress bar if requested
    pbar = None
    if show_progress and total_comparisons > 0:
        pbar = tqdm(total=total_comparisons,
                    desc="Calculating MSA-based distances",
                    unit=" comparisons")

    # Compute pairwise distances from MSA
    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_msa_distance(
                aligned_sequences[i],
                aligned_sequences[j],
                original_len1=len(sequences[i]),
                original_len2=len(sequences[j]),
                normalization_length=normalization_length
            )
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