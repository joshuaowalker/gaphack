"""Abstract interface for sequence neighborhood discovery."""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path


class NeighborhoodFinder(ABC):
    """Abstract interface for sequence neighborhood discovery.

    Implementations provide different search methods (BLAST, vsearch, etc.)
    for finding similar sequences in a database.
    """

    def __init__(self, sequences: List[str], headers: List[str],
                 output_dir: Optional[Path] = None):
        """Initialize finder with sequence database.

        Args:
            sequences: List of DNA sequences as strings
            headers: List of sequence headers/identifiers
            output_dir: Output directory for database files and cache

        Raises:
            ValueError: If sequences and headers have different lengths
        """
        if len(sequences) != len(headers):
            raise ValueError("Number of sequences and headers must match")

        self.sequences = sequences
        self.headers = headers
        self.output_dir = output_dir

    @abstractmethod
    def find_neighborhood(self, target_headers: List[str],
                         max_hits: int = 1000,
                         e_value_threshold: float = 1e-5,
                         min_identity: Optional[float] = None) -> List[str]:
        """Find neighborhood sequences for given target headers.

        Args:
            target_headers: List of sequence headers to find neighborhoods for
            max_hits: Maximum number of hits to return per target
            e_value_threshold: E-value threshold (BLAST) or identity threshold (vsearch)
            min_identity: Minimum percent identity (auto-calculated if None)

        Returns:
            List of sequence headers in the neighborhood (including targets)
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up temporary files and databases."""
        pass
