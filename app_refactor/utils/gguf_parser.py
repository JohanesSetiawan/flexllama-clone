"""
GGUF Metadata Parser for Model Info and SWA Detection

This module provides a parser for reading metadata from GGUF model files
without requiring the official gguf library. Used for:
- Detecting if model uses Sliding Window Attention (SWA)
- Reading model architecture information
- Determining optimal parallel slots based on context window

GGUF Format:
    GGUF is the model file format for llama.cpp that stores:
    - Magic number: 'GGUF' (4 bytes)
    - Version (uint32)
    - Tensor count (uint64)
    - Metadata KV count (uint64)
    - Metadata key-value pairs
    - Tensor data

Components:
    - GGUFModelInfo: Dataclass with structured model info
    - GGUFReader: Minimal parser for GGUF header and metadata
    - detect_swa(): Quick check for SWA support
    - get_model_info(): Get full model info
    - get_optimal_parallel(): Calculate optimal parallel slots

SWA (Sliding Window Attention):
    Models with SWA (like Gemma, Mistral) use a limited attention window
    instead of full attention. This affects:
    - Context handling: Model only "sees" tokens within the window
    - Context shifting: Must be enabled for long conversations
    - Parallel slots: Need sufficient context per slot

Usage:
    # Quick SWA check
    is_swa, window_size = detect_swa("/path/to/model.gguf")
    
    # Full model info
    info = get_model_info("/path/to/model.gguf")
    print(f"Architecture: {info.architecture}")
    print(f"SWA: {info.is_swa}, Window: {info.swa_window_size}")
    
    # Optimal parallel calculation
    parallel, reason = get_optimal_parallel(
        model_path="/path/to/model.gguf",
        n_ctx=8192,
        default_parallel=4
    )
"""

import struct
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List


logger = logging.getLogger(__name__)


# GGUF Magic Number
GGUF_MAGIC = b"GGUF"

# GGUF Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


@dataclass
class GGUFModelInfo:
    """
    Container for model information from GGUF metadata.

    Attributes:
        architecture: Model architecture (e.g., "llama", "gemma")
        name: Model name from metadata
        context_length: Maximum context length
        embedding_length: Hidden size / embedding dimension
        block_count: Number of transformer blocks/layers
        is_swa: Whether model uses Sliding Window Attention
        swa_window_size: SWA window size (0 if not SWA)
        vocab_size: Vocabulary size
        raw_metadata: Full metadata dict for debugging
    """
    architecture: str = "unknown"
    name: str = "unknown"
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0
    is_swa: bool = False
    swa_window_size: int = 0
    vocab_size: int = 0
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


class GGUFReader:
    """
    Minimal GGUF reader for extracting header and metadata.

    This reader parses only the header and metadata sections of a GGUF file,
    without loading the tensor data. This is sufficient for model info extraction.

    Attributes:
        file_path: Path to the GGUF file
        metadata: Parsed metadata key-value pairs
        version: GGUF format version
        tensor_count: Number of tensors in the file
        metadata_kv_count: Number of metadata entries
    """

    def __init__(self, file_path: str):
        """
        Initialize the GGUF reader.

        Args:
            file_path: Path to the GGUF model file
        """
        self.file_path = Path(file_path)
        self.metadata: Dict[str, Any] = {}
        self.version: int = 0
        self.tensor_count: int = 0
        self.metadata_kv_count: int = 0

    def _read_string(self, f) -> str:
        """Read a length-prefixed string from the file."""
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read_value(self, f, value_type: int) -> Any:
        """
        Read a value of the specified type from the file.

        Args:
            f: File handle
            value_type: GGUF type identifier

        Returns:
            Parsed value of the appropriate Python type
        """
        type_readers = {
            GGUF_TYPE_UINT8: lambda: struct.unpack("<B", f.read(1))[0],
            GGUF_TYPE_INT8: lambda: struct.unpack("<b", f.read(1))[0],
            GGUF_TYPE_UINT16: lambda: struct.unpack("<H", f.read(2))[0],
            GGUF_TYPE_INT16: lambda: struct.unpack("<h", f.read(2))[0],
            GGUF_TYPE_UINT32: lambda: struct.unpack("<I", f.read(4))[0],
            GGUF_TYPE_INT32: lambda: struct.unpack("<i", f.read(4))[0],
            GGUF_TYPE_FLOAT32: lambda: struct.unpack("<f", f.read(4))[0],
            GGUF_TYPE_BOOL: lambda: struct.unpack("<B", f.read(1))[0] != 0,
            GGUF_TYPE_STRING: lambda: self._read_string(f),
            GGUF_TYPE_UINT64: lambda: struct.unpack("<Q", f.read(8))[0],
            GGUF_TYPE_INT64: lambda: struct.unpack("<q", f.read(8))[0],
            GGUF_TYPE_FLOAT64: lambda: struct.unpack("<d", f.read(8))[0],
        }

        if value_type == GGUF_TYPE_ARRAY:
            array_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            return [self._read_value(f, array_type) for _ in range(array_len)]

        reader = type_readers.get(value_type)
        if reader is None:
            raise ValueError(f"Unknown GGUF value type: {value_type}")
        return reader()

    def read_metadata(self) -> Dict[str, Any]:
        """
        Read all metadata from GGUF file.

        Returns:
            Dictionary of metadata key-value pairs

        Raises:
            FileNotFoundError: If GGUF file does not exist
            ValueError: If file is not valid GGUF format
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.file_path}")

        with open(self.file_path, "rb") as f:
            # Read and validate magic number
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(
                    f"Invalid GGUF file: magic={magic}, expected={GGUF_MAGIC}"
                )

            # Read version
            self.version = struct.unpack("<I", f.read(4))[0]
            if self.version < 2:
                raise ValueError(f"Unsupported GGUF version: {self.version}")

            # Read counts
            self.tensor_count = struct.unpack("<Q", f.read(8))[0]
            self.metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            # Read metadata key-value pairs
            for _ in range(self.metadata_kv_count):
                key = self._read_string(f)
                value_type = struct.unpack("<I", f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata[key] = value

        return self.metadata

    def get_model_info(self) -> GGUFModelInfo:
        """
        Parse metadata and return structured model info.

        Returns:
            GGUFModelInfo with parsed model information
        """
        if not self.metadata:
            self.read_metadata()

        info = GGUFModelInfo()
        info.raw_metadata = self.metadata.copy()

        # Get architecture
        info.architecture = self.metadata.get(
            "general.architecture", "unknown")
        info.name = self.metadata.get("general.name", "unknown")

        # Get context length (try multiple keys)
        arch = info.architecture
        info.context_length = (
            self.metadata.get(f"{arch}.context_length", 0)
            or self.metadata.get("llama.context_length", 0)
            or self.metadata.get("general.context_length", 0)
        )

        # Get embedding/hidden size
        info.embedding_length = (
            self.metadata.get(f"{arch}.embedding_length", 0)
            or self.metadata.get("llama.embedding_length", 0)
        )

        # Get block/layer count
        info.block_count = (
            self.metadata.get(f"{arch}.block_count", 0)
            or self.metadata.get("llama.block_count", 0)
        )

        # Detect SWA (Sliding Window Attention)
        info.is_swa, info.swa_window_size = self._detect_swa(arch)

        # Get vocab size
        info.vocab_size = self.metadata.get(f"{arch}.vocab_size", 0)

        return info

    def _detect_swa(self, arch: str) -> Tuple[bool, int]:
        """
        Detect if model uses Sliding Window Attention.

        Args:
            arch: Model architecture name

        Returns:
            Tuple of (is_swa, window_size)
        """
        # Different models use different key names for SWA
        swa_keys = [
            f"{arch}.attention.sliding_window",
            f"{arch}.attention.sliding_window_size",
            "llama.attention.sliding_window",
            "mistral.attention.sliding_window",
            "gemma.attention.sliding_window",
            "gemma2.attention.sliding_window",
            "gemma3.attention.sliding_window",
        ]

        for key in swa_keys:
            if key in self.metadata:
                swa_value = self.metadata[key]
                if isinstance(swa_value, (int, float)) and swa_value > 0:
                    return True, int(swa_value)

        return False, 0


def detect_swa(model_path: str) -> Tuple[bool, int]:
    """
    Quick function to detect if model uses SWA.

    Args:
        model_path: Path to GGUF model file

    Returns:
        Tuple of (is_swa, window_size)
        - is_swa: True if model uses Sliding Window Attention
        - window_size: Size of sliding window (0 if not SWA)
    """
    try:
        reader = GGUFReader(model_path)
        info = reader.get_model_info()
        return info.is_swa, info.swa_window_size
    except Exception as e:
        logger.warning(f"Failed to detect SWA for {model_path}: {e}")
        return False, 0


def get_model_info(model_path: str) -> Optional[GGUFModelInfo]:
    """
    Get full model info from GGUF file.

    Args:
        model_path: Path to GGUF model file

    Returns:
        GGUFModelInfo or None if parsing failed
    """
    try:
        reader = GGUFReader(model_path)
        return reader.get_model_info()
    except Exception as e:
        logger.warning(f"Failed to read GGUF metadata for {model_path}: {e}")
        return None


def get_optimal_parallel(
    model_path: str,
    n_ctx: int,
    default_parallel: int,
    min_ctx_per_slot: int = 2048
) -> Tuple[int, str]:
    """
    Determine optimal parallel setting based on model characteristics.

    For SWA models, parallel slots need more context to allow proper
    context shifting. This function analyzes the model and adjusts
    the parallel setting accordingly.

    Args:
        model_path: Path to GGUF model file
        n_ctx: Total context size
        default_parallel: Default parallel value from config
        min_ctx_per_slot: Minimum context tokens per slot

    Returns:
        Tuple of (optimal_parallel, reason)
    """
    try:
        reader = GGUFReader(model_path)
        info = reader.get_model_info()

        if info.is_swa:
            return _calculate_swa_parallel(
                info, n_ctx, default_parallel, min_ctx_per_slot
            )
        else:
            return _calculate_standard_parallel(
                n_ctx, default_parallel, min_ctx_per_slot
            )

    except Exception as e:
        logger.warning(
            f"Failed to analyze model {model_path}: {e}. "
            f"Using default parallel={default_parallel}"
        )
        return default_parallel, f"Analysis failed: {e}"


def _calculate_swa_parallel(
    info: GGUFModelInfo,
    n_ctx: int,
    default_parallel: int,
    min_ctx_per_slot: int
) -> Tuple[int, str]:
    """Calculate optimal parallel for SWA models."""
    swa_window = info.swa_window_size

    # Minimum context per slot should be at least 4x SWA window
    min_ctx_for_swa = max(min_ctx_per_slot, swa_window * 4)

    # Calculate optimal parallel
    optimal = max(1, n_ctx // min_ctx_for_swa)

    if optimal < default_parallel:
        reason = (
            f"SWA model detected (window={swa_window}). "
            f"Reduced parallel from {default_parallel} to {optimal} "
            f"(ctx/slot: {n_ctx // optimal} tokens)"
        )
        return optimal, reason
    else:
        reason = (
            f"SWA model (window={swa_window}), "
            f"parallel={default_parallel} is OK"
        )
        return default_parallel, reason


def _calculate_standard_parallel(
    n_ctx: int,
    default_parallel: int,
    min_ctx_per_slot: int
) -> Tuple[int, str]:
    """Calculate optimal parallel for non-SWA models."""
    effective_ctx = n_ctx // default_parallel

    if effective_ctx < min_ctx_per_slot and default_parallel > 1:
        optimal = max(1, n_ctx // min_ctx_per_slot)
        reason = (
            f"Non-SWA model, but context per slot too small. "
            f"Adjusted parallel from {default_parallel} to {optimal}"
        )
        return optimal, reason
    else:
        reason = f"Non-SWA model, parallel={default_parallel} is OK"
        return default_parallel, reason
