"""
Qwen3 Forced Aligner - Audio-Text Alignment Service

A production-ready service for audio-text forced alignment using Qwen3-ForcedAligner.

Features:
- CLI and Server modes
- Lazy model loading
- Auto-unload after idle timeout (like Ollama)
- Thread-safe concurrent access
- REST API with FastAPI
"""

__version__ = "0.1.0"

from .config import Config, get_config, set_config
from .model_manager import ModelManager, get_model_manager
from .schemas import AlignmentItem, AlignRequest, AlignResponse

__all__ = [
    "Config",
    "get_config",
    "set_config",
    "ModelManager",
    "get_model_manager",
    "AlignmentItem",
    "AlignRequest",
    "AlignResponse",
]
