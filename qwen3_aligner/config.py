"""
Configuration management for Qwen3 Aligner service.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Supported languages and their aliases
SUPPORTED_LANGUAGES = [
    "Chinese", "Cantonese", "English", "German", "Spanish",
    "French", "Italian", "Portuguese", "Russian", "Korean", "Japanese",
]

# Audio duration limits (seconds)
MAX_AUDIO_DURATION = 180  # 3 minutes, same as qwen_asr MAX_FORCE_ALIGN_INPUT_SECONDS
MIN_AUDIO_DURATION = 0.5

LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ko": "Korean",
    "ja": "Japanese",
}


def normalize_language(language: str) -> str:
    """Normalize language input to the canonical name.

    Accepts full names (case-insensitive) or short codes like 'zh', 'en'.
    """
    lang = language.strip()
    # Check aliases first
    if lang.lower() in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[lang.lower()]
    # Title-case normalization (same as qwen_asr)
    normalized = lang[:1].upper() + lang[1:].lower()
    if normalized in SUPPORTED_LANGUAGES:
        return normalized
    raise ValueError(
        f"Unsupported language: '{language}'. "
        f"Supported: {', '.join(SUPPORTED_LANGUAGES)}. "
        f"Aliases: {', '.join(LANGUAGE_ALIASES.keys())}"
    )


def check_audio_duration(audio_path: str) -> float:
    """Check audio duration and raise ValueError if out of range.

    Returns the duration in seconds.
    """
    import librosa

    duration = librosa.get_duration(filename=audio_path)
    if duration < MIN_AUDIO_DURATION:
        raise ValueError(
            f"Audio too short: {duration:.1f}s (minimum {MIN_AUDIO_DURATION}s)"
        )
    if duration > MAX_AUDIO_DURATION:
        raise ValueError(
            f"Audio too long: {duration:.1f}s (maximum {MAX_AUDIO_DURATION}s / {MAX_AUDIO_DURATION // 60} minutes)"
        )
    return duration


def get_app_dir() -> Path:
    """
    Get application directory.
    - For PyInstaller: directory containing the executable
    - For normal Python: package directory
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller executable
        return Path(sys.executable).parent.resolve()
    else:
        # Normal Python
        return Path(__file__).parent.resolve()


# Application directory
APP_DIR = get_app_dir()
# Default local model path (models/ next to executable or package)
DEFAULT_LOCAL_MODEL_PATH = APP_DIR / "models"


def get_default_model_path() -> str:
    """
    Get default model path.
    Priority:
    1. models/ directory next to executable (for PyInstaller)
    2. models/ directory in package (for development)
    3. HuggingFace model ID (fallback, will download)
    """
    # Check models/ next to app
    if DEFAULT_LOCAL_MODEL_PATH.exists() and (DEFAULT_LOCAL_MODEL_PATH / "config.json").exists():
        return str(DEFAULT_LOCAL_MODEL_PATH)

    # For PyInstaller, also check _internal/models if exists
    if getattr(sys, 'frozen', False):
        internal_models = APP_DIR / "_internal" / "models"
        if internal_models.exists() and (internal_models / "config.json").exists():
            return str(internal_models)

    # Fallback to HuggingFace
    return "Qwen/Qwen3-ForcedAligner-0.6B"


@dataclass
class ModelConfig:
    """Model configuration."""
    model_path: str = field(default_factory=get_default_model_path)
    device: str = "cpu"  # "cpu" or "cuda:0"
    dtype: str = "float32"  # "float32" or "bfloat16"


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8765
    workers: int = 1  # Number of uvicorn workers


@dataclass
class KeepAliveConfig:
    """Model keep-alive configuration (similar to Ollama)."""
    # How long to keep model in memory after last request (in seconds)
    # -1 = keep forever, 0 = unload immediately, >0 = seconds
    timeout: int = 300  # Default 5 minutes, like Ollama
    # Check interval for idle timeout
    check_interval: int = 30


@dataclass
class Config:
    """Main configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    keep_alive: KeepAliveConfig = field(default_factory=KeepAliveConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()

        # Model config
        if model_path := os.getenv("ALIGNER_MODEL_PATH"):
            config.model.model_path = model_path
        if device := os.getenv("ALIGNER_DEVICE"):
            config.model.device = device
        if dtype := os.getenv("ALIGNER_DTYPE"):
            config.model.dtype = dtype

        # Server config
        if host := os.getenv("ALIGNER_HOST"):
            config.server.host = host
        if port := os.getenv("ALIGNER_PORT"):
            config.server.port = int(port)
        if workers := os.getenv("ALIGNER_WORKERS"):
            config.server.workers = int(workers)

        # Keep-alive config
        if timeout := os.getenv("ALIGNER_KEEP_ALIVE"):
            config.keep_alive.timeout = int(timeout)

        return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set global config instance."""
    global _config
    _config = config
