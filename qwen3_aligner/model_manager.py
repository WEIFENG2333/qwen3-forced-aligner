"""
Thread-safe model manager with lazy loading and auto-unload.

Similar to Ollama's model management:
- Lazy loading: Model is loaded on first request
- Keep-alive: Model stays in memory for a configurable time after last use
- Auto-unload: Model is automatically unloaded after idle timeout
- Thread-safe: Safe for concurrent access
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

import torch

from .config import Config, get_config
from .schemas import AlignmentItem

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton model manager with lazy loading and auto-unload.

    Thread-safe implementation using double-checked locking pattern.
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[Config] = None) -> "ModelManager":
        """Thread-safe singleton pattern with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, config: Optional[Config] = None):
        """Initialize the model manager."""
        # Avoid re-initialization
        if self._initialized:
            return

        self._config = config or get_config()
        self._model: Optional[Any] = None
        self._model_lock = threading.Lock()
        self._last_used: Optional[float] = None
        self._unload_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aligner")
        self._initialized = True

        logger.info(f"ModelManager initialized with keep_alive={self._config.keep_alive.timeout}s")

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None

    @property
    def last_used(self) -> Optional[float]:
        """Get timestamp of last model use."""
        return self._last_used

    @property
    def idle_seconds(self) -> Optional[float]:
        """Get seconds since last use."""
        if self._last_used is None:
            return None
        return time.time() - self._last_used

    def _load_model(self) -> None:
        """Load the model (internal, not thread-safe)."""
        from qwen_asr import Qwen3ForcedAligner

        device = self._config.model.device
        dtype_str = self._config.model.dtype

        # Determine dtype
        if dtype_str == "bfloat16" and device.startswith("cuda"):
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Determine device_map
        device_map = device if device.startswith("cuda") else None

        logger.info(f"Loading model: {self._config.model.model_path}")
        logger.info(f"Device: {device}, Dtype: {dtype}")

        start_time = time.time()
        self._model = Qwen3ForcedAligner.from_pretrained(
            self._config.model.model_path,
            dtype=dtype,
            device_map=device_map,
        )
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")

    def load(self) -> None:
        """Load the model (thread-safe, lazy loading)."""
        if self._model is not None:
            return

        with self._model_lock:
            # Double-check after acquiring lock
            if self._model is None:
                self._load_model()
                self._last_used = time.time()

    def unload(self) -> bool:
        """Unload the model from memory."""
        with self._model_lock:
            if self._model is None:
                logger.info("Model already unloaded")
                return False

            logger.info("Unloading model...")
            del self._model
            self._model = None
            self._last_used = None

            # Force garbage collection
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded successfully")
            return True

    def _align_sync(
        self,
        audio: str,
        text: str,
        language: str = "Chinese"
    ) -> List[AlignmentItem]:
        """Perform alignment synchronously (internal)."""
        # Ensure model is loaded
        self.load()

        # Update last used time
        self._last_used = time.time()

        # Perform alignment
        results = self._model.align(
            audio=audio,
            text=text,
            language=language,
        )

        # Convert to AlignmentItem list
        items = []
        for r in results:
            for item in r:
                items.append(AlignmentItem(
                    text=item.text,
                    start_time=round(item.start_time, 3),
                    end_time=round(item.end_time, 3),
                ))

        return items

    async def align(
        self,
        audio: str,
        text: str,
        language: str = "Chinese"
    ) -> List[AlignmentItem]:
        """
        Perform alignment asynchronously.

        Uses ThreadPoolExecutor to avoid blocking the event loop,
        as model inference is CPU-intensive.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._align_sync,
            audio,
            text,
            language
        )

    def align_sync(
        self,
        audio: str,
        text: str,
        language: str = "Chinese"
    ) -> List[AlignmentItem]:
        """Perform alignment synchronously (public API for CLI)."""
        return self._align_sync(audio, text, language)

    async def start_auto_unload_checker(self) -> None:
        """Start background task to check for idle timeout and auto-unload."""
        timeout = self._config.keep_alive.timeout

        # -1 means keep forever
        if timeout < 0:
            logger.info("Auto-unload disabled (keep_alive=-1)")
            return

        # 0 means unload immediately after each request (handled elsewhere)
        if timeout == 0:
            logger.info("Immediate unload mode (keep_alive=0)")
            return

        logger.info(f"Starting auto-unload checker (timeout={timeout}s)")

        while True:
            await asyncio.sleep(self._config.keep_alive.check_interval)

            if not self.is_loaded:
                continue

            idle = self.idle_seconds
            if idle is not None and idle >= timeout:
                logger.info(f"Model idle for {idle:.0f}s, auto-unloading...")
                self.unload()

    def get_status(self) -> dict:
        """Get current model status."""
        return {
            "loaded": self.is_loaded,
            "model_path": self._config.model.model_path,
            "device": self._config.model.device,
            "last_used": self._last_used,
            "keep_alive_timeout": self._config.keep_alive.timeout,
            "idle_seconds": self.idle_seconds,
        }

    def shutdown(self) -> None:
        """Shutdown the model manager."""
        logger.info("Shutting down ModelManager...")
        self.unload()
        self._executor.shutdown(wait=True)
        logger.info("ModelManager shutdown complete")


# Convenience functions
def get_model_manager(config: Optional[Config] = None) -> ModelManager:
    """Get the singleton model manager instance."""
    return ModelManager(config)
