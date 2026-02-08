"""
FastAPI server for Qwen3 Forced Aligner.

Features:
- REST API for audio-text alignment
- Model status and management endpoints
- Auto-unload after idle timeout
- Health check endpoint
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import Config, get_config, normalize_language, set_config
from .model_manager import get_model_manager
from .schemas import (
    AlignRequest,
    AlignResponse,
    HealthResponse,
    ModelStatus,
    UnloadResponse,
)

logger = logging.getLogger(__name__)

# Global reference to auto-unload task
_auto_unload_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.

    Handles startup and shutdown events.
    """
    global _auto_unload_task

    logger.info("Starting Qwen3 Aligner Server...")

    # Get model manager (will be created if not exists)
    manager = get_model_manager()

    # Start auto-unload checker as background task
    _auto_unload_task = asyncio.create_task(
        manager.start_auto_unload_checker()
    )

    logger.info("Server started successfully")

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down server...")

    # Cancel auto-unload task
    if _auto_unload_task:
        _auto_unload_task.cancel()
        try:
            await _auto_unload_task
        except asyncio.CancelledError:
            pass

    # Shutdown model manager
    manager.shutdown()

    logger.info("Server shutdown complete")


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config:
        set_config(config)

    app = FastAPI(
        title="Qwen3 Forced Aligner API",
        description="Audio-Text Forced Alignment Service using Qwen3-ForcedAligner",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register API routes."""

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Health check endpoint."""
        manager = get_model_manager()
        return HealthResponse(
            status="ok",
            model_loaded=manager.is_loaded,
            version="0.1.0"
        )

    @app.get("/model/status", response_model=ModelStatus, tags=["Model"])
    async def get_model_status():
        """Get current model status."""
        manager = get_model_manager()
        status = manager.get_status()
        return ModelStatus(**status)

    @app.post("/model/load", response_model=ModelStatus, tags=["Model"])
    async def load_model():
        """
        Manually load the model into memory.

        The model will be loaded automatically on first alignment request,
        but you can use this endpoint to pre-load it.
        """
        manager = get_model_manager()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, manager.load)

        status = manager.get_status()
        return ModelStatus(**status)

    @app.post("/model/unload", response_model=UnloadResponse, tags=["Model"])
    async def unload_model():
        """
        Manually unload the model from memory.

        This will free up memory/VRAM. The model will be reloaded
        on the next alignment request.
        """
        manager = get_model_manager()
        success = manager.unload()

        if success:
            return UnloadResponse(
                success=True,
                message="Model unloaded successfully"
            )
        else:
            return UnloadResponse(
                success=True,
                message="Model was already unloaded"
            )

    @app.post("/align", response_model=AlignResponse, tags=["Alignment"])
    async def align_audio_text(request: AlignRequest):
        """
        Perform forced alignment between audio and text.

        The model will be loaded automatically if not already loaded.
        After alignment, the model stays in memory according to keep_alive settings.
        """
        manager = get_model_manager()
        start_time = time.time()

        try:
            language = normalize_language(request.language)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            alignments = await manager.align(
                audio=request.audio,
                text=request.text,
                language=language,
            )

            processing_time = time.time() - start_time

            # Check if immediate unload is configured
            config = get_config()
            if config.keep_alive.timeout == 0:
                manager.unload()

            return AlignResponse(
                success=True,
                alignments=alignments,
                processing_time=round(processing_time, 3)
            )

        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"Audio file not found: {e}")
        except Exception as e:
            logger.exception("Alignment failed")
            return AlignResponse(
                success=False,
                error=str(e),
                processing_time=round(time.time() - start_time, 3)
            )

    @app.get("/config", tags=["System"])
    async def get_current_config():
        """Get current server configuration."""
        config = get_config()
        return {
            "model": {
                "model_path": config.model.model_path,
                "device": config.model.device,
                "dtype": config.model.dtype,
            },
            "server": {
                "host": config.server.host,
                "port": config.server.port,
                "workers": config.server.workers,
            },
            "keep_alive": {
                "timeout": config.keep_alive.timeout,
                "check_interval": config.keep_alive.check_interval,
            }
        }


# Create default app instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    workers: int = 1,
    config: Optional[Config] = None,
    log_level: str = "info"
) -> None:
    """Run the server using uvicorn."""
    import uvicorn

    if config:
        set_config(config)
        config.server.host = host
        config.server.port = port
        config.server.workers = workers

    uvicorn.run(
        "qwen3_aligner.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,
    )
