"""
Pydantic schemas for request/response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class AlignmentItem(BaseModel):
    """Single alignment result item."""
    text: str = Field(..., description="Aligned text segment")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")


class AlignRequest(BaseModel):
    """Request for audio-text alignment."""
    audio: str = Field(
        ...,
        description="Audio source. Supports: local file path, HTTP/HTTPS URL, or base64 data URL (data:audio/wav;base64,...)"
    )
    text: str = Field(..., description="Text to align with audio")
    language: str = Field(default="Chinese", description="Language of the text (e.g., Chinese, English)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "audio": "/path/to/audio.wav",
                    "text": "你好世界",
                    "language": "Chinese"
                },
                {
                    "audio": "https://example.com/audio.wav",
                    "text": "Hello world",
                    "language": "English"
                },
                {
                    "audio": "data:audio/wav;base64,UklGRi...",
                    "text": "你好",
                    "language": "Chinese"
                }
            ]
        }
    }


class AlignResponse(BaseModel):
    """Response for alignment request."""
    success: bool = Field(..., description="Whether the request was successful")
    alignments: List[AlignmentItem] = Field(default_factory=list, description="Alignment results")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    processing_time: float = Field(..., description="Processing time in seconds")


class ModelStatus(BaseModel):
    """Model status information."""
    loaded: bool = Field(..., description="Whether the model is loaded")
    model_path: str = Field(..., description="Model path or ID")
    device: str = Field(..., description="Device (cpu/cuda)")
    last_used: Optional[float] = Field(default=None, description="Last used timestamp")
    keep_alive_timeout: int = Field(..., description="Keep-alive timeout in seconds")
    idle_seconds: Optional[float] = Field(default=None, description="Seconds since last use")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(default="0.1.0", description="Service version")


class UnloadResponse(BaseModel):
    """Response for model unload request."""
    success: bool = Field(..., description="Whether unload was successful")
    message: str = Field(..., description="Status message")
