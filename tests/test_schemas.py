"""Tests for Pydantic schemas."""

import pytest

from qwen3_aligner.schemas import (
    AlignmentItem,
    AlignRequest,
    AlignResponse,
    HealthResponse,
    ModelStatus,
    UnloadResponse,
)


class TestAlignmentItem:

    def test_basic(self):
        item = AlignmentItem(text="hello", start_time=0.0, end_time=1.0)
        assert item.text == "hello"
        assert item.start_time == 0.0
        assert item.end_time == 1.0

    def test_model_dump(self):
        item = AlignmentItem(text="world", start_time=1.5, end_time=2.3)
        d = item.model_dump()
        assert d == {"text": "world", "start_time": 1.5, "end_time": 2.3}


class TestAlignRequest:

    def test_required_fields(self):
        req = AlignRequest(audio="/path/to/audio.wav", text="hello")
        assert req.audio == "/path/to/audio.wav"
        assert req.text == "hello"
        assert req.language == "Chinese"  # default

    def test_custom_language(self):
        req = AlignRequest(audio="audio.wav", text="hello", language="English")
        assert req.language == "English"

    def test_missing_audio(self):
        with pytest.raises(Exception):
            AlignRequest(text="hello")

    def test_missing_text(self):
        with pytest.raises(Exception):
            AlignRequest(audio="audio.wav")


class TestAlignResponse:

    def test_success(self):
        items = [AlignmentItem(text="hi", start_time=0.0, end_time=0.5)]
        resp = AlignResponse(success=True, alignments=items, processing_time=0.1)
        assert resp.success is True
        assert len(resp.alignments) == 1
        assert resp.error is None

    def test_failure(self):
        resp = AlignResponse(success=False, error="model not found", processing_time=0.0)
        assert resp.success is False
        assert resp.error == "model not found"
        assert resp.alignments == []


class TestModelStatus:

    def test_basic(self):
        status = ModelStatus(
            loaded=True,
            model_path="/models",
            device="cpu",
            keep_alive_timeout=300,
        )
        assert status.loaded is True
        assert status.last_used is None
        assert status.idle_seconds is None


class TestHealthResponse:

    def test_basic(self):
        resp = HealthResponse(status="ok", model_loaded=False)
        assert resp.status == "ok"
        assert resp.version == "0.1.0"


class TestUnloadResponse:

    def test_basic(self):
        resp = UnloadResponse(success=True, message="done")
        assert resp.success is True
