"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from qwen3_aligner.config import (
    LANGUAGE_ALIASES,
    SUPPORTED_LANGUAGES,
    Config,
    KeepAliveConfig,
    ModelConfig,
    ServerConfig,
    get_default_model_path,
    normalize_language,
)


class TestLanguageNormalization:
    """Test language normalization and alias resolution."""

    def test_full_name_exact(self):
        assert normalize_language("Chinese") == "Chinese"
        assert normalize_language("English") == "English"
        assert normalize_language("Japanese") == "Japanese"

    def test_full_name_case_insensitive(self):
        assert normalize_language("chinese") == "Chinese"
        assert normalize_language("ENGLISH") == "English"
        assert normalize_language("japanese") == "Japanese"
        assert normalize_language("cHiNeSe") == "Chinese"

    def test_aliases(self):
        assert normalize_language("zh") == "Chinese"
        assert normalize_language("en") == "English"
        assert normalize_language("ja") == "Japanese"
        assert normalize_language("ko") == "Korean"
        assert normalize_language("de") == "German"
        assert normalize_language("fr") == "French"
        assert normalize_language("es") == "Spanish"
        assert normalize_language("it") == "Italian"
        assert normalize_language("pt") == "Portuguese"
        assert normalize_language("ru") == "Russian"
        assert normalize_language("yue") == "Cantonese"

    def test_aliases_case_insensitive(self):
        assert normalize_language("ZH") == "Chinese"
        assert normalize_language("EN") == "English"
        assert normalize_language("Ja") == "Japanese"

    def test_whitespace_stripped(self):
        assert normalize_language("  Chinese  ") == "Chinese"
        assert normalize_language(" zh ") == "Chinese"

    def test_unsupported_language(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            normalize_language("Klingon")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            normalize_language("")

    def test_all_supported_languages(self):
        for lang in SUPPORTED_LANGUAGES:
            assert normalize_language(lang) == lang

    def test_all_aliases_mapped(self):
        for alias, expected in LANGUAGE_ALIASES.items():
            assert normalize_language(alias) == expected
            assert expected in SUPPORTED_LANGUAGES


class TestConfig:
    """Test configuration classes."""

    def test_default_config(self):
        config = Config()
        assert config.model.device == "cpu"
        assert config.model.dtype == "float32"
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8765
        assert config.server.workers == 1
        assert config.keep_alive.timeout == 300
        assert config.keep_alive.check_interval == 30

    def test_model_config_defaults(self):
        mc = ModelConfig()
        assert mc.device == "cpu"
        assert mc.dtype == "float32"

    def test_server_config_defaults(self):
        sc = ServerConfig()
        assert sc.host == "0.0.0.0"
        assert sc.port == 8765

    def test_keep_alive_config_defaults(self):
        kc = KeepAliveConfig()
        assert kc.timeout == 300
        assert kc.check_interval == 30

    def test_from_env(self):
        env = {
            "ALIGNER_MODEL_PATH": "/tmp/test-model",
            "ALIGNER_DEVICE": "cuda:0",
            "ALIGNER_DTYPE": "bfloat16",
            "ALIGNER_HOST": "127.0.0.1",
            "ALIGNER_PORT": "9999",
            "ALIGNER_WORKERS": "4",
            "ALIGNER_KEEP_ALIVE": "600",
        }
        with patch.dict(os.environ, env, clear=False):
            config = Config.from_env()
            assert config.model.model_path == "/tmp/test-model"
            assert config.model.device == "cuda:0"
            assert config.model.dtype == "bfloat16"
            assert config.server.host == "127.0.0.1"
            assert config.server.port == 9999
            assert config.server.workers == 4
            assert config.keep_alive.timeout == 600

    def test_from_env_partial(self):
        env = {"ALIGNER_PORT": "1234"}
        with patch.dict(os.environ, env, clear=False):
            config = Config.from_env()
            assert config.server.port == 1234
            assert config.server.host == "0.0.0.0"  # default


class TestGetDefaultModelPath:
    """Test model path resolution."""

    def test_returns_string(self):
        result = get_default_model_path()
        assert isinstance(result, str)

    def test_fallback_to_huggingface(self):
        # When no local model exists, should return HuggingFace ID
        result = get_default_model_path()
        # Either a local path or HuggingFace ID
        assert result == "Qwen/Qwen3-ForcedAligner-0.6B" or os.path.isabs(result)
