# Qwen3 Forced Aligner

[![Build and Release](https://github.com/YOUR_USERNAME/qwen3-forced-aligner/actions/workflows/build-release.yml/badge.svg)](https://github.com/YOUR_USERNAME/qwen3-forced-aligner/actions/workflows/build-release.yml)
[![Test](https://github.com/YOUR_USERNAME/qwen3-forced-aligner/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR_USERNAME/qwen3-forced-aligner/actions/workflows/test.yml)

Audio-text forced alignment service based on [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) model.

## Features

- **CLI Mode**: Command-line audio-text alignment
- **Server Mode**: REST API service with model caching
- **Multi-language Support**: Chinese, English, Japanese, Korean, and 7 more languages
- **Multiple Audio Inputs**: Local files, URLs, Base64
- **Cross-platform**: Linux, macOS, Windows binaries available

## Quick Start

### Option 1: Download Pre-built Binary

1. Download the latest release from [Releases](https://github.com/YOUR_USERNAME/qwen3-forced-aligner/releases)
2. Extract the archive
3. Download the model:
   ```bash
   ./qwen3-aligner download-model
   ```
4. Run alignment:
   ```bash
   ./qwen3-aligner align -a audio.wav -t "Hello world" -l English
   ```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qwen3-forced-aligner.git
cd qwen3-forced-aligner

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Download model
qwen3-aligner download-model

# Run alignment
qwen3-aligner align -a audio.wav -t "Hello world" -l English
```

## Usage

### CLI Mode

```bash
# Basic usage
qwen3-aligner align -a audio.wav -t "Text content" -l Chinese

# Specify output format
qwen3-aligner align -a audio.wav -t "Hello world" -l English -f json

# Use custom model path
qwen3-aligner align -a audio.wav -t "Text" -l Chinese -m ./models
```

### Server Mode

```bash
# Start server
qwen3-aligner serve -p 8765

# Run in background
qwen3-aligner serve -p 8765 &

# Set model auto-unload timeout (seconds)
qwen3-aligner serve -p 8765 -k 300
```

### REST API

```bash
# Health check
curl http://localhost:8765/health

# Alignment request
curl -X POST http://localhost:8765/align \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "/path/to/audio.wav",
    "text": "Text content",
    "language": "Chinese"
  }'

# Using Base64 audio
curl -X POST http://localhost:8765/align \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "data:audio/wav;base64,UklGR...",
    "text": "Text content",
    "language": "Chinese"
  }'

# Model status
curl http://localhost:8765/model/status

# Manually unload model
curl -X POST http://localhost:8765/model/unload
```

### Model Management

```bash
# Download model
qwen3-aligner download-model

# Download to custom location
qwen3-aligner download-model -o ./my-models

# Show model information
qwen3-aligner model-info
```

## API Response Format

```json
{
  "success": true,
  "alignments": [
    {"text": "Hello", "start_time": 0.24, "end_time": 0.64},
    {"text": "world", "start_time": 0.64, "end_time": 0.96}
  ],
  "processing_time": 0.25
}
```

## Supported Languages

| Language | Code |
|----------|------|
| Chinese | Chinese |
| Cantonese | Cantonese |
| English | English |
| German | German |
| Spanish | Spanish |
| French | French |
| Italian | Italian |
| Portuguese | Portuguese |
| Russian | Russian |
| Korean | Korean |
| Japanese | Japanese |

## Building from Source

### Prerequisites

- Python 3.9+
- PyInstaller
- PyTorch

### Build

```bash
# Install build dependencies
pip install -e ".[dev]"

# Build (without model)
python build.py

# Build with model included
python build.py --with-model

# Build and create archive
python build.py --archive
```

### Output

Built files will be in `dist/qwen3-aligner/`:
- `qwen3-aligner` (or `qwen3-aligner.exe` on Windows)
- `models/` directory (copy model files here)

## Project Structure

```
qwen3-forced-aligner/
├── qwen3_aligner/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py           # CLI entry point
│   ├── server.py        # FastAPI server
│   ├── model_manager.py # Model management (singleton, auto-unload)
│   ├── config.py        # Configuration
│   └── schemas.py       # Pydantic models
├── qwen_asr/            # Core ASR/alignment library
├── build.py             # PyInstaller build script
├── pyproject.toml       # Project configuration
└── README.md
```

## Performance

| Scenario | Time |
|----------|------|
| Model loading (first time) | ~5-7s |
| Inference (model loaded) | ~0.2-0.5s |
| Server startup | ~1-2s |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ALIGNER_MODEL_PATH` | Model path | Auto-detect |
| `ALIGNER_DEVICE` | Device (cpu/cuda) | cpu |
| `ALIGNER_DTYPE` | Data type (float32/bfloat16) | float32 |
| `ALIGNER_HOST` | Server host | 0.0.0.0 |
| `ALIGNER_PORT` | Server port | 8765 |
| `ALIGNER_KEEP_ALIVE` | Keep-alive timeout | 300 |

## License

Apache 2.0

## Acknowledgments

- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) by Alibaba Qwen Team
- [qwen_asr](https://github.com/QwenLM/Qwen3-ASR) - Core ASR library
