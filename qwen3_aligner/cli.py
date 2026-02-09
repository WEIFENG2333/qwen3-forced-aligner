"""
CLI interface for Qwen3 Forced Aligner.

Commands:
- align: Perform audio-text alignment directly
- serve: Start the alignment server
- download-model: Download model files from HuggingFace
"""

import argparse
import json
import sys
import time


def format_output(alignments: list, output_format: str) -> str:
    """Format alignment results."""
    if output_format == "json":
        items = []
        for item in alignments:
            if hasattr(item, "model_dump"):
                items.append(item.model_dump())
            elif isinstance(item, dict):
                items.append(item)
            else:
                items.append({
                    "text": item.text,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                })
        return json.dumps(items, ensure_ascii=False, indent=2)
    else:
        lines = []
        for item in alignments:
            if isinstance(item, dict):
                text = item["text"]
                start = item["start_time"]
                end = item["end_time"]
            else:
                text = item.text
                start = item.start_time
                end = item.end_time
            lines.append(f"[{start:.2f}s - {end:.2f}s] {text}")
        return "\n".join(lines)


def cmd_align(args):
    """Handle align command."""
    from .config import Config, check_audio_duration, normalize_language
    from .model_manager import get_model_manager

    # Normalize language input
    try:
        language = normalize_language(args.language)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Check audio duration for local files
    audio = args.audio
    if not audio.startswith(("http://", "https://", "data:")):
        try:
            duration = check_audio_duration(audio)
            print(f"Audio duration: {duration:.1f}s", file=sys.stderr)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    config = Config()
    if args.model:
        config.model.model_path = args.model
    if args.dtype:
        config.model.dtype = args.dtype

    manager = get_model_manager(config)

    print(f"Model: {config.model.model_path}", file=sys.stderr)
    print(f"Audio: {audio[:80]}..." if len(audio) > 80 else f"Audio: {audio}", file=sys.stderr)
    start_time = time.time()

    alignments = manager.align_sync(args.audio, args.text, language)

    processing_time = time.time() - start_time
    print(f"Alignment completed in {processing_time:.2f}s", file=sys.stderr)

    result = format_output(alignments, args.format)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Results written to: {args.output}", file=sys.stderr)
    else:
        print(result)


def cmd_serve(args):
    """Handle serve command."""
    from .config import Config
    from .server import run_server

    config = Config()
    if args.model:
        config.model.model_path = args.model
    if args.dtype:
        config.model.dtype = args.dtype
    config.keep_alive.timeout = args.keep_alive

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model: {config.model.model_path}")
    print(f"Keep-alive: {args.keep_alive}s (-1=forever, 0=immediate unload)")

    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        config=config,
        log_level=args.log_level,
    )


def cmd_download_model(args):
    """Handle download-model command."""
    from pathlib import Path

    from .config import APP_DIR

    model_id = args.model_id or "Qwen/Qwen3-ForcedAligner-0.6B"

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default to models/ next to the executable/package
        output_dir = APP_DIR / "models"

    output_dir = output_dir.resolve()

    print(f"Downloading model: {model_id}")
    print(f"Output directory: {output_dir}")

    # Check if already exists
    if (output_dir / "config.json").exists() and not args.force:
        print(f"\nModel already exists at {output_dir}")
        print("Use --force to re-download")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        print("\nDownloading from HuggingFace Hub...")
        print("This may take a while (~1.8GB)...\n")

        # Download model files
        snapshot_download(
            repo_id=model_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print(f"\nModel downloaded successfully to: {output_dir}")
        print("\nYou can now use the aligner:")
        print("  qwen3-aligner align -a audio.wav -t 'text' -l Chinese")

    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        sys.exit(1)


def cmd_model_info(args):
    """Handle model-info command."""
    from pathlib import Path

    from .config import APP_DIR, get_default_model_path

    default_path = get_default_model_path()

    print("Model Information")
    print("=" * 50)
    print(f"Application directory: {APP_DIR}")
    print(f"Default model path: {default_path}")

    # Check if local model exists
    model_path = Path(default_path)
    if model_path.exists() and (model_path / "config.json").exists():
        print("Model status: Found locally")

        # Calculate size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        size_gb = total_size / (1024 * 1024 * 1024)
        print(f"Model size: {size_mb:.1f} MB ({size_gb:.2f} GB)")

        # List files
        print("\nModel files:")
        for f in sorted(model_path.iterdir()):
            if f.is_file():
                fsize = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name}: {fsize:.1f} MB")
    else:
        print("Model status: Not found locally")
        print("\nTo download the model, run:")
        print("  qwen3-aligner download-model")


def main():
    """Main entry point for CLI."""
    from .config import get_default_model_path

    default_model = get_default_model_path()

    parser = argparse.ArgumentParser(
        prog="qwen3-aligner",
        description="Qwen3 Forced Aligner - Audio-Text Alignment Tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # align command
    align_parser = subparsers.add_parser(
        "align",
        help="Perform audio-text alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Audio input formats:
  - Local file path: /path/to/audio.wav
  - URL: https://example.com/audio.wav
  - Base64 data URL: data:audio/wav;base64,UklGRi...

Examples:
  %(prog)s -a audio.wav -t "你好世界" -l Chinese
  %(prog)s -a https://example.com/audio.wav -t "Hello" -l English -f json
"""
    )
    align_parser.add_argument(
        "--audio", "-a",
        required=True,
        help="Audio source: file path, URL, or base64 data URL"
    )
    align_parser.add_argument(
        "--text", "-t",
        required=True,
        help="Text to align with audio"
    )
    align_parser.add_argument(
        "--language", "-l",
        default="Chinese",
        help="Language: Chinese/zh, English/en, Japanese/ja, Korean/ko, "
             "Cantonese/yue, German/de, Spanish/es, French/fr, Italian/it, "
             "Portuguese/pt, Russian/ru (default: Chinese)"
    )
    align_parser.add_argument(
        "--model", "-m",
        default=None,
        help=f"Model path (default: {default_model})"
    )
    align_parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default=None,
        help="Model dtype (default: float32, use bfloat16 for CUDA)"
    )
    align_parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    align_parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)"
    )
    align_parser.set_defaults(func=cmd_align)

    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the alignment server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
API Endpoints:
  POST /align          - Perform alignment
  GET  /health         - Health check
  GET  /model/status   - Model status
  POST /model/load     - Pre-load model
  POST /model/unload   - Unload model

Example:
  %(prog)s --port 8765 --keep-alive 300
  curl -X POST http://localhost:8765/align \\
    -H "Content-Type: application/json" \\
    -d '{"audio": "path/to/audio.wav", "text": "你好", "language": "Chinese"}'
"""
    )
    serve_parser.add_argument(
        "--host", "-H",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8765,
        help="Port to bind (default: 8765)"
    )
    serve_parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of workers (default: 1)"
    )
    serve_parser.add_argument(
        "--model", "-m",
        default=None,
        help=f"Model path (default: {default_model})"
    )
    serve_parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default=None,
        help="Model dtype (default: float32, use bfloat16 for CUDA)"
    )
    serve_parser.add_argument(
        "--keep-alive", "-k",
        type=int,
        default=300,
        help="Keep model in memory for N seconds after last use (-1=forever, 0=immediate unload, default: 300)"
    )
    serve_parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    serve_parser.set_defaults(func=cmd_serve)

    # download-model command
    download_parser = subparsers.add_parser(
        "download-model",
        help="Download model files from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Download to default location
  %(prog)s -o ./my-models            # Download to custom directory
  %(prog)s --model-id Qwen/Qwen3-ForcedAligner-0.6B  # Specify model ID
"""
    )
    download_parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="HuggingFace model ID (default: Qwen/Qwen3-ForcedAligner-0.6B)"
    )
    download_parser.add_argument(
        "--output", "-o",
        help="Output directory (default: models/ next to executable)"
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    download_parser.set_defaults(func=cmd_download_model)

    # model-info command
    info_parser = subparsers.add_parser(
        "model-info",
        help="Show model information and status"
    )
    info_parser.set_defaults(func=cmd_model_info)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
