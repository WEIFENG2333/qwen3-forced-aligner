#!/usr/bin/env python3
"""
Build script for creating PyInstaller package.

Usage:
    python build.py [--with-model] [--cpu-only]

Options:
    --with-model    Include the model files in the package (adds ~1.8GB)
    --cpu-only      Use CPU-only PyTorch (smaller package size)
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.resolve()


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("ERROR: PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("ERROR: PyTorch not installed.")
        sys.exit(1)


def create_spec_file(project_root: Path, with_model: bool = False):
    """Create PyInstaller spec file with relative paths."""
    # Use os.path.join for cross-platform compatibility
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Get project root (directory containing this spec file)
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))

# Collect hidden imports
hiddenimports = [
    'torch',
    'transformers',
    'transformers.models.auto',
    'transformers.models.auto.modeling_auto',
    'transformers.models.auto.configuration_auto',
    'transformers.models.auto.processing_auto',
    'librosa',
    'soundfile',
    'numpy',
    'fastapi',
    'uvicorn',
    'pydantic',
    'click',
    'nagisa',
    'nagisa.mecab',
    'nagisa.tagger',
    'soynlp',
    'soynlp.tokenizer',
    'six',
    'six.moves',
]

# Add nagisa submodules
try:
    hiddenimports.extend(collect_submodules('nagisa'))
except:
    pass

# Collect data files
datas = []

# Collect qwen_asr data (from installed package)
try:
    datas.extend(collect_data_files('qwen_asr'))
except:
    pass

# Collect transformers data
try:
    datas.extend(collect_data_files('transformers'))
except:
    pass

# Collect nagisa data
try:
    datas.extend(collect_data_files('nagisa'))
except:
    pass

a = Analysis(
    [os.path.join(SPEC_DIR, 'main.py')],
    pathex=[SPEC_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'scipy',
        'sklearn',
        'numba',
        'PIL',
        'cv2',
        'pandas',
        'sympy',
        'jedi',
        'parso',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='qwen3-aligner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='qwen3-aligner',
)
'''
    spec_path = project_root / "qwen3_aligner.spec"
    with open(spec_path, "w") as f:
        f.write(spec_content)
    print(f"Created spec file: {spec_path}")
    return spec_path


def build_package(project_root: Path, spec_path: Path):
    """Run PyInstaller to build the package."""
    print("\n" + "=" * 60)
    print("Building with PyInstaller...")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_path),
    ]

    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode == 0


def copy_model_files(project_root: Path, with_model: bool):
    """Copy model files to the dist directory."""
    dist_dir = project_root / "dist" / "qwen3-aligner"
    models_dir = dist_dir / "models"

    if not dist_dir.exists():
        print(f"ERROR: dist directory not found: {dist_dir}")
        return False

    # Create models directory
    models_dir.mkdir(exist_ok=True)

    if with_model:
        # Copy model files from source
        source_models = project_root / "qwen3_aligner" / "models"
        if source_models.exists():
            print(f"\nCopying model files from {source_models}...")
            for item in source_models.iterdir():
                dest = models_dir / item.name
                if item.is_file():
                    print(f"  {item.name}")
                    shutil.copy2(item, dest)
            print("Model files copied.")
        else:
            print(f"WARNING: Model directory not found: {source_models}")
            print("You'll need to manually copy the model files to:")
            print(f"  {models_dir}")
    else:
        print(f"\nModel files not included. Copy them manually to:")
        print(f"  {models_dir}")
        print("\nOr download using:")
        print("  qwen3-aligner download-model")
        print("\nExpected files:")
        print("  - config.json")
        print("  - model.safetensors")
        print("  - tokenizer_config.json")
        print("  - vocab.json")
        print("  - merges.txt")
        print("  - preprocessor_config.json")
        print("  - generation_config.json")
        print("  - chat_template.json")

    return True


def create_archive(project_root: Path):
    """Create a zip archive of the dist folder."""
    import platform

    dist_dir = project_root / "dist" / "qwen3-aligner"
    if not dist_dir.exists():
        print("ERROR: dist directory not found")
        return None

    # Determine platform name
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "x64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine

    archive_name = f"qwen3-aligner-{system}-{arch}"
    archive_path = project_root / "dist" / archive_name

    print(f"\nCreating archive: {archive_name}.zip")
    shutil.make_archive(str(archive_path), 'zip', project_root / "dist", "qwen3-aligner")

    return archive_path.with_suffix('.zip')


def get_package_size(project_root: Path):
    """Calculate total package size."""
    dist_dir = project_root / "dist" / "qwen3-aligner"
    if not dist_dir.exists():
        return 0

    total_size = 0
    for path in dist_dir.rglob("*"):
        if path.is_file():
            total_size += path.stat().st_size

    return total_size


def main():
    parser = argparse.ArgumentParser(description="Build Qwen3-Aligner package")
    parser.add_argument("--with-model", action="store_true",
                        help="Include model files in the package")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Use CPU-only PyTorch (not implemented yet)")
    parser.add_argument("--archive", action="store_true",
                        help="Create a zip archive after building")
    args = parser.parse_args()

    project_root = get_project_root()
    print(f"Project root: {project_root}")

    # Check dependencies
    print("\nChecking dependencies...")
    check_dependencies()

    # Create spec file
    print("\nCreating spec file...")
    spec_path = create_spec_file(project_root, args.with_model)

    # Build
    if not build_package(project_root, spec_path):
        print("\nERROR: Build failed!")
        sys.exit(1)

    # Copy model files
    print("\nSetting up model directory...")
    copy_model_files(project_root, args.with_model)

    # Create archive if requested
    archive_path = None
    if args.archive:
        archive_path = create_archive(project_root)

    # Report
    size = get_package_size(project_root)
    size_mb = size / (1024 * 1024)
    size_gb = size / (1024 * 1024 * 1024)

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Output directory: {project_root / 'dist' / 'qwen3-aligner'}")
    print(f"Package size: {size_mb:.1f} MB ({size_gb:.2f} GB)")
    if archive_path:
        archive_size = archive_path.stat().st_size / (1024 * 1024)
        print(f"Archive: {archive_path} ({archive_size:.1f} MB)")
    print("\nTo run:")
    print("  ./dist/qwen3-aligner/qwen3-aligner --help")
    print("  ./dist/qwen3-aligner/qwen3-aligner align -a audio.wav -t 'text' -l Chinese")
    print("  ./dist/qwen3-aligner/qwen3-aligner serve -p 8765")


if __name__ == "__main__":
    main()
