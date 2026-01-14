#!/bin/bash

# PyTorch Project Dependencies Installer
# This script sets up a new Python project with uv and installs all required dependencies

set -e  # Exit on error

echo "==================================="
echo "PyTorch Project Setup with UV"
echo "==================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed successfully"
    echo ""
    echo "⚠️  Please restart your shell and run this script again to complete setup."
    exit 0
fi

echo "✅ uv found: $(uv --version)"
echo ""

# Ask for Python version
echo "Select Python version:"
echo "  1) Python 3.11 (Recommended)"
echo "  2) Python 3.10"
echo "  3) Python 3.12"
read -p "Enter choice [1-3] (default: 1): " python_choice
python_choice=${python_choice:-1}

case $python_choice in
    1) PYTHON_VERSION="3.11" ;;
    2) PYTHON_VERSION="3.10" ;;
    3) PYTHON_VERSION="3.12" ;;
    *) PYTHON_VERSION="3.11" ;;
esac

# Initialize uv project
echo "🔧 Initializing uv project..."
uv init --no-readme
echo ""

echo "📦 Setting Python version to $PYTHON_VERSION"
uv python pin $PYTHON_VERSION
echo ""

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv --python $PYTHON_VERSION
echo ""

# Detect platform
PLATFORM=$(uname -s)

if [[ "$PLATFORM" == "Darwin" ]]; then
    # macOS - use default PyPI (has MPS support for Apple Silicon)
    echo "ℹ️  Detected macOS - will install PyTorch with MPS (Metal) support"
    USE_CUSTOM_INDEX=false
elif [[ "$PLATFORM" == "Linux" ]] || [[ "$PLATFORM" == MINGW* ]] || [[ "$PLATFORM" == MSYS* ]]; then
    # Linux or Windows - ask for CUDA support
    echo "Select PyTorch installation type:"
    echo "  1) CUDA 12.8 (for NVIDIA GPUs - latest)"
    echo "  2) CUDA 12.1 (for NVIDIA GPUs)"
    echo "  3) CUDA 11.8 (for older NVIDIA GPUs)"
    echo "  4) CPU only (no GPU support)"
    read -p "Enter choice [1-4] (default: 1): " cuda_choice
    cuda_choice=${cuda_choice:-1}

    case $cuda_choice in
        1) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
        2) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        3) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        4) TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
        *) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
    esac
    USE_CUSTOM_INDEX=true
else
    echo "⚠️  Unknown platform - using default PyPI"
    USE_CUSTOM_INDEX=false
fi

echo ""
echo "🚀 Installing dependencies..."
echo ""

# Install PyTorch
echo "📥 Installing torch and torchvision..."
if [ "$USE_CUSTOM_INDEX" = true ]; then
    uv pip install torch torchvision --index-url $TORCH_INDEX
else
    uv add torch torchvision
fi
echo ""

# Install PyTorch Lightning
echo "📥 Installing PyTorch Lightning..."
uv add pytorch-lightning
echo ""

# Install data science stack
echo "📥 Installing data science libraries..."
uv add numpy pandas scikit-learn scipy
echo ""

# Install visualization libraries
echo "📥 Installing visualization libraries..."
uv add matplotlib seaborn
echo ""

# Install image and text processing
echo "📥 Installing image and text processing libraries..."
uv add pillow opencv-python
uv add transformers torchtext
echo ""

# Install experiment tracking
echo "📥 Installing experiment tracking tools..."
uv add tensorboard mlflow
echo ""

# Install utilities
echo "📥 Installing utilities..."
uv add tqdm pyyaml python-dotenv
echo ""

# Install Jupyter support
echo "📥 Installing Jupyter support..."
uv add jupyter ipykernel notebook ipywidgets torchsummary
echo ""

# Create Jupyter kernel
echo "🔧 Creating Jupyter kernel..."
uv run python -m ipykernel install --user --name=pytorch-env --display-name="Python (PyTorch)"
echo ""

# Verify installation
echo "✅ Verifying PyTorch installation..."
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "==================================="
echo "✅ Installation Complete!"
echo "==================================="
