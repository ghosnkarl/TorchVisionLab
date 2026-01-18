#!/bin/bash

# PyTorch Project Dependencies Installer
# This script sets up a new Python project with pip and installs all required dependencies

set -e  # Exit on error

echo "==================================="
echo "PyTorch Project Setup with pip"
echo "==================================="
echo ""

# Detect platform first
PLATFORM=$(uname -s)

# Determine Python command based on platform
if [[ "$PLATFORM" == MINGW* ]] || [[ "$PLATFORM" == MSYS* ]]; then
    # Windows uses 'python' not 'python3'
    PYTHON_CMD="python"
else
    # Linux/macOS uses 'python3'
    PYTHON_CMD="python3"
fi

# Check if Python is installed
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.10 or higher."
    exit 1
fi

echo "✅ Python found: $($PYTHON_CMD --version)"
echo ""

if [[ "$PLATFORM" == "Darwin" ]]; then
    # macOS - use default PyPI (has MPS support for Apple Silicon)
    echo "ℹ️  Detected macOS - will install PyTorch with MPS (Metal) support"
    USE_CUSTOM_INDEX=false
elif [[ "$PLATFORM" == "Linux" ]] || [[ "$PLATFORM" == MINGW* ]] || [[ "$PLATFORM" == MSYS* ]]; then
    # Linux or Windows - ask for CUDA support
    echo "Select PyTorch installation type:"
    echo "  1) CUDA 12.8 (for NVIDIA GPUs - latest)"
    echo "  2) CUDA 12.6 (for NVIDIA GPUs)"
    echo "  3) CUDA 11.8 (for older NVIDIA GPUs)"
    echo "  4) CPU only (no GPU support)"
    read -p "Enter choice [1-4] (default: 1): " cuda_choice
    cuda_choice=${cuda_choice:-1}

    case $cuda_choice in
        1) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
        2) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
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

# Create virtual environment
echo "🔧 Creating virtual environment..."
$PYTHON_CMD -m venv .venv
echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$PLATFORM" == MINGW* ]] || [[ "$PLATFORM" == MSYS* ]]; then
    # Windows (Git Bash)
    source .venv/Scripts/activate
else
    # Linux/macOS
    source .venv/bin/activate
fi
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip
echo ""

echo "🚀 Installing dependencies..."
echo ""

# Install PyTorch
echo "📥 Installing torch and torchvision..."
if [ "$USE_CUSTOM_INDEX" = true ]; then
    pip install torch torchvision --index-url $TORCH_INDEX
else
    pip install torch torchvision
fi
echo ""

# Install PyTorch Lightning
echo "📥 Installing PyTorch Lightning..."
pip install pytorch-lightning
echo ""

# Install data science stack
echo "📥 Installing data science libraries..."
pip install numpy pandas scikit-learn scipy
echo ""

# Install visualization libraries
echo "📥 Installing visualization libraries..."
pip install matplotlib seaborn
echo ""

# Install image and text processing
echo "📥 Installing image and text processing libraries..."
pip install pillow opencv-python
pip install transformers torchtext
echo ""

# Install experiment tracking
echo "📥 Installing experiment tracking tools..."
pip install tensorboard mlflow
echo ""

# Install utilities
echo "📥 Installing utilities..."
pip install tqdm pyyaml python-dotenv
echo ""

# Install Jupyter support
echo "📥 Installing Jupyter support..."
pip install jupyter ipykernel notebook ipywidgets torchsummary
echo ""

# Create Jupyter kernel
echo "🔧 Creating Jupyter kernel..."
python -m ipykernel install --user --name=pytorch-env --display-name="Python (PyTorch)"
echo ""

# Verify installation
echo "✅ Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "==================================="
echo "✅ Installation Complete!"
echo "==================================="
echo ""
echo "To activate the virtual environment:"
if [[ "$PLATFORM" == MINGW* ]] || [[ "$PLATFORM" == MSYS* ]]; then
    echo "  source .venv/Scripts/activate"
else
    echo "  source .venv/bin/activate"
fi
