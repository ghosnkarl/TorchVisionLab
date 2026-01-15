# TorchVisionLab: Complete PyTorch & Computer Vision Mastery

A comprehensive PyTorch learning lab designed to take you from fundamentals to production-ready deep learning models. This hands-on course emphasizes computer vision, transformers, and modern deep learning techniques with practical implementations.

## 🎯 Course Overview

**Target Audience:** Intermediate Python developers, ML practitioners, CS students
**Prerequisites:** Python fundamentals, basic NumPy, high school mathematics
**Outcome:** Build and deploy production-ready deep learning models with PyTorch

## 📚 Course Structure

### 01. Basics (Fundamentals) ✅ COMPLETED

Master PyTorch fundamentals, neural networks, and essential training concepts.

- ✅ `00_introduction_to_machine_learning.ipynb` - ML concepts and terminology
- ✅ `01_pytorch_fundamentals.ipynb` - Tensors, operations, and PyTorch basics
- ✅ `02_neural_networks_theory.ipynb` - Mathematical foundations
- ✅ `03_neural_networks_intro.ipynb` - Building your first neural network
- ✅ `04_classification.ipynb` - Classification tasks and techniques
- ✅ `05_loss_functions_and_optimizers.ipynb` - Optimization fundamentals
- ✅ `06_datasets_and_dataloaders.ipynb` - Custom datasets and data loading

### 02. Computer Vision 🔁 IN PROGRESS

CNNs, transfer learning, object detection, and practical vision projects.

- ✅ `01_computer_vision.ipynb` - Introduction to CNNs and convolutions
- ✅ `02_transfer_learning.ipynb` - Using pre-trained models
- ✅ `03_experiment_tracking_tensorboard.ipynb` - TensorBoard integration
- ✅ `04_data_augmentation_and_preprocessing.ipynb` - Image transformations
- 🔜 `05_object_detection_fundamentals.ipynb` - Detection concepts
- 🔜 `06_yolo_implementation.ipynb` - YOLO from scratch
- 🔜 `07_image_classification_project.ipynb` - End-to-end classification

### 03. Vision Transformers 🔜 COMING SOON

Vision Transformers, attention mechanisms, and modern vision architectures.

- 🔜 `01_attention_mechanisms.ipynb` - Self-attention and multi-head attention
- 🔜 `02_vision_transformers.ipynb` - ViT implementation and theory
- 🔜 `03_advanced_vit_architectures.ipynb` - SWIN, DeiT, and variants
- 🔜 `04_vision_transformer_project.ipynb` - Practical ViT application

### 04. Segmentation 🔜 COMING SOON

Semantic segmentation, U-Net, and advanced architectures.

- 🔜 `01_segmentation_theory.ipynb` - Segmentation fundamentals
- 🔜 `02_unet_fundamentals.ipynb` - U-Net architecture deep dive
- 🔜 `03_oxford_pets_segmentation.ipynb` - Hands-on segmentation project
- 🔜 `04_advanced_segmentation_architectures.ipynb` - DeepLab, Mask R-CNN
- 🔜 `05_segmentation_metrics_and_evaluation.ipynb` - IoU, Dice coefficient
- 🔜 `06_real_time_segmentation_and_mobile_deployment.ipynb` - Optimization

### 05. GANs (Generative Adversarial Networks) 🔜 COMING SOON

Generative models, GANs, and image-to-image translation.

- 🔜 `01_introduction_to_gans.ipynb` - GAN theory and fundamentals
- 🔜 `02_dcgan_implementation.ipynb` - Deep Convolutional GAN
- 🔜 `03_conditional_gan.ipynb` - Conditional generation
- 🔜 `04_pix2pix_image_translation.ipynb` - Paired image translation
- 🔜 `05_cyclegan.ipynb` - Unpaired image translation
- 🔜 `06_gan_training_tricks.ipynb` - Stability and best practices
- 🔜 `07_gan_project.ipynb` - Creative GAN project

### 06. Advanced Training 🔜 COMING SOON

Production-level training techniques for robust model development.

- 🔜 `01_learning_rate_scheduling.ipynb` - StepLR, CosineAnnealing, ReduceLROnPlateau, OneCycleLR
- 🔜 `02_early_stopping_and_checkpointing.ipynb` - Model saving, best model selection, resume training
- 🔜 `03_mixed_precision_training.ipynb` - AMP, float16 training
- 🔜 `04_gradient_accumulation_and_clipping.ipynb` - Large batch training, gradient explosion
- 🔜 `05_distributed_training.ipynb` - Multi-GPU, DDP, parallel training
- 🔜 `06_pytorch_lightning.ipynb` - Clean, modular training loops
- 🔜 `07_hyperparameter_tuning.ipynb` - Grid search, random search, Optuna
- 🔜 `08_debugging_and_profiling.ipynb` - Finding bottlenecks, memory issues, gradient checking
- 🔜 `09_callbacks_and_hooks.ipynb` - Custom training logic, feature extraction

### 07. Model Optimization 🔜 COMING SOON

Profiling, quantization, pruning, and model compression.

- 🔜 `01_model_profiling.ipynb` - Bottleneck analysis
- 🔜 `02_quantization.ipynb` - INT8 quantization
- 🔜 `03_pruning.ipynb` - Weight pruning techniques
- 🔜 `04_knowledge_distillation.ipynb` - Teacher-student models
- 🔜 `05_onnx_export.ipynb` - Cross-platform deployment
- 🔜 `06_mobile_optimization.ipynb` - Edge device deployment

### 08. Production Deployment 🔜 COMING SOON

Model serving, containerization, cloud deployment, and MLOps.

- 🔜 `01_model_serving_fastapi.ipynb` - REST API with FastAPI
- 🔜 `02_docker_containerization.ipynb` - Packaging models
- 🔜 `03_cloud_deployment_aws.ipynb` - AWS SageMaker and Lambda
- 🔜 `04_monitoring_and_logging.ipynb` - MLOps fundamentals
- 🔜 `05_end_to_end_pipeline.ipynb` - Complete production pipeline

### 09. Advanced Topics 🔜 COMING SOON

Deep dives into PyTorch internals, production training, and optimization techniques.

- 🔜 `01_tensors_advanced.ipynb` - Advanced tensor operations
- 🔜 `02_custom_autograd_functions.ipynb` - Extending PyTorch
- 🔜 `03_custom_layers_and_modules.ipynb` - Building custom blocks
- 🔜 `04_debugging_pytorch.ipynb` - Common issues and solutions
- 🔜 `05_performance_optimization.ipynb` - Speed and memory optimization
- 🔜 `06_autograd_deep_dive.ipynb` - Computational graphs and autograd internals
- 🔜 `07_production_training.ipynb` - LR scheduling, checkpointing, mixed precision
- 🔜 `08_comprehensive_evaluation.ipynb` - Advanced metrics and model evaluation
- 🔜 `09_regularization_mastery.ipynb` - Dropout, batch norm, weight decay

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.11 or higher (recommended)
python --version

# Git
git --version

# uv (ultra-fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Quick Setup (Automated)

The easiest way to get started is using our automated setup script:

```bash
# Clone the repository
git clone https://github.com/yourusername/TorchVisionLab.git
cd TorchVisionLab

# Run the setup script
chmod +x setup.sh
./setup.sh
```

The setup script will:

1. Check for uv installation (install if missing)
2. Let you choose Python version (3.10, 3.11, or 3.12)
3. Initialize a uv project
4. Create a virtual environment
5. Detect your platform (macOS/Linux/Windows)
6. Install PyTorch with appropriate accelerator support:
   - **macOS:** MPS (Metal Performance Shaders) for Apple Silicon
   - **Linux/Windows:** Choose CUDA 12.8, 12.1, 11.8, or CPU-only
7. Install all required dependencies
8. Create a Jupyter kernel
9. Verify the installation

### Manual Setup

If you prefer manual installation:

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/TorchVisionLab.git
cd TorchVisionLab
```

#### 2. Set Up Python Environment

```bash
# Pin Python version (3.11 recommended)
uv python pin 3.11

# Create virtual environment
uv venv --python 3.11

# Activate environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Option A: Install from pyproject.toml (recommended)
uv sync

# Option B: Install PyTorch with specific CUDA version
# For CUDA 12.8:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# For CUDA 12.1:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For macOS (includes MPS support):
uv add torch torchvision

# Then install remaining dependencies
uv sync
```

#### 4. Set Up Jupyter Kernel

```bash
# Create Jupyter kernel for this project
uv run python -m ipykernel install --user --name=torchvisionlab --display-name="Python (TorchVisionLab)"
```

#### 5. Verify Installation

```bash
# Check PyTorch installation
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

Expected output:

```
PyTorch: 2.9.1+cu128  # or your version
CUDA: True            # True if NVIDIA GPU, False otherwise
MPS: True             # True on Apple Silicon Macs
```

### Starting Jupyter

```bash
# Launch Jupyter Notebook
uv run jupyter notebook

# Or use Jupyter Lab
uv run jupyter lab
```

Then select the **"Python (TorchVisionLab)"** kernel when opening notebooks.

### Using Google Colab

If you prefer using Google Colab (free GPU access):

1. Open any notebook from this repository in GitHub
2. Change the URL from `github.com` to `githubtocolab.com`
3. Or manually upload to Google Colab
4. Enable GPU: Runtime → Change runtime type → Hardware accelerator → T4 GPU
5. Run the first cell to install dependencies:

```python
!pip install torch torchvision pytorch-lightning transformers
```

## 📖 How to Use This Course

### Recommended Learning Path

1. **Start with Basics (Section 01)** - Build strong fundamentals
2. **Choose your interest:**
   - Computer Vision → Sections 02, 03, 04, 05
   - Production & Deployment → Sections 06, 07, 08
3. **Deep dive into Advanced Topics (Section 09)** - Production-level techniques
4. **Build projects** - Apply everything you've learned

### For Each Notebook

1. **Read the introduction** - Understand the goals
2. **Run code cells sequentially** - Don't skip cells
3. **Experiment** - Modify parameters, try variations
4. **Complete exercises** - Practice makes perfect
5. **Review summary** - Reinforce key concepts

### Study Tips

- **Code along:** Type the code yourself, don't just run cells
- **Take notes:** Use markdown cells to add your observations
- **Break things:** Intentionally make errors to understand debugging
- **Build projects:** Apply concepts to your own datasets
- **Track progress:** Mark completed notebooks

## 🛠️ Project Structure

```
TorchVisionLab/
├── 01_basics/                 # ⏳ IN PROGRESS - Fundamentals
├── 02_computer_vision/        # 🔜 COMING SOON - CNNs and vision
├── 03_vision_transformers/    # 🔜 COMING SOON - Vision Transformers
├── 04_segmentation/           # 🔜 COMING SOON - Image segmentation
├── 05_gans/                   # 🔜 COMING SOON - Generative models
├── 06_advanced_training/      # 🔜 COMING SOON - Training techniques
├── 07_model_optimization/     # 🔜 COMING SOON - Compression and optimization
├── 08_production_deployment/  # 🔜 COMING SOON - Deployment
├── 09_advanced_topics/        # 🔜 COMING SOON - Advanced techniques
├── 12_images/                 # Course images and media
├── 13_datasets/               # Sample datasets
├── .venv/                     # Virtual environment (created by setup)
├── setup.sh                   # Automated setup script
├── pyproject.toml            # Project dependencies (uv)
├── uv.lock                   # Dependency lock file
├── main.py                   # Main entry point
└── README.md                 # This file
```

## 🎓 Learning Resources

### Official Documentation

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [uv Documentation](https://docs.astral.sh/uv/)

### Recommended Reading

- Deep Learning Book (Goodfellow et al.)
- Dive into Deep Learning (d2l.ai)
- Papers with Code (paperswithcode.com)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

- **Report bugs:** Open an issue with reproduction steps
- **Suggest improvements:** Propose new topics or enhancements
- **Fix typos:** Submit PRs for documentation improvements
- **Add examples:** Contribute additional examples or use cases
- **Share projects:** Show what you built with this course

### Development Workflow

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/TorchVisionLab.git
cd TorchVisionLab

# Create a new branch
git checkout -b feature/your-feature-name

# Make changes and test
uv run pytest tests/  # if tests exist

# Commit and push
git add .
git commit -m "Description of changes"
git push origin feature/your-feature-name

# Open a Pull Request
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Acknowledgments

- PyTorch team for the amazing framework
- Fast.ai for inspiration on teaching deep learning
- Astral (uv creators) for the blazing-fast package manager
- All contributors and students who provide feedback

## 📊 Project Status

- **Current Phase:** Fundamentals (Section 01)
- **Completed Notebooks:** 1
- **In Progress:** 1
- **Upcoming Sections:** 8+ sections planned
- **Last Updated:** January 2026

## ⭐ Star This Repository

If you find this course helpful, please consider giving it a star!

---

**Happy Learning! 🚀**
