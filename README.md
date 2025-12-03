# Keras Deep Learning Course

A comprehensive, structured deep learning course using Keras and TensorFlow for developers who want to master AI and deep learning.

## 📚 Course Overview

This repository contains a complete, progressive deep learning curriculum with **35+ Jupyter notebooks** organized into 10 progressive modules, from beginner fundamentals to advanced research topics.

### Course Modules

- **01_Basics** - CNN fundamentals and regression tasks
- **02_Image_Classification** - Image classification from scratch
- **03_Advanced_CNN** - Advanced architectures (DenseNet, etc.)
- **04_Regularization** - Preventing overfitting and improving generalization
- **05_Transfer_Learning** - Leveraging pretrained models
- **06_Image_Segmentation** - Pixel-level predictions and medical imaging
- **07_Time_Series** - Sequence modeling and forecasting
- **08_NLP_Text** - Natural language processing and text classification
- **09_Advanced_Topics** - Physics-informed neural networks, PDEs, real-time video AI

**→ Full course documentation**: See [Keras/README.md](Keras/README.md)

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/goodplacedeeplearning.git
cd goodplacedeeplearning

# Install dependencies
pip install -e .
```

Or with conda:

```bash
conda create -n keras-course python=3.9
conda activate keras-course
pip install -e .
```

### Start Learning

```bash
# Navigate to the Keras folder
cd Keras

# Launch Jupyter
jupyter notebook

# Start with 01_Basics
```

## 📋 System Requirements

- **RAM**: 8GB minimum (16GB recommended for transfer learning)
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA support)
- **Storage**: 10GB free space for datasets

## 🛠️ Dependencies

The course requires:

- **TensorFlow/Keras** 2.10+
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** & **Seaborn** - Visualization
- **Scikit-learn** - Machine learning utilities
- **Jupyter** - Interactive notebooks
- **Pillow** - Image processing
- **h5py** - HDF5 support
- **NVIDIA Modulus** (optional, for PDE notebooks)

All dependencies are specified in `setup.cfg` and will be installed automatically.

## 📖 Learning Paths

### Beginner Path (1-2 weeks)
1. **01_Basics** - Learn CNN fundamentals
2. **02_Image_Classification** - Classic image tasks
3. **03_Advanced_CNN** - Deeper architectures
4. **04_Regularization** - Improve your models

### Intermediate Path (2-3 weeks)
1. Complete beginner path first
2. **05_Transfer_Learning** - Efficient learning from pretrained models
3. **06_Image_Segmentation** - Advanced vision tasks
4. **07_Time_Series** - Sequential data

### Advanced Path (3+ weeks)
1. Complete beginner and intermediate paths
2. **08_NLP_Text** - Natural language understanding
3. **09_Advanced_Topics** - Research-level topics

## 🎯 Key Topics Covered

- **Deep Learning Fundamentals**
  - Convolutional Neural Networks (CNNs)
  - Recurrent Neural Networks (RNNs)
  - Attention mechanisms

- **Computer Vision**
  - Image classification
  - Semantic segmentation
  - Object detection (implicit)
  - Medical image analysis

- **Natural Language Processing**
  - Text classification
  - Sentiment analysis
  - Sequence models

- **Advanced Techniques**
  - Transfer learning and fine-tuning
  - Regularization and dropout
  - Batch normalization
  - Physics-informed neural networks
  - Fourier Neural Operators

- **Practical Applications**
  - Real-time video processing
  - Time series forecasting
  - Model deployment

## 📊 Notebook Statistics

| Module | Notebooks | Topics |
|--------|-----------|--------|
| 01_Basics | 3 | CNN, Regression |
| 02_Classification | 1 | Image Classification |
| 03_Advanced_CNN | 1 | DenseNet, Advanced Architectures |
| 04_Regularization | 2 | Overfitting, Underfitting |
| 05_Transfer_Learning | 2 | Pretrained Models, Visualization |
| 06_Segmentation | 4 | Semantic Segmentation, Medical Imaging |
| 07_Time_Series | 2 | Sequence Modeling, Forecasting |
| 08_NLP_Text | 3 | Text Classification, NLP |
| 09_Advanced_Topics | 17 | PINNs, PDEs, Video AI, DeepStream |
| **Total** | **35+** | **Comprehensive Deep Learning** |

## 💡 Tips for Learning

1. **Run every cell** - Don't just read, execute and experiment
2. **Modify code** - Change hyperparameters and observe results
3. **Visualize** - Pay attention to plots and interpretations
4. **Take notes** - Jot down key concepts and insights
5. **Challenge yourself** - Try to solve problems before looking at solutions
6. **GPU acceleration** - Use GPU for faster training if available

## 🔧 Development Setup

For development and contributions:

```bash
# Clone and install in editable mode
git clone https://github.com/yourusername/goodplacedeeplearning.git
cd goodplacedeeplearning
pip install -e ".[dev]"

# Run linters and formatters
black Keras/
flake8 Keras/
```

## 📝 Notebook Naming Convention

Each notebook follows the pattern: `NN_notebook_name.ipynb`

- `NN` = Lesson number within the module
- `notebook_name` = Descriptive name of the content

Example: `00_fashion_mnist_basic_cnn.ipynb`

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a new branch for your changes
3. Commit with clear messages
4. Push and create a pull request

## 📄 License

This course material is provided for educational purposes.

## 🎓 About This Course

This course was designed to provide a comprehensive, hands-on introduction to deep learning using Keras and TensorFlow. It progresses from fundamental concepts to cutting-edge research applications, making it suitable for:

- Students learning deep learning for the first time
- Practitioners looking to expand their skills
- Researchers exploring advanced techniques
- Developers building production ML systems

## 🆘 Support

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: See individual notebook docs and [Keras/README.md](Keras/README.md)

## 🔗 Useful Resources

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [NVIDIA Modulus (for PDE notebooks)](https://developer.nvidia.com/modulus)
- [DeepStream SDK](https://developer.nvidia.com/deepstream)

---

**Happy Learning!** 🚀

Start with `Keras/01_Basics` and work your way through the course at your own pace.
