# Keras Deep Learning Course

A comprehensive deep learning course using Keras and TensorFlow, organized from beginner to advanced topics.

## Course Structure

### 00 - Introduction
Foundation concepts and course overview.

### 01 - Basics
- **00_fashion_mnist_basic_cnn.ipynb**: Basic CNN on Fashion MNIST dataset
- **01_learn_sine_regression.ipynb**: Regression task - learning a sine function
- **02_boston_house_price_regression.ipynb**: House price prediction with regression

### 02 - Image Classification
- **00_cifar10_classification.ipynb**: CIFAR-10 image classification from scratch

### 03 - Advanced CNN Architectures
- **00_densenet_architecture.ipynb**: DenseNet architecture and implementation

### 04 - Regularization Techniques
- **00_cifar10_regularization.ipynb**: Regularization techniques applied to CIFAR-10
- **01_imdb_overfit_underfit.ipynb**: Understanding overfitting and underfitting with IMDB data

### 05 - Transfer Learning
- **00_imagenet_transfer_learning.ipynb**: Transfer learning using ImageNet pretrained models
- **01_visualize_heat_maps.ipynb**: Visualizing attention maps and model interpretability

### 06 - Image Segmentation
- **00_cell_tissue_segmentation.ipynb**: Semantic segmentation of cells and tissues
- **01_mitosis_detection_brightfield.ipynb**: Detecting mitosis in HeLa cells (brightfield)
- **02_mitosis_detection_phase_contrast.ipynb**: Detecting mitosis with phase contrast microscopy
- **03_mitosis_xenopus_detection.ipynb**: Mitosis detection in Xenopus cells

### 07 - Time Series
- **00_time_series_training.ipynb**: Training models for time series data
- **01_time_series_prediction.ipynb**: Predicting time series responses to stimuli

### 08 - NLP & Text Processing
- **00_text_classification_welcome.ipynb**: Introduction to text classification
- **01_text_classification_deployment.ipynb**: Deploying text classification models
- **02_imdb_reviews_classification.ipynb**: Sentiment analysis on IMDB reviews

### 09 - Advanced Topics

#### Physics-Informed Neural Networks (PINNs) & PDE Solving
- **00_pde_introductory.ipynb**: Introduction to solving PDEs with neural networks
- **01_pde_modulus_anatomy.ipynb**: NVIDIA Modulus framework anatomy
- **02_pde_spring_mass_problem.ipynb**: Spring-mass system modeling
- **03_pde_spring_mass_inverse.ipynb**: Inverse problem for spring-mass systems
- **04_pde_diffusion_problem.ipynb**: Diffusion equation solving
- **05_pde_diffusion_parameterized.ipynb**: Parameterized diffusion problems
- **06_pde_cfd_problem.ipynb**: Computational fluid dynamics problems
- **07_pde_challenge_1.ipynb**: Challenge problem 1
- **08_pde_challenge_2.ipynb**: Challenge problem 2
- **09_pde_challenge_3.ipynb**: Challenge problem 3
- **10_pde_fno_darcy.ipynb**: Fourier Neural Operators (FNO) for Darcy flow
- **11_pde_parameterized_inverse.ipynb**: Parameterized inverse problems

#### Video & Real-time AI
- **12_video_ai_introduction.ipynb**: Introduction to video AI
- **13_video_realtime_ai_applications.ipynb**: Real-time video AI applications
- **14_video_deepstream_sdk.ipynb**: NVIDIA DeepStream SDK introduction
- **15_video_deepstream_application.ipynb**: Building DeepStream applications
- **16_video_mdnn_deepstream_application.ipynb**: Multi-DNN DeepStream applications

## Learning Path

### For Beginners
1. Start with **01_Basics** - Learn fundamental concepts
2. Move to **02_Image_Classification** - Classic image tasks
3. Try **03_Advanced_CNN** - More sophisticated architectures

### For Intermediate Learners
1. Explore **04_Regularization** - Prevent overfitting
2. Learn **05_Transfer_Learning** - Leverage pretrained models
3. Study **06_Image_Segmentation** - Pixel-level predictions

### For Advanced Learners
1. **07_Time_Series** - Sequential data processing
2. **08_NLP_Text** - Natural language understanding
3. **09_Advanced_Topics** - Cutting-edge techniques:
   - Physics-informed networks for scientific computing
   - Real-time video processing with DeepStream

## Prerequisites

- Python 3.7+
- TensorFlow/Keras
- NumPy, Pandas, Matplotlib
- Jupyter Notebook

## Setup

```bash
pip install tensorflow keras numpy pandas matplotlib jupyter
```

## Recommended Order

1. **Basics** (01) → Foundation
2. **Classification** (02) → Core skills
3. **CNN Architectures** (03) → Deeper understanding
4. **Regularization** (04) → Improve models
5. **Transfer Learning** (05) → Efficient learning
6. **Segmentation** (06) → Advanced vision
7. **Time Series** (07) → Sequential data
8. **NLP** (08) → Language understanding
9. **Advanced** (09) → Specialized applications

## Key Topics Covered

- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs) for time series
- Transfer Learning and fine-tuning
- Regularization and dropout
- Batch normalization
- Image segmentation
- Text classification and NLP
- Physics-informed neural networks
- Real-time video processing

## Notes

- Each notebook is self-contained and can be run independently
- Notebooks include explanations, visualizations, and exercises
- Data is either downloaded automatically or provided in the repo
