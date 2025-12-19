# PyTorch Deep Learning Notebooks

A complete collection of 18 PyTorch notebooks converted from Keras, covering fundamental to advanced deep learning topics.

## Quick Start

### 1. Installation
```bash
# Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy matplotlib scikit-learn
```

### 2. Run a Notebook
```bash
jupyter notebook PyTorch/01_Basics/00_fashion_mnist_basic_cnn.ipynb
```

## Notebook Structure

### Level 1: Basics (3 notebooks)
**Location**: `PyTorch/01_Basics/`

- **00_fashion_mnist_basic_cnn.ipynb** - Image Classification Fundamentals
  - Learn: CNN architecture, model.fit() equivalent, training loops
  - Dataset: Fashion MNIST (28x28 images, 10 classes)
  - Model: Simple CNN (Flatten → Dense → Dense)
  - Duration: ~5-10 minutes

- **01_learn_sine_regression.ipynb** - Regression & Training Dynamics
  - Learn: Regression with neural networks, train/val split
  - Dataset: Synthetic sine wave
  - Model: Multi-layer perceptron
  - Duration: ~5 minutes

- **02_boston_house_price_regression.ipynb** - Real-world Regression
  - Learn: Feature scaling, regression metrics (MSE, RMSE, MAE)
  - Dataset: Boston housing dataset
  - Model: Dense neural network
  - Duration: ~5 minutes

### Level 2: Image Classification (1 notebook)
**Location**: `PyTorch/02_Image_Classification/`

- **00_cifar10_classification.ipynb** - Binary Image Classification
  - Learn: Binary classification, CNN architecture, data augmentation
  - Dataset: CIFAR-10 (cats vs dogs)
  - Model: Deeper CNN with regularization
  - Duration: ~10 minutes

### Level 3: Advanced CNN (1 notebook)
**Location**: `PyTorch/03_Advanced_CNN/`

- **00_densenet_architecture.ipynb** - Modern CNN Architecture
  - Learn: DenseNet, skip connections, pre-trained models
  - Architecture: DenseNet121 from ImageNet
  - Task: Feature extraction and fine-tuning
  - Duration: ~10 minutes

### Level 4: Regularization (2 notebooks)
**Location**: `PyTorch/04_Regularization/`

- **00_cifar10_regularization.ipynb** - Preventing Overfitting
  - Learn: L2 regularization, Dropout, Batch Normalization
  - Techniques: weight_decay, Dropout layers, augmentation
  - Dataset: CIFAR-10
  - Duration: ~10 minutes

- **01_imdb_overfit_underfit.ipynb** - Model Capacity Analysis
  - Learn: Overfitting vs underfitting, model complexity
  - Models: Simple, Medium, Complex (3 variants)
  - Analysis: Train/val curve comparison
  - Duration: ~5 minutes

### Level 5: Transfer Learning (2 notebooks)
**Location**: `PyTorch/05_Transfer_Learning/`

- **00_imagenet_transfer_learning.ipynb** - Fine-tuning Pre-trained Models
  - Learn: Transfer learning, layer freezing, selective optimization
  - Base Model: ResNet50 (ImageNet pre-trained)
  - Strategy: Freeze backbone, retrain classifier
  - Duration: ~15 minutes

- **01_visualize_heat_maps.ipynb** - Model Interpretability
  - Learn: Feature visualization, Grad-CAM, attention maps
  - Techniques: Hook-based feature extraction, gradient visualization
  - Output: Class activation maps
  - Duration: ~10 minutes

### Level 6: Image Segmentation (4 notebooks)
**Location**: `PyTorch/06_Image_Segmentation/`

All use the same U-Net architecture with different dataset themes:

- **00_cell_tissue_segmentation.ipynb** - Cell segmentation
- **01_mitosis_detection_brightfield.ipynb** - Brightfield microscopy
- **02_mitosis_detection_phase_contrast.ipynb** - Phase contrast microscopy
- **03_mitosis_xenopus_detection.ipynb** - Xenopus embryo segmentation

Features:
- Learn: U-Net encoder-decoder, skip connections, upsampling
- Architecture: Fully convolutional network with dense connections
- Loss: BCEWithLogits for binary segmentation
- Duration: ~10 minutes each

### Level 7: Time Series (2 notebooks)
**Location**: `PyTorch/07_Time_Series/`

- **00_time_series_training.ipynb** - LSTM Fundamentals
- **01_time_series_prediction.ipynb** - RNN for Sequences

Features:
- Learn: LSTM, GRU, sequence-to-sequence, temporal patterns
- Architecture: Multi-layer LSTM
- Dataset: Synthetic time series signals
- Task: Next-step prediction
- Duration: ~10 minutes each

### Level 8: NLP & Text Classification (3 notebooks)
**Location**: `PyTorch/08_NLP_Text/`

- **00_text_classification_welcome.ipynb** - Text Classification Intro
- **01_text_classification_deployment.ipynb** - Production-ready model
- **02_imdb_reviews_classification.ipynb** - Sentiment analysis

Features:
- Learn: Embeddings, text encoding, sequence processing
- Architecture: Embedding → LSTM → Dense
- Task: Binary text classification
- Dataset: Synthetic review data
- Duration: ~10 minutes each

## Learning Path Recommendations

### Beginner (2-3 hours)
1. `01_Basics/00_fashion_mnist_basic_cnn.ipynb` - Start here!
2. `01_Basics/01_learn_sine_regression.ipynb`
3. `02_Image_Classification/00_cifar10_classification.ipynb`

### Intermediate (2-3 hours)
4. `04_Regularization/01_imdb_overfit_underfit.ipynb` - Understand generalization
5. `03_Advanced_CNN/00_densenet_architecture.ipynb` - Modern architectures
6. `01_Basics/02_boston_house_price_regression.ipynb` - Real data

### Advanced (3-4 hours)
7. `05_Transfer_Learning/00_imagenet_transfer_learning.ipynb` - Leverage pre-trained models
8. `05_Transfer_Learning/01_visualize_heat_maps.ipynb` - Interpretability
9. `06_Image_Segmentation/00_cell_tissue_segmentation.ipynb` - Dense prediction
10. `07_Time_Series/00_time_series_training.ipynb` - Sequence modeling
11. `08_NLP_Text/00_text_classification_welcome.ipynb` - NLP basics

## Key Concepts Covered

### Architectures
- ✓ Fully Connected Networks (FCN)
- ✓ Convolutional Neural Networks (CNN)
- ✓ Recurrent Neural Networks (RNN/LSTM/GRU)
- ✓ U-Net (Encoder-Decoder with skip connections)
- ✓ DenseNet (Dense connections)
- ✓ Embedding layers (for text)

### Techniques
- ✓ Supervised learning
- ✓ Classification and regression
- ✓ Transfer learning
- ✓ Data augmentation
- ✓ Regularization (L2, Dropout, Batch Norm)
- ✓ Feature visualization
- ✓ Custom training loops

### Domains
- ✓ Computer Vision (image classification, segmentation)
- ✓ Time Series (temporal prediction)
- ✓ Natural Language Processing (text classification)
- ✓ Regression (continuous prediction)

## Common Patterns

### Model Definition
```python
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

model = MyNet().to(device)
```

### Training Loop
```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Evaluation
```python
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        # Compute metrics
```

## Tips & Best Practices

### 1. GPU Acceleration
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x, y = x.to(device), y.to(device)
```

### 2. Monitoring Training
- Track loss curves (train vs validation)
- Monitor metrics (accuracy, precision, recall)
- Save best models with checkpoints
- Use early stopping for regularization

### 3. Data Handling
- Always normalize/standardize inputs
- Use appropriate train/val/test splits
- Apply augmentation only to training data
- Handle class imbalance with weighted loss

### 4. Model Debugging
- Start with simple models first
- Gradually increase complexity
- Visualize intermediate representations
- Check gradient flow with `model.eval()`

## Troubleshooting

### GPU Out of Memory
- Reduce batch size
- Use gradient accumulation
- Use mixed precision training
- Try CPU training first

### Model Not Learning
- Check data normalization
- Verify loss function choice
- Increase learning rate or use scheduling
- Add regularization (Dropout, L2)

### Slow Training
- Enable GPU acceleration
- Use appropriate batch size (32-128)
- Profile bottlenecks
- Use faster data loading (num_workers)

## Additional Resources

### PyTorch Documentation
- [PyTorch Official Docs](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TorchVision](https://pytorch.org/vision/stable/index.html)

### Learning Materials
- PyTorch Fundamentals course
- FastAI courses
- Stanford CS231N (CNNs)
- Stanford CS224N (NLP)

## Comparison with Original Keras

| Aspect | Keras | PyTorch |
|--------|-------|---------|
| Model | Sequential/Functional | nn.Module |
| Training | model.fit() | Custom loop |
| Data | Generators | DataLoader |
| Layers | keras.layers | torch.nn |
| Loss | compile() | nn.* classes |
| Optimization | Built-in | torch.optim |
| Device | Implicit | Explicit (.to) |

## File Organization

```
PyTorch/
├── 01_Basics/                    # Fundamental concepts
├── 02_Image_Classification/      # Computer vision intro
├── 03_Advanced_CNN/              # Modern architectures
├── 04_Regularization/            # Preventing overfitting
├── 05_Transfer_Learning/         # Pre-trained models
├── 06_Image_Segmentation/        # Pixel-level prediction
├── 07_Time_Series/               # Temporal sequences
├── 08_NLP_Text/                  # Language understanding
└── 09_Advanced_Topics/           # (Placeholder for future)
```

## Contributing

To add new notebooks or improve existing ones:
1. Follow the existing structure and naming conventions
2. Include markdown explanations for each section
3. Use PyTorch best practices
4. Add comments explaining PyTorch-specific concepts
5. Include visualization of results

## License

These notebooks are educational materials converted from the original Keras versions.

## Version Info

- **PyTorch**: 1.9+
- **Python**: 3.7+
- **Last Updated**: December 2024
- **Total Notebooks**: 18
- **Total Estimated Training Time**: 3-4 hours (CPU)

## Quick Checklist for Each Notebook

- [ ] Read through markdown explanations
- [ ] Execute code cells in order
- [ ] Understand the architecture diagram
- [ ] Review the training loop
- [ ] Analyze the results/visualizations
- [ ] Modify hyperparameters and re-run
- [ ] Understand PyTorch-specific parts

Enjoy learning with PyTorch! 🔥
