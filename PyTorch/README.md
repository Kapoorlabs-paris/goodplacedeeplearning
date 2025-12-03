# PyTorch Deep Learning Course

A comprehensive deep learning course using PyTorch, organized from beginner to advanced topics. This is a direct translation of the Keras course, offering the same content and structure but with PyTorch implementations.

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
- **00_densenet_architecture.ipynb**: DenseNet architecture and implementation with transfer learning

### 04 - Regularization Techniques
- **00_cifar10_regularization.ipynb**: Regularization techniques (L2, Dropout, Batch Norm) applied to CIFAR-10
- **01_imdb_overfit_underfit.ipynb**: Understanding overfitting and underfitting with IMDB data

### 05 - Transfer Learning
- **00_imagenet_transfer_learning.ipynb**: Transfer learning using ImageNet pretrained models (ResNet50)
- **01_visualize_heat_maps.ipynb**: Visualizing attention maps and model interpretability (Grad-CAM)

### 06 - Image Segmentation
- **00_cell_tissue_segmentation.ipynb**: Semantic segmentation of cells and tissues with U-Net
- **01_mitosis_detection_brightfield.ipynb**: Detecting mitosis in HeLa cells (brightfield)
- **02_mitosis_detection_phase_contrast.ipynb**: Detecting mitosis with phase contrast microscopy
- **03_mitosis_xenopus_detection.ipynb**: Mitosis detection in Xenopus cells

### 07 - Time Series
- **00_time_series_training.ipynb**: Training LSTM models for time series data
- **01_time_series_prediction.ipynb**: Predicting time series responses to stimuli

### 08 - NLP & Text Processing
- **00_text_classification_welcome.ipynb**: Introduction to text classification with embeddings
- **01_text_classification_deployment.ipynb**: Deploying text classification models
- **02_imdb_reviews_classification.ipynb**: Sentiment analysis on IMDB reviews

### 09 - Advanced Topics
Reserved for advanced research topics and specialized applications.

## Learning Path

### For Beginners
1. Start with **01_Basics** - Learn fundamental concepts with PyTorch
2. Move to **02_Image_Classification** - Classic image tasks
3. Try **03_Advanced_CNN** - More sophisticated architectures

### For Intermediate Learners
1. Explore **04_Regularization** - Prevent overfitting
2. Learn **05_Transfer_Learning** - Leverage pretrained models
3. Study **06_Image_Segmentation** - Pixel-level predictions

### For Advanced Learners
1. **07_Time_Series** - Sequential data processing with LSTMs
2. **08_NLP_Text** - Natural language understanding

## Prerequisites

- Python 3.7+
- PyTorch >= 1.9.0
- NumPy, Pandas, Matplotlib
- Jupyter Notebook
- TorchVision (for pretrained models)

## Setup

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib jupyter scikit-learn pillow h5py tqdm
```

Or use the main repository requirements:
```bash
cd ..
pip install -r requirements.txt
```

## PyTorch Specific Features

### Custom Training Loops
Unlike Keras's `model.fit()`, PyTorch notebooks use explicit training loops for transparency:

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)
```

### Model Definition
All models are defined as `nn.Module` subclasses for clarity and flexibility:

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # ... more layers
        )
        self.classifier = nn.Linear(32 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### Data Loading
Efficient batch processing with PyTorch's DataLoader:

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_idx, (inputs, labels) in enumerate(train_loader):
    # Process batch
```

## Device Management

All notebooks automatically handle GPU/CPU:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

## Key Differences from Keras Version

| Aspect | Keras | PyTorch |
|--------|-------|---------|
| Model Definition | Sequential API | nn.Module classes |
| Training | model.fit() | Custom training loops |
| Data Loading | tf.data.Dataset | torch.utils.data.DataLoader |
| Layers | Functional API | torch.nn layers |
| Activation | Built-in layer parameters | Separate nn.ReLU, nn.Sigmoid |
| Device Handling | Automatic | Explicit .to(device) |
| Visualization | TensorBoard | Matplotlib/TensorBoard |

## Topics Covered

- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs) with LSTMs
- Transfer Learning and fine-tuning
- Regularization and dropout
- Batch normalization
- Image segmentation with U-Net
- Text classification with embeddings
- Custom training loops and optimization

## Recommended Order

1. **Basics** (01) → Foundation
2. **Classification** (02) → Core skills
3. **CNN Architectures** (03) → Deeper understanding
4. **Regularization** (04) → Improve models
5. **Transfer Learning** (05) → Efficient learning
6. **Segmentation** (06) → Advanced vision
7. **Time Series** (07) → Sequential data
8. **NLP** (08) → Language understanding

## Comparison with Keras Course

This PyTorch version covers the **same topics** as the Keras course with **identical learning objectives**. The main differences are:

1. **Implementation Language**: PyTorch instead of TensorFlow/Keras
2. **Training Approach**: Explicit loops instead of `model.fit()`
3. **Flexibility**: More control over training process
4. **Performance**: Comparable or better performance depending on hardware

## Notes

- Each notebook is self-contained and can be run independently
- All notebooks include explanations, visualizations, and exercises
- Data is either downloaded automatically or provided in the repo
- GPU support is optional but recommended for faster training

## PyTorch Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [PyTorch Community Forums](https://discuss.pytorch.org/)

## Common PyTorch Patterns

### Training/Evaluation Mode
```python
model.train()   # Enable dropout, batch norm updates
model.eval()    # Disable dropout, use running stats
```

### Gradient Management
```python
optimizer.zero_grad()  # Reset gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights
```

### Device Compatibility
```python
# Always ensure tensors and model are on same device
model = model.to(device)
input_tensor = input_tensor.to(device)
```

## Troubleshooting

### GPU Not Available
```python
device = torch.device('cpu')  # Falls back automatically
# Or specify explicitly:
device = torch.device('cuda:0')  # First GPU
```

### Memory Issues
- Reduce batch size
- Use gradient checkpointing for large models
- Enable mixed precision training

### Model Convergence
- Check learning rate
- Verify data normalization
- Ensure proper weight initialization

---

Happy Learning with PyTorch! 🚀
