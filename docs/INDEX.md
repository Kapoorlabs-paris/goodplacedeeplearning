# PyTorch Notebooks - Complete Index

## Quick Navigation

### Getting Started
- **PYTORCH_README.md** - Start here! Complete learning guide with quick start
- **PYTORCH_CONVERSION_SUMMARY.md** - Technical details of conversion
- **CONVERSION_COMPLETE.md** - Project completion summary

### Notebooks by Learning Level

#### Beginner (Start here!)
1. `PyTorch/01_Basics/00_fashion_mnist_basic_cnn.ipynb` - Image classification
2. `PyTorch/01_Basics/01_learn_sine_regression.ipynb` - Regression
3. `PyTorch/02_Image_Classification/00_cifar10_classification.ipynb` - Binary classification

#### Intermediate
4. `PyTorch/01_Basics/02_boston_house_price_regression.ipynb` - Real-world regression
5. `PyTorch/04_Regularization/01_imdb_overfit_underfit.ipynb` - Generalization
6. `PyTorch/03_Advanced_CNN/00_densenet_architecture.ipynb` - Modern architectures

#### Advanced
7. `PyTorch/04_Regularization/00_cifar10_regularization.ipynb` - Regularization
8. `PyTorch/05_Transfer_Learning/00_imagenet_transfer_learning.ipynb` - Transfer learning
9. `PyTorch/05_Transfer_Learning/01_visualize_heat_maps.ipynb` - Interpretability
10. `PyTorch/06_Image_Segmentation/00_cell_tissue_segmentation.ipynb` - Segmentation
11. `PyTorch/07_Time_Series/00_time_series_training.ipynb` - Time series
12. `PyTorch/08_NLP_Text/00_text_classification_welcome.ipynb` - NLP/Text

### Notebooks by Category

#### 01_Basics (3 notebooks)
- Image classification with Fashion MNIST
- Sine function regression
- Boston housing price prediction

#### 02_Image_Classification (1 notebook)
- CIFAR-10 cat vs dog binary classification

#### 03_Advanced_CNN (1 notebook)
- DenseNet architecture and transfer learning

#### 04_Regularization (2 notebooks)
- CIFAR-10 with L2, Dropout, Batch Norm
- Overfitting vs underfitting analysis

#### 05_Transfer_Learning (2 notebooks)
- Fine-tuning ResNet50 on CIFAR-10
- Feature visualization with Grad-CAM

#### 06_Image_Segmentation (4 notebooks)
- U-Net for cell tissue segmentation
- Mitosis detection (brightfield, phase contrast)
- Xenopus embryo segmentation

#### 07_Time_Series (2 notebooks)
- LSTM time series training
- Time series prediction

#### 08_NLP_Text (3 notebooks)
- Text classification introduction
- Production-ready text classifier
- IMDB reviews sentiment analysis

### Conversion Resources

#### Conversion Scripts
- `convert_notebooks.py` - Creates basic notebooks (1-5)
- `convert_remaining_notebooks.py` - Creates intermediate notebooks (6-9)
- `convert_advanced_notebooks.py` - Creates advanced notebooks (10-18)

## File Organization

```
goodplacedeeplearning/
├── PyTorch/                          # Main notebooks directory
│   ├── 01_Basics/                    # 3 notebooks
│   ├── 02_Image_Classification/      # 1 notebook
│   ├── 03_Advanced_CNN/              # 1 notebook
│   ├── 04_Regularization/            # 2 notebooks
│   ├── 05_Transfer_Learning/         # 2 notebooks
│   ├── 06_Image_Segmentation/        # 4 notebooks
│   ├── 07_Time_Series/               # 2 notebooks
│   ├── 08_NLP_Text/                  # 3 notebooks
│   └── 09_Advanced_Topics/           # (Reserved for future)
│
├── Keras/                            # Original Keras notebooks
│
├── Documentation:
│   ├── PYTORCH_README.md             # Learning guide and best practices
│   ├── PYTORCH_CONVERSION_SUMMARY.md # Technical conversion details
│   ├── CONVERSION_COMPLETE.md        # Project summary
│   └── INDEX.md                      # This file
│
├── Conversion Tools:
│   ├── convert_notebooks.py          # Script for basic notebooks
│   ├── convert_remaining_notebooks.py# Script for intermediate notebooks
│   └── convert_advanced_notebooks.py # Script for advanced notebooks
```

## Key Statistics

| Metric | Value |
|--------|-------|
| Total Notebooks | 18 |
| Total Categories | 8 |
| Estimated Learning Time | 8-10 hours |
| Total Size | ~160 KB |
| Valid Notebooks | 18/18 (100%) |
| PyTorch Version | 1.9+ |

## Quick Commands

```bash
# Navigate to repository
cd /Users/vkapoor/python_workspace/goodplacedeeplearning

# Start Jupyter
jupyter notebook

# Open specific notebook
jupyter notebook PyTorch/01_Basics/00_fashion_mnist_basic_cnn.ipynb

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn
```

## Topics Covered

### Deep Learning Concepts
- Neural network architecture design
- Forward and backward propagation
- Loss functions and optimizers
- Training and validation procedures
- Hyperparameter tuning

### Computer Vision
- Image classification (CNN)
- Image segmentation (U-Net)
- Object detection concepts
- Feature extraction
- Transfer learning

### Sequence Modeling
- Recurrent neural networks
- LSTM and GRU cells
- Time series prediction
- Sequence-to-sequence models

### Natural Language Processing
- Word embeddings
- Text classification
- Sentiment analysis
- RNNs for language

### Regularization & Optimization
- Dropout
- Batch normalization
- L2 regularization
- Model capacity and generalization

### Model Interpretability
- Feature visualization
- Gradient-based attention maps (Grad-CAM)
- Model behavior analysis

## Recommended Study Order

### Week 1: Fundamentals
1. PYTORCH_README.md (20 min)
2. 00_fashion_mnist_basic_cnn.ipynb (30 min)
3. 01_learn_sine_regression.ipynb (20 min)
4. 02_boston_house_price_regression.ipynb (20 min)

### Week 2: Computer Vision
5. 00_cifar10_classification.ipynb (30 min)
6. 01_imdb_overfit_underfit.ipynb (20 min)
7. 00_densenet_architecture.ipynb (30 min)
8. 00_cifar10_regularization.ipynb (30 min)

### Week 3: Advanced Topics
9. 00_imagenet_transfer_learning.ipynb (30 min)
10. 01_visualize_heat_maps.ipynb (30 min)
11. 00_cell_tissue_segmentation.ipynb (30 min)

### Week 4: Sequential Data
12. 00_time_series_training.ipynb (30 min)
13. 00_text_classification_welcome.ipynb (30 min)

## Support & Resources

### Official Documentation
- [PyTorch Official Docs](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Within This Repository
- PYTORCH_README.md - Tips, troubleshooting, patterns
- Each notebook - Inline documentation and explanations

### External Resources
- Stanford CS231N (CNNs)
- Stanford CS224N (NLP)
- FastAI Courses
- Kaggle Competitions

## Troubleshooting

### Common Issues & Solutions

**ImportError: No module named 'torch'**
- Solution: `pip install torch torchvision`

**CUDA out of memory**
- Solution: Reduce batch size, use CPU, enable gradient checkpointing

**Model not learning**
- Solution: Check data normalization, verify loss function, increase learning rate

**Slow training**
- Solution: Enable GPU, increase batch size appropriately, profile code

See PYTORCH_README.md for more troubleshooting tips.

## Contributing

To improve or add notebooks:
1. Follow existing naming and structure
2. Include clear markdown explanations
3. Use PyTorch best practices
4. Add visualization of results
5. Document all PyTorch-specific patterns

## License & Attribution

These notebooks are educational conversions of the original Keras notebooks. All content is meant for learning purposes.

---

**Last Updated**: December 3, 2024
**Status**: Complete and Validated
**Version**: 1.0
