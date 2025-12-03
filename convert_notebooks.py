#!/usr/bin/env python3
"""
Convert Keras notebooks to PyTorch notebooks.
This script reads Keras notebooks and creates PyTorch equivalents with proper conversions.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Mapping of Keras code patterns to PyTorch equivalents
CONVERSION_PATTERNS = {
    # Imports
    "import keras": "import torch",
    "from keras": "from torch",
    "tensorflow": "torch",
    "import tensorflow as tf": "import torch",
    "from tensorflow": "# PyTorch equivalent",

    # Dataset imports
    "keras.datasets.fashion_mnist": "torch",  # handled separately
    "keras.datasets.cifar10": "torchvision.datasets",  # handled separately

    # Model definition
    "keras.Sequential": "nn.Sequential",
    "keras.layers": "nn",
    "keras.Input": "# Input shape defined in first layer",

    # Common layers
    "layers.Flatten": "nn.Flatten",
    "layers.Dense": "nn.Linear",
    "layers.Conv2D": "nn.Conv2d",
    "layers.MaxPooling2D": "nn.MaxPool2d",
    "layers.Dropout": "nn.Dropout",
    "layers.BatchNormalization": "nn.BatchNorm2d",
    "layers.Dense": "nn.Linear",

    # Optimizers
    "keras.optimizers.Adam": "torch.optim.Adam",
    "keras.optimizers.SGD": "torch.optim.SGD",

    # Loss functions
    "'sparse_categorical_crossentropy'": "nn.CrossEntropyLoss()",
    "'categorical_crossentropy'": "nn.CrossEntropyLoss()",
    "'mse'": "nn.MSELoss()",
    "'binary_crossentropy'": "nn.BCELoss()",

    # Model methods
    ".compile": "# Configure training (loss, optimizer, metrics)",
    ".fit": "# Custom training loop",
    ".evaluate": "# Evaluation loop",
    ".predict": "# Inference",
}


def create_pytorch_fashion_mnist_notebook() -> Dict:
    """Create a PyTorch version of the Fashion MNIST notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Train your first neural network: basic classification"],
            "id": "cell-0"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "This notebook trains a neural network model to classify images of Zalando clothing articles, ",
                "like sneakers and shirts etc. It's okay if you don't understand all the details, we will explain more ",
                "and you can always ask!\n\n",
                "This guide uses PyTorch, a deep learning framework that provides flexible and efficient tools for ",
                "building and training models."
            ],
            "id": "cell-1"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "from torch.utils.data import DataLoader, TensorDataset\n",
                "import torchvision.datasets as datasets\n",
                "import torchvision.transforms as transforms\n",
                "\n",
                "# Helper libraries\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "print(torch.__version__)"
            ],
            "id": "cell-2"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Import the Fashion MNIST dataset"],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "This guide uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. ",
                "The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here:\n\n",
                "<table>\n",
                "  <tr><td>\n",
                "    <img src=\"https://tensorflow.org/images/fashion-mnist-sprite.png\"\n",
                "         alt=\"Fashion MNIST sprite\"  width=\"600\">\n",
                "  </td></tr>\n",
                "  <tr><td align=\"center\">\n",
                "    <b>Figure 1.</b> <a href=\"https://github.com/zalandoresearch/fashion-mnist\">Fashion-MNIST samples</a> ",
                "(by Zalando, MIT License).<br/>&nbsp;\n",
                "  </td></tr>\n",
                "</table>\n\n",
                "Fashion MNIST is intended as a drop-in replacement for the classic MNIST dataset. ",
                "We will use 60,000 images to train the network and 10,000 images to evaluate accuracy."
            ],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define transforms\n",
                "transform = transforms.Compose([\n",
                "    transforms.ToTensor(),  # Converts to tensor and normalizes to [0, 1]\n",
                "])\n",
                "\n",
                "# Load Fashion MNIST dataset\n",
                "train_dataset = datasets.FashionMNIST(root='./data', train=True, \n",
                "                                       download=True, transform=transform)\n",
                "test_dataset = datasets.FashionMNIST(root='./data', train=False, \n",
                "                                      download=True, transform=transform)\n",
                "\n",
                "train_images = train_dataset.data.float() / 255.0\n",
                "train_labels = train_dataset.targets\n",
                "test_images = test_dataset.data.float() / 255.0\n",
                "test_labels = test_dataset.targets"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Loading the dataset returns tensors:\n\n",
                "* The `train_images` and `train_labels` tensors are the training set—the data the model uses to learn.\n",
                "* The model is tested against the test set, the `test_images`, and `test_labels` tensors.\n\n",
                "The images are 28x28 tensors, with pixel values normalized to [0, 1]. ",
                "The labels are integers from 0 to 9 corresponding to clothing classes."
            ],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', \n",
                "               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Explore the data\n\nLet's explore the format of the dataset before training the model."],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(f'Training set shape: {train_images.shape}')\n",
                "print(f'Training labels shape: {train_labels.shape}')\n",
                "print(f'Test set shape: {test_images.shape}')\n",
                "print(f'Test labels shape: {test_labels.shape}')"
            ],
            "id": "cell-9"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Preprocess the data\n\nThe data must be preprocessed before training the network."],
            "id": "cell-10"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize the first image\n",
                "plt.figure()\n",
                "plt.imshow(train_images[0], cmap=plt.cm.binary)\n",
                "plt.colorbar()\n",
                "plt.grid(False)\n",
                "plt.show()"
            ],
            "id": "cell-11"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Display the first 25 images from the training set and verify the data is in the correct format."
            ],
            "id": "cell-12"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10,10))\n",
                "for i in range(25):\n",
                "    plt.subplot(5,5,i+1)\n",
                "    plt.xticks([])\n",
                "    plt.yticks([])\n",
                "    plt.grid(False)\n",
                "    plt.imshow(train_images[i], cmap=plt.cm.binary)\n",
                "    plt.xlabel(class_names[train_labels[i]])\n",
                "plt.show()"
            ],
            "id": "cell-13"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Build the model\n\nBuilding the neural network requires configuring the layers of the model."],
            "id": "cell-14"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Setup the layers\n\n",
                "Most of deep learning consists of chaining together simple layers. ",
                "Most layers, like nn.Linear, have parameters that are learned during training."
            ],
            "id": "cell-15"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class FashionMNISTNet(nn.Module):\n",
                "    def __init__(self):\n",
                "        super(FashionMNISTNet, self).__init__()\n",
                "        self.flatten = nn.Flatten()\n",
                "        self.dense1 = nn.Linear(28 * 28, 128)\n",
                "        self.relu = nn.ReLU()\n",
                "        self.dense2 = nn.Linear(128, 10)\n",
                "    \n",
                "    def forward(self, x):\n",
                "        x = self.flatten(x)\n",
                "        x = self.dense1(x)\n",
                "        x = self.relu(x)\n",
                "        x = self.dense2(x)\n",
                "        return x\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = FashionMNISTNet().to(device)\n",
                "\n",
                "print(model)"
            ],
            "id": "cell-16"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "The first layer flattens the 28x28 images into a 784-dimensional vector. ",
                "The network then has a 128-neuron hidden layer with ReLU activation, ",
                "followed by a 10-neuron output layer for the 10 clothing classes."
            ],
            "id": "cell-17"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define loss function and optimizer\n",
                "loss_fn = nn.CrossEntropyLoss()\n",
                "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
                "\n",
                "# Create DataLoaders\n",
                "train_dataset_torch = TensorDataset(train_images, train_labels)\n",
                "test_dataset_torch = TensorDataset(test_images, test_labels)\n",
                "\n",
                "train_loader = DataLoader(train_dataset_torch, batch_size=32, shuffle=True)\n",
                "test_loader = DataLoader(test_dataset_torch, batch_size=32, shuffle=False)"
            ],
            "id": "cell-18"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Train the model\n\nWe'll now train the model using a custom training loop."],
            "id": "cell-19"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def train_epoch(model, train_loader, loss_fn, optimizer, device):\n",
                "    model.train()\n",
                "    total_loss = 0\n",
                "    correct = 0\n",
                "    total = 0\n",
                "    \n",
                "    for images, labels in train_loader:\n",
                "        images, labels = images.to(device), labels.to(device)\n",
                "        \n",
                "        # Forward pass\n",
                "        outputs = model(images)\n",
                "        loss = loss_fn(outputs, labels)\n",
                "        \n",
                "        # Backward pass\n",
                "        optimizer.zero_grad()\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        # Metrics\n",
                "        total_loss += loss.item()\n",
                "        _, predicted = torch.max(outputs.data, 1)\n",
                "        total += labels.size(0)\n",
                "        correct += (predicted == labels).sum().item()\n",
                "    \n",
                "    avg_loss = total_loss / len(train_loader)\n",
                "    accuracy = 100 * correct / total\n",
                "    return avg_loss, accuracy\n",
                "\n",
                "# Train for 5 epochs\n",
                "num_epochs = 5\n",
                "train_losses = []\n",
                "train_accuracies = []\n",
                "\n",
                "for epoch in range(num_epochs):\n",
                "    loss, accuracy = train_epoch(model, train_loader, loss_fn, optimizer, device)\n",
                "    train_losses.append(loss)\n",
                "    train_accuracies.append(accuracy)\n",
                "    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')"
            ],
            "id": "cell-20"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Evaluate accuracy\n\nNext, evaluate the model performance on the test dataset."],
            "id": "cell-21"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluate(model, test_loader, device):\n",
                "    model.eval()\n",
                "    correct = 0\n",
                "    total = 0\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        for images, labels in test_loader:\n",
                "            images, labels = images.to(device), labels.to(device)\n",
                "            outputs = model(images)\n",
                "            _, predicted = torch.max(outputs.data, 1)\n",
                "            total += labels.size(0)\n",
                "            correct += (predicted == labels).sum().item()\n",
                "    \n",
                "    accuracy = 100 * correct / total\n",
                "    return accuracy\n",
                "\n",
                "test_accuracy = evaluate(model, test_loader, device)\n",
                "print(f'Test accuracy: {test_accuracy:.2f}%')"
            ],
            "id": "cell-22"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "The accuracy on the test dataset is slightly less than on the training dataset. ",
                "This gap is an example of overfitting—when a model performs worse on new data than on training data."
            ],
            "id": "cell-23"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Make predictions\n\nWith the model trained, we can use it to make predictions about images."],
            "id": "cell-24"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def make_predictions(model, images, device):\n",
                "    model.eval()\n",
                "    with torch.no_grad():\n",
                "        outputs = model(images.to(device))\n",
                "        probabilities = torch.softmax(outputs, dim=1)\n",
                "    return probabilities.cpu().numpy()\n",
                "\n",
                "predictions = make_predictions(model, test_images, device)"
            ],
            "id": "cell-25"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "A prediction is an array of 10 numbers representing the model's confidence ",
                "for each of the 10 clothing classes."
            ],
            "id": "cell-26"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('First prediction:')\n",
                "print(predictions[0])\n",
                "print(f'\\nPredicted class: {np.argmax(predictions[0])}')\n",
                "print(f'True class: {test_labels[0].item()}')"
            ],
            "id": "cell-27"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def plot_image(i, predictions_array, true_label, img):\n",
                "    predictions_array = predictions_array[i]\n",
                "    true_label = true_label[i]\n",
                "    img = img[i]\n",
                "    \n",
                "    plt.grid(False)\n",
                "    plt.xticks([])\n",
                "    plt.yticks([])\n",
                "    \n",
                "    plt.imshow(img, cmap=plt.cm.binary)\n",
                "    \n",
                "    predicted_label = np.argmax(predictions_array)\n",
                "    if predicted_label == true_label:\n",
                "        color = 'blue'\n",
                "    else:\n",
                "        color = 'red'\n",
                "    \n",
                "    plt.xlabel(\"{} {:2.0f}% ({})\".format(class_names[predicted_label],\n",
                "                                    100*np.max(predictions_array),\n",
                "                                    class_names[true_label]),\n",
                "                                    color=color)\n",
                "\n",
                "def plot_value_array(i, predictions_array, true_label):\n",
                "    predictions_array = predictions_array[i]\n",
                "    true_label = true_label[i]\n",
                "    \n",
                "    plt.grid(False)\n",
                "    plt.xticks([])\n",
                "    plt.yticks([])\n",
                "    thisplot = plt.bar(range(10), predictions_array, color=\"#777777\")\n",
                "    plt.ylim([0, 1])\n",
                "    \n",
                "    predicted_label = np.argmax(predictions_array)\n",
                "    thisplot[predicted_label].set_color('red')\n",
                "    thisplot[true_label].set_color('blue')"
            ],
            "id": "cell-28"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "i = 0\n",
                "plt.figure(figsize=(6,3))\n",
                "plt.subplot(1,2,1)\n",
                "plot_image(i, predictions, test_labels, test_images)\n",
                "plt.subplot(1,2,2)\n",
                "plot_value_array(i, predictions, test_labels)\n",
                "plt.show()"
            ],
            "id": "cell-29"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot the first X test images with their predictions\n",
                "num_rows = 5\n",
                "num_cols = 3\n",
                "num_images = num_rows * num_cols\n",
                "plt.figure(figsize=(2*2*num_cols, 2*num_rows))\n",
                "for i in range(num_images):\n",
                "    plt.subplot(num_rows, 2*num_cols, 2*i+1)\n",
                "    plot_image(i, predictions, test_labels, test_images)\n",
                "    plt.subplot(num_rows, 2*num_cols, 2*i+2)\n",
                "    plot_value_array(i, predictions, test_labels)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
            "id": "cell-30"
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook


def create_pytorch_sine_regression_notebook() -> Dict:
    """Create a PyTorch version of the sine regression notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Learn Sine Function with Neural Network Regression"],
            "id": "cell-0"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "from torch.utils.data import TensorDataset, DataLoader\n",
                "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt"
            ],
            "id": "cell-1"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Generate Data\n\nCreate training data by sampling from a sine function."],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate x and y data\n",
                "xpts = np.arange(0, 1000, 0.1)\n",
                "ypts = (np.sin(xpts) + 1) / 2  # Normalize to [0, 1]\n",
                "\n",
                "# Convert to PyTorch tensors\n",
                "X = torch.from_numpy(xpts).float().unsqueeze(1)  # Shape: (n_samples, 1)\n",
                "y = torch.from_numpy(ypts).float().unsqueeze(1)  # Shape: (n_samples, 1)\n",
                "\n",
                "print(f'X shape: {X.shape}')\n",
                "print(f'y shape: {y.shape}')"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Define Model\n\nCreate a neural network to learn the sine function."],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class SineRegressionNet(nn.Module):\n",
                "    def __init__(self):\n",
                "        super(SineRegressionNet, self).__init__()\n",
                "        self.fc1 = nn.Linear(1, 64)\n",
                "        self.fc2 = nn.Linear(64, 32)\n",
                "        self.fc3 = nn.Linear(32, 1)\n",
                "        self.relu = nn.ReLU()\n",
                "    \n",
                "    def forward(self, x):\n",
                "        x = self.fc1(x)\n",
                "        x = self.relu(x)\n",
                "        x = self.fc2(x)\n",
                "        x = self.relu(x)\n",
                "        x = self.fc3(x)\n",
                "        return x\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = SineRegressionNet().to(device)\n",
                "\n",
                "# Print model summary\n",
                "print(model)\n",
                "print(f'\\nTotal parameters: {sum(p.numel() for p in model.parameters())}')"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Setup Training\n\nDefine loss function and optimizer."],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Loss and optimizer\n",
                "loss_fn = nn.MSELoss()\n",
                "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
                "\n",
                "# Create DataLoader\n",
                "batch_size = 16\n",
                "epochs = 200\n",
                "\n",
                "# Split data into train/validation\n",
                "train_size = int(0.9 * len(X))\n",
                "train_indices = torch.randperm(len(X))[:train_size]\n",
                "val_indices = torch.randperm(len(X))[train_size:]\n",
                "\n",
                "X_train, y_train = X[train_indices], y[train_indices]\n",
                "X_val, y_val = X[val_indices], y[val_indices]\n",
                "\n",
                "train_dataset = TensorDataset(X_train, y_train)\n",
                "val_dataset = TensorDataset(X_val, y_val)\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
                "val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Train Model\n\nTrain the model for the specified number of epochs."],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_losses = []\n",
                "val_losses = []\n",
                "\n",
                "for epoch in range(epochs):\n",
                "    # Training\n",
                "    model.train()\n",
                "    train_loss = 0\n",
                "    for X_batch, y_batch in train_loader:\n",
                "        X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
                "        \n",
                "        # Forward pass\n",
                "        y_pred = model(X_batch)\n",
                "        loss = loss_fn(y_pred, y_batch)\n",
                "        \n",
                "        # Backward pass\n",
                "        optimizer.zero_grad()\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        train_loss += loss.item()\n",
                "    \n",
                "    train_loss /= len(train_loader)\n",
                "    train_losses.append(train_loss)\n",
                "    \n",
                "    # Validation\n",
                "    model.eval()\n",
                "    val_loss = 0\n",
                "    with torch.no_grad():\n",
                "        for X_batch, y_batch in val_loader:\n",
                "            X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
                "            y_pred = model(X_batch)\n",
                "            loss = loss_fn(y_pred, y_batch)\n",
                "            val_loss += loss.item()\n",
                "    \n",
                "    val_loss /= len(val_loader)\n",
                "    val_losses.append(val_loss)\n",
                "    \n",
                "    if (epoch + 1) % 50 == 0:\n",
                "        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')"
            ],
            "id": "cell-9"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Plot Training History"],
            "id": "cell-10"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 5))\n",
                "plt.plot(train_losses, label='Training Loss')\n",
                "plt.plot(val_losses, label='Validation Loss')\n",
                "plt.title('Model Loss')\n",
                "plt.ylabel('Loss')\n",
                "plt.xlabel('Epoch')\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ],
            "id": "cell-11"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Test Predictions\n\nCompare model predictions with ground truth on unseen data."],
            "id": "cell-12"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate test data\n",
                "test_x = np.arange(0, 100, 0.1)\n",
                "test_X = torch.from_numpy(test_x).float().unsqueeze(1).to(device)\n",
                "\n",
                "# Make predictions\n",
                "model.eval()\n",
                "with torch.no_grad():\n",
                "    test_y_pred = model(test_X).cpu().numpy().flatten()\n",
                "\n",
                "# Ground truth\n",
                "test_y_true = (np.sin(test_x) + 1) / 2\n",
                "\n",
                "# Plot\n",
                "plt.figure(figsize=(12, 5))\n",
                "plt.plot(test_x, test_y_pred, label='Predicted', linewidth=2)\n",
                "plt.plot(test_x, test_y_true, label='True Sine', linewidth=2)\n",
                "plt.xlabel('x')\n",
                "plt.ylabel('y')\n",
                "plt.title('Model Predictions vs Ground Truth')\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ],
            "id": "cell-13"
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook


def create_pytorch_boston_house_notebook() -> Dict:
    """Create a PyTorch version of the Boston house price notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Boston House Price Regression"],
            "id": "cell-0"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "from torch.utils.data import TensorDataset, DataLoader\n",
                "from sklearn.datasets import load_boston\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.model_selection import train_test_split\n",
                "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt"
            ],
            "id": "cell-1"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Load Data\n\nLoad the Boston house price dataset."],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load Boston dataset (or use alternative if deprecated)\n",
                "try:\n",
                "    boston = load_boston()\n",
                "    X = boston.data\n",
                "    y = boston.target\n",
                "except:\n",
                "    # Alternative: use pandas\n",
                "    import pandas as pd\n",
                "    data_url = \"http://lib.stat.cmu.edu/datasets/boston\"\n",
                "    raw_df = pd.read_csv(data_url, sep=\"\\s+\", skiprows=22, header=None)\n",
                "    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])\n",
                "    y = raw_df.values[1::2, 2]\n",
                "\n",
                "print(f'Feature shape: {X.shape}')\n",
                "print(f'Target shape: {y.shape}')"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Preprocess Data"],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Normalize features\n",
                "scaler = StandardScaler()\n",
                "X_scaled = scaler.fit_transform(X)\n",
                "\n",
                "# Convert to tensors\n",
                "X_tensor = torch.from_numpy(X_scaled).float()\n",
                "y_tensor = torch.from_numpy(y).float().unsqueeze(1)\n",
                "\n",
                "# Split into train and test sets\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X_tensor, y_tensor, test_size=0.2, random_state=42\n",
                ")\n",
                "\n",
                "print(f'Training set: {X_train.shape}')\n",
                "print(f'Test set: {X_test.shape}')"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Define Model"],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class BostonRegressionNet(nn.Module):\n",
                "    def __init__(self, input_size=13):\n",
                "        super(BostonRegressionNet, self).__init__()\n",
                "        self.fc1 = nn.Linear(input_size, 64)\n",
                "        self.fc2 = nn.Linear(64, 32)\n",
                "        self.fc3 = nn.Linear(32, 1)\n",
                "        self.relu = nn.ReLU()\n",
                "    \n",
                "    def forward(self, x):\n",
                "        x = self.fc1(x)\n",
                "        x = self.relu(x)\n",
                "        x = self.fc2(x)\n",
                "        x = self.relu(x)\n",
                "        x = self.fc3(x)\n",
                "        return x\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = BostonRegressionNet(input_size=X_train.shape[1]).to(device)\n",
                "\n",
                "print(model)"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Setup Training"],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "loss_fn = nn.MSELoss()\n",
                "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
                "\n",
                "batch_size = 32\n",
                "epochs = 100\n",
                "\n",
                "train_dataset = TensorDataset(X_train, y_train)\n",
                "test_dataset = TensorDataset(X_test, y_test)\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
                "test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)"
            ],
            "id": "cell-9"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Train Model"],
            "id": "cell-10"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_losses = []\n",
                "test_losses = []\n",
                "\n",
                "for epoch in range(epochs):\n",
                "    # Training\n",
                "    model.train()\n",
                "    train_loss = 0\n",
                "    for X_batch, y_batch in train_loader:\n",
                "        X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
                "        \n",
                "        y_pred = model(X_batch)\n",
                "        loss = loss_fn(y_pred, y_batch)\n",
                "        \n",
                "        optimizer.zero_grad()\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        train_loss += loss.item()\n",
                "    \n",
                "    train_loss /= len(train_loader)\n",
                "    train_losses.append(train_loss)\n",
                "    \n",
                "    # Testing\n",
                "    model.eval()\n",
                "    test_loss = 0\n",
                "    with torch.no_grad():\n",
                "        for X_batch, y_batch in test_loader:\n",
                "            X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
                "            y_pred = model(X_batch)\n",
                "            loss = loss_fn(y_pred, y_batch)\n",
                "            test_loss += loss.item()\n",
                "    \n",
                "    test_loss /= len(test_loader)\n",
                "    test_losses.append(test_loss)\n",
                "    \n",
                "    if (epoch + 1) % 20 == 0:\n",
                "        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')"
            ],
            "id": "cell-11"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Evaluate Results"],
            "id": "cell-12"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
                "\n",
                "# Loss history\n",
                "ax1.plot(train_losses, label='Training Loss')\n",
                "ax1.plot(test_losses, label='Test Loss')\n",
                "ax1.set_title('Model Loss')\n",
                "ax1.set_ylabel('Loss')\n",
                "ax1.set_xlabel('Epoch')\n",
                "ax1.legend()\n",
                "ax1.grid(True)\n",
                "\n",
                "# Final test predictions vs actual\n",
                "model.eval()\n",
                "with torch.no_grad():\n",
                "    y_pred_all = model(X_test.to(device)).cpu().numpy()\n",
                "\n",
                "ax2.scatter(y_test.numpy(), y_pred_all, alpha=0.5)\n",
                "ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
                "ax2.set_xlabel('Actual Price')\n",
                "ax2.set_ylabel('Predicted Price')\n",
                "ax2.set_title('Test Set Predictions')\n",
                "ax2.grid(True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Calculate metrics\n",
                "mse = np.mean((y_test.numpy() - y_pred_all) ** 2)\n",
                "rmse = np.sqrt(mse)\n",
                "mae = np.mean(np.abs(y_test.numpy() - y_pred_all))\n",
                "\n",
                "print(f'\\nFinal Test Results:')\n",
                "print(f'MSE: {mse:.4f}')\n",
                "print(f'RMSE: {rmse:.4f}')\n",
                "print(f'MAE: {mae:.4f}')"
            ],
            "id": "cell-13"
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook


def create_pytorch_cifar10_notebook() -> Dict:
    """Create a PyTorch version of the CIFAR-10 classification notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# CIFAR-10 Classification: Cats and Dogs"],
            "id": "cell-0"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "from torch.utils.data import DataLoader, Dataset, TensorDataset\n",
                "import torchvision.datasets as datasets\n",
                "import torchvision.transforms as transforms\n",
                "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt"
            ],
            "id": "cell-1"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Load and Prepare Data"],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load CIFAR-10 dataset\n",
                "transform = transforms.Compose([\n",
                "    transforms.ToTensor(),\n",
                "])\n",
                "\n",
                "train_dataset = datasets.CIFAR10(root='./data', train=True, \n",
                "                                  download=True, transform=transform)\n",
                "test_dataset = datasets.CIFAR10(root='./data', train=False, \n",
                "                                 download=True, transform=transform)\n",
                "\n",
                "# Extract cats (class 3) and dogs (class 5)\n",
                "def extract_cats_dogs(dataset):\n",
                "    images = []\n",
                "    labels = []\n",
                "    for i, (img, label) in enumerate(dataset):\n",
                "        if label in [3, 5]:  # 3=cat, 5=dog\n",
                "            images.append(img)\n",
                "            labels.append(1 if label == 5 else 0)  # 0=cat, 1=dog\n",
                "    return torch.stack(images), torch.tensor(labels, dtype=torch.float32).unsqueeze(1)\n",
                "\n",
                "x_train, y_train = extract_cats_dogs(train_dataset)\n",
                "x_test, y_test = extract_cats_dogs(test_dataset)\n",
                "\n",
                "print(f'Training set: {x_train.shape}, {y_train.shape}')\n",
                "print(f'Test set: {x_test.shape}, {y_test.shape}')"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Visualize Data"],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
                "axes[0].imshow(x_train[0].permute(1, 2, 0))\n",
                "axes[0].set_title('A cat')\n",
                "axes[0].axis('off')\n",
                "\n",
                "# Find a dog\n",
                "dog_idx = (y_train == 1).nonzero()[0].item()\n",
                "axes[1].imshow(x_train[dog_idx].permute(1, 2, 0))\n",
                "axes[1].set_title('A dog')\n",
                "axes[1].axis('off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Define Models"],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class SimpleCNN(nn.Module):\n",
                "    \"\"\"Simple CNN for cat/dog classification\"\"\"\n",
                "    def __init__(self):\n",
                "        super(SimpleCNN, self).__init__()\n",
                "        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)\n",
                "        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)\n",
                "        self.pool = nn.MaxPool2d(2, 2)\n",
                "        self.dropout1 = nn.Dropout(0.25)\n",
                "        self.dropout2 = nn.Dropout(0.5)\n",
                "        self.fc1 = nn.Linear(64 * 8 * 8, 256)\n",
                "        self.fc2 = nn.Linear(256, 128)\n",
                "        self.fc3 = nn.Linear(128, 1)\n",
                "        self.relu = nn.ReLU()\n",
                "        self.sigmoid = nn.Sigmoid()\n",
                "    \n",
                "    def forward(self, x):\n",
                "        x = self.relu(self.conv1(x))\n",
                "        x = self.relu(self.conv2(x))\n",
                "        x = self.pool(x)\n",
                "        x = self.dropout1(x)\n",
                "        x = x.view(x.size(0), -1)\n",
                "        x = self.relu(self.fc1(x))\n",
                "        x = self.dropout2(x)\n",
                "        x = self.relu(self.fc2(x))\n",
                "        x = self.sigmoid(self.fc3(x))\n",
                "        return x\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = SimpleCNN().to(device)\n",
                "print(model)"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Train Model"],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "batch_size = 128\n",
                "epochs = 10\n",
                "\n",
                "loss_fn = nn.BCELoss()\n",
                "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
                "\n",
                "train_dataset = TensorDataset(x_train, y_train)\n",
                "test_dataset = TensorDataset(x_test, y_test)\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
                "test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n",
                "\n",
                "def train_epoch(model, train_loader, loss_fn, optimizer, device):\n",
                "    model.train()\n",
                "    train_loss = 0\n",
                "    correct = 0\n",
                "    total = 0\n",
                "    \n",
                "    for images, labels in train_loader:\n",
                "        images, labels = images.to(device), labels.to(device)\n",
                "        \n",
                "        outputs = model(images)\n",
                "        loss = loss_fn(outputs, labels)\n",
                "        \n",
                "        optimizer.zero_grad()\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        train_loss += loss.item()\n",
                "        predicted = (outputs > 0.5).float().squeeze()\n",
                "        total += labels.size(0)\n",
                "        correct += (predicted == labels.squeeze()).sum().item()\n",
                "    \n",
                "    return train_loss / len(train_loader), 100 * correct / total\n",
                "\n",
                "def evaluate(model, test_loader, loss_fn, device):\n",
                "    model.eval()\n",
                "    test_loss = 0\n",
                "    correct = 0\n",
                "    total = 0\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        for images, labels in test_loader:\n",
                "            images, labels = images.to(device), labels.to(device)\n",
                "            outputs = model(images)\n",
                "            loss = loss_fn(outputs, labels)\n",
                "            test_loss += loss.item()\n",
                "            predicted = (outputs > 0.5).float().squeeze()\n",
                "            total += labels.size(0)\n",
                "            correct += (predicted == labels.squeeze()).sum().item()\n",
                "    \n",
                "    return test_loss / len(test_loader), 100 * correct / total\n",
                "\n",
                "train_losses = []\n",
                "train_accs = []\n",
                "val_losses = []\n",
                "val_accs = []\n",
                "\n",
                "for epoch in range(epochs):\n",
                "    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)\n",
                "    val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)\n",
                "    \n",
                "    train_losses.append(train_loss)\n",
                "    train_accs.append(train_acc)\n",
                "    val_losses.append(val_loss)\n",
                "    val_accs.append(val_acc)\n",
                "    \n",
                "    print(f'Epoch {epoch+1}/{epochs}')\n",
                "    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')\n",
                "    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')"
            ],
            "id": "cell-9"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Analyze Results"],
            "id": "cell-10"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
                "\n",
                "axes[0].plot(train_losses, label='Training Loss')\n",
                "axes[0].plot(val_losses, label='Validation Loss')\n",
                "axes[0].set_title('Model Loss')\n",
                "axes[0].set_ylabel('Loss')\n",
                "axes[0].set_xlabel('Epoch')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True)\n",
                "\n",
                "axes[1].plot(train_accs, label='Training Accuracy')\n",
                "axes[1].plot(val_accs, label='Validation Accuracy')\n",
                "axes[1].set_title('Model Accuracy')\n",
                "axes[1].set_ylabel('Accuracy (%)')\n",
                "axes[1].set_xlabel('Epoch')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
            "id": "cell-11"
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook


def create_pytorch_densenet_notebook() -> Dict:
    """Create a PyTorch version of the DenseNet architecture notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# DenseNet Architecture in PyTorch"],
            "id": "cell-0"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "import torchvision.models as models\n",
                "from torch.utils.data import DataLoader\n",
                "import torchvision.datasets as datasets\n",
                "import torchvision.transforms as transforms\n",
                "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt"
            ],
            "id": "cell-1"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Introduction to DenseNet\n\nDenseNet (Densely Connected Convolutional Network) is an architecture where each layer is connected to every other layer in a feed-forward fashion."],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load a pre-trained DenseNet121\n",
                "densenet = models.densenet121(pretrained=True)\n",
                "print(densenet)"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Key Components of DenseNet"],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Count parameters\n",
                "def count_parameters(model):\n",
                "    return sum(p.numel() for p in model.parameters())\n",
                "\n",
                "print(f'DenseNet121 has {count_parameters(densenet) / 1e6:.1f}M parameters')"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Custom DenseNet Block"],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class DenseBlock(nn.Module):\n",
                "    \"\"\"A dense block consisting of batch norm, relu, and convolution\"\"\"\n",
                "    def __init__(self, in_channels, growth_rate):\n",
                "        super(DenseBlock, self).__init__()\n",
                "        self.bn = nn.BatchNorm2d(in_channels)\n",
                "        self.relu = nn.ReLU(inplace=True)\n",
                "        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)\n",
                "    \n",
                "    def forward(self, x):\n",
                "        x = self.bn(x)\n",
                "        x = self.relu(x)\n",
                "        x = self.conv(x)\n",
                "        return x\n",
                "\n",
                "class DenseSequential(nn.Module):\n",
                "    \"\"\"Concatenates outputs from multiple dense blocks\"\"\"\n",
                "    def __init__(self, num_blocks, in_channels, growth_rate):\n",
                "        super(DenseSequential, self).__init__()\n",
                "        self.blocks = nn.ModuleList()\n",
                "        for i in range(num_blocks):\n",
                "            self.blocks.append(DenseBlock(in_channels + i * growth_rate, growth_rate))\n",
                "    \n",
                "    def forward(self, x):\n",
                "        features = [x]\n",
                "        for block in self.blocks:\n",
                "            new_feature = block(torch.cat(features, 1))\n",
                "            features.append(new_feature)\n",
                "        return torch.cat(features, 1)\n",
                "\n",
                "# Test the dense block\n",
                "dense_block = DenseSequential(num_blocks=4, in_channels=64, growth_rate=32)\n",
                "x = torch.randn(2, 64, 32, 32)\n",
                "out = dense_block(x)\n",
                "print(f'Input shape: {x.shape}')\n",
                "print(f'Output shape: {out.shape}')"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Transfer Learning with DenseNet"],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load CIFAR-10 dataset\n",
                "transform = transforms.Compose([\n",
                "    transforms.Resize(224),\n",
                "    transforms.ToTensor(),\n",
                "    transforms.Normalize(mean=[0.5, 0.5, 0.5],\n",
                "                       std=[0.5, 0.5, 0.5])\n",
                "])\n",
                "\n",
                "train_dataset = datasets.CIFAR10(root='./data', train=True, \n",
                "                                  download=True, transform=transform)\n",
                "test_dataset = datasets.CIFAR10(root='./data', train=False, \n",
                "                                 download=True, transform=transform)\n",
                "\n",
                "# Use a subset for quick training\n",
                "train_dataset.data = train_dataset.data[:5000]\n",
                "train_dataset.targets = train_dataset.targets[:5000]\n",
                "test_dataset.data = test_dataset.data[:1000]\n",
                "test_dataset.targets = test_dataset.targets[:1000]\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
                "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
            ],
            "id": "cell-9"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Modify the final layer for CIFAR-10 (10 classes)\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = models.densenet121(pretrained=True).to(device)\n",
                "\n",
                "# Replace the final classifier\n",
                "num_ftrs = model.classifier.in_features\n",
                "model.classifier = nn.Linear(num_ftrs, 10)\n",
                "model = model.to(device)\n",
                "\n",
                "print(f'Model on device: {next(model.parameters()).device}')"
            ],
            "id": "cell-10"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train for a few epochs\n",
                "loss_fn = nn.CrossEntropyLoss()\n",
                "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
                "\n",
                "epochs = 3\n",
                "train_losses = []\n",
                "val_accs = []\n",
                "\n",
                "for epoch in range(epochs):\n",
                "    # Training\n",
                "    model.train()\n",
                "    train_loss = 0\n",
                "    for images, labels in train_loader:\n",
                "        images, labels = images.to(device), labels.to(device)\n",
                "        \n",
                "        outputs = model(images)\n",
                "        loss = loss_fn(outputs, labels)\n",
                "        \n",
                "        optimizer.zero_grad()\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        train_loss += loss.item()\n",
                "    \n",
                "    train_loss /= len(train_loader)\n",
                "    train_losses.append(train_loss)\n",
                "    \n",
                "    # Validation\n",
                "    model.eval()\n",
                "    correct = 0\n",
                "    total = 0\n",
                "    with torch.no_grad():\n",
                "        for images, labels in test_loader:\n",
                "            images, labels = images.to(device), labels.to(device)\n",
                "            outputs = model(images)\n",
                "            _, predicted = torch.max(outputs.data, 1)\n",
                "            total += labels.size(0)\n",
                "            correct += (predicted == labels).sum().item()\n",
                "    \n",
                "    val_acc = 100 * correct / total\n",
                "    val_accs.append(val_acc)\n",
                "    \n",
                "    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%')"
            ],
            "id": "cell-11"
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook


def save_notebook(notebook: Dict, path: str):
    """Save a notebook as JSON to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Saved: {path}")


if __name__ == "__main__":
    base_path = "/Users/vkapoor/python_workspace/goodplacedeeplearning/PyTorch"

    # Create notebooks
    print("Creating PyTorch notebooks...")

    # 1. Fashion MNIST
    notebook1 = create_pytorch_fashion_mnist_notebook()
    save_notebook(notebook1, f"{base_path}/01_Basics/00_fashion_mnist_basic_cnn.ipynb")

    # 2. Sine Regression
    notebook2 = create_pytorch_sine_regression_notebook()
    save_notebook(notebook2, f"{base_path}/01_Basics/01_learn_sine_regression.ipynb")

    # 3. Boston House Price
    notebook3 = create_pytorch_boston_house_notebook()
    save_notebook(notebook3, f"{base_path}/01_Basics/02_boston_house_price_regression.ipynb")

    # 4. CIFAR-10 Classification
    notebook4 = create_pytorch_cifar10_notebook()
    save_notebook(notebook4, f"{base_path}/02_Image_Classification/00_cifar10_classification.ipynb")

    # 5. DenseNet Architecture
    notebook5 = create_pytorch_densenet_notebook()
    save_notebook(notebook5, f"{base_path}/03_Advanced_CNN/00_densenet_architecture.ipynb")

    print("\nConversion complete! First 5 priority notebooks created.")
