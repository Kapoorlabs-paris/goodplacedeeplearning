#!/usr/bin/env python3
"""
Convert remaining Keras notebooks (6-18) to PyTorch.
"""

import json
import os
from typing import Dict


def create_pytorch_regularization_notebook() -> Dict:
    """Create CIFAR-10 Regularization notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# CIFAR-10 Regularization Techniques"],
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
                "from torch.utils.data import DataLoader, TensorDataset\n",
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
            "source": ["## Load CIFAR-10 Dataset"],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Data augmentation for training\n",
                "train_transform = transforms.Compose([\n",
                "    transforms.RandomHorizontalFlip(),\n",
                "    transforms.RandomVerticalFlip(),\n",
                "    transforms.ToTensor(),\n",
                "])\n",
                "\n",
                "test_transform = transforms.Compose([\n",
                "    transforms.ToTensor(),\n",
                "])\n",
                "\n",
                "train_dataset = datasets.CIFAR10(root='./data', train=True, \n",
                "                                  download=True, transform=train_transform)\n",
                "test_dataset = datasets.CIFAR10(root='./data', train=False, \n",
                "                                 download=True, transform=test_transform)\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)\n",
                "test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Regularized CNN Model"],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class RegularizedCNN(nn.Module):\n",
                "    \"\"\"CNN with L2 regularization (weight decay)\"\"\"\n",
                "    def __init__(self, dropout_rate=0.5):\n",
                "        super(RegularizedCNN, self).__init__()\n",
                "        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)\n",
                "        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)\n",
                "        self.pool = nn.MaxPool2d(2, 2)\n",
                "        self.bn1 = nn.BatchNorm2d(64)\n",
                "        self.bn2 = nn.BatchNorm2d(128)\n",
                "        self.dropout1 = nn.Dropout(0.25)\n",
                "        self.dropout2 = nn.Dropout(dropout_rate)\n",
                "        self.fc1 = nn.Linear(128 * 8 * 8, 256)\n",
                "        self.fc2 = nn.Linear(256, 10)\n",
                "        self.relu = nn.ReLU()\n",
                "    \n",
                "    def forward(self, x):\n",
                "        x = self.relu(self.bn1(self.conv1(x)))\n",
                "        x = self.pool(x)\n",
                "        x = self.dropout1(x)\n",
                "        \n",
                "        x = self.relu(self.bn2(self.conv2(x)))\n",
                "        x = self.pool(x)\n",
                "        x = self.dropout1(x)\n",
                "        \n",
                "        x = x.view(x.size(0), -1)\n",
                "        x = self.relu(self.fc1(x))\n",
                "        x = self.dropout2(x)\n",
                "        x = self.fc2(x)\n",
                "        return x\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = RegularizedCNN(dropout_rate=0.5).to(device)\n",
                "print(model)"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Training with Regularization"],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# L2 regularization is applied via weight_decay parameter\n",
                "loss_fn = nn.CrossEntropyLoss()\n",
                "optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)\n",
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
                "        _, predicted = torch.max(outputs.data, 1)\n",
                "        total += labels.size(0)\n",
                "        correct += (predicted == labels).sum().item()\n",
                "    \n",
                "    return train_loss / len(train_loader), 100 * correct / total\n",
                "\n",
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
                "    return 100 * correct / total\n",
                "\n",
                "epochs = 10\n",
                "train_losses = []\n",
                "val_accs = []\n",
                "\n",
                "for epoch in range(epochs):\n",
                "    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)\n",
                "    val_acc = evaluate(model, test_loader, device)\n",
                "    \n",
                "    train_losses.append(train_loss)\n",
                "    val_accs.append(val_acc)\n",
                "    \n",
                "    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Plot Results"],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
                "\n",
                "ax1.plot(train_losses)\n",
                "ax1.set_title('Training Loss')\n",
                "ax1.set_ylabel('Loss')\n",
                "ax1.set_xlabel('Epoch')\n",
                "ax1.grid(True)\n",
                "\n",
                "ax2.plot(val_accs)\n",
                "ax2.set_title('Validation Accuracy')\n",
                "ax2.set_ylabel('Accuracy (%)')\n",
                "ax2.set_xlabel('Epoch')\n",
                "ax2.grid(True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
            "id": "cell-9"
        }
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def create_pytorch_imdb_overfit_notebook() -> Dict:
    """Create IMDB Overfitting/Underfitting notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# IMDB Reviews: Overfitting and Underfitting"],
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
                "from torch.utils.data import DataLoader, TensorDataset\n",
                "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt"
            ],
            "id": "cell-1"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Generate Synthetic Data"],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate synthetic text classification data\n",
                "np.random.seed(42)\n",
                "torch.manual_seed(42)\n",
                "\n",
                "# Create synthetic features (e.g., TF-IDF vectors)\n",
                "n_samples = 1000\n",
                "n_features = 100\n",
                "\n",
                "X = torch.randn(n_samples, n_features)\n",
                "y = (X[:, 0] + X[:, 1] > 0).long()\n",
                "\n",
                "# Split into train/test\n",
                "train_size = int(0.8 * n_samples)\n",
                "X_train, X_test = X[:train_size], X[train_size:]\n",
                "y_train, y_test = y[:train_size], y[train_size:]\n",
                "\n",
                "train_dataset = TensorDataset(X_train, y_train)\n",
                "test_dataset = TensorDataset(X_test, y_test)\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
                "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Models of Different Complexities"],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class SimpleModel(nn.Module):\n",
                "    \"\"\"Underfit model\"\"\"\n",
                "    def __init__(self, input_size=100):\n",
                "        super(SimpleModel, self).__init__()\n",
                "        self.fc = nn.Linear(input_size, 2)\n",
                "    \n",
                "    def forward(self, x):\n",
                "        return self.fc(x)\n",
                "\n",
                "class MediumModel(nn.Module):\n",
                "    \"\"\"Well-fit model\"\"\"\n",
                "    def __init__(self, input_size=100):\n",
                "        super(MediumModel, self).__init__()\n",
                "        self.fc1 = nn.Linear(input_size, 64)\n",
                "        self.fc2 = nn.Linear(64, 32)\n",
                "        self.fc3 = nn.Linear(32, 2)\n",
                "        self.relu = nn.ReLU()\n",
                "    \n",
                "    def forward(self, x):\n",
                "        x = self.relu(self.fc1(x))\n",
                "        x = self.relu(self.fc2(x))\n",
                "        return self.fc3(x)\n",
                "\n",
                "class ComplexModel(nn.Module):\n",
                "    \"\"\"Overfit model\"\"\"\n",
                "    def __init__(self, input_size=100):\n",
                "        super(ComplexModel, self).__init__()\n",
                "        self.fc1 = nn.Linear(input_size, 256)\n",
                "        self.fc2 = nn.Linear(256, 256)\n",
                "        self.fc3 = nn.Linear(256, 256)\n",
                "        self.fc4 = nn.Linear(256, 2)\n",
                "        self.relu = nn.ReLU()\n",
                "    \n",
                "    def forward(self, x):\n",
                "        x = self.relu(self.fc1(x))\n",
                "        x = self.relu(self.fc2(x))\n",
                "        x = self.relu(self.fc3(x))\n",
                "        return self.fc4(x)"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Train and Compare Models"],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "\n",
                "def train_model(model, train_loader, test_loader, epochs=100):\n",
                "    model = model.to(device)\n",
                "    loss_fn = nn.CrossEntropyLoss()\n",
                "    optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
                "    \n",
                "    train_losses = []\n",
                "    test_losses = []\n",
                "    train_accs = []\n",
                "    test_accs = []\n",
                "    \n",
                "    for epoch in range(epochs):\n",
                "        # Train\n",
                "        model.train()\n",
                "        train_loss = 0\n",
                "        train_correct = 0\n",
                "        train_total = 0\n",
                "        \n",
                "        for X_batch, y_batch in train_loader:\n",
                "            X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
                "            \n",
                "            outputs = model(X_batch)\n",
                "            loss = loss_fn(outputs, y_batch)\n",
                "            \n",
                "            optimizer.zero_grad()\n",
                "            loss.backward()\n",
                "            optimizer.step()\n",
                "            \n",
                "            train_loss += loss.item()\n",
                "            _, predicted = torch.max(outputs, 1)\n",
                "            train_total += y_batch.size(0)\n",
                "            train_correct += (predicted == y_batch).sum().item()\n",
                "        \n",
                "        train_loss /= len(train_loader)\n",
                "        train_acc = 100 * train_correct / train_total\n",
                "        train_losses.append(train_loss)\n",
                "        train_accs.append(train_acc)\n",
                "        \n",
                "        # Test\n",
                "        model.eval()\n",
                "        test_loss = 0\n",
                "        test_correct = 0\n",
                "        test_total = 0\n",
                "        \n",
                "        with torch.no_grad():\n",
                "            for X_batch, y_batch in test_loader:\n",
                "                X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
                "                outputs = model(X_batch)\n",
                "                loss = loss_fn(outputs, y_batch)\n",
                "                test_loss += loss.item()\n",
                "                _, predicted = torch.max(outputs, 1)\n",
                "                test_total += y_batch.size(0)\n",
                "                test_correct += (predicted == y_batch).sum().item()\n",
                "        \n",
                "        test_loss /= len(test_loader)\n",
                "        test_acc = 100 * test_correct / test_total\n",
                "        test_losses.append(test_loss)\n",
                "        test_accs.append(test_acc)\n",
                "    \n",
                "    return train_losses, test_losses, train_accs, test_accs\n",
                "\n",
                "# Train all three models\n",
                "simple_results = train_model(SimpleModel(), train_loader, test_loader, epochs=100)\n",
                "medium_results = train_model(MediumModel(), train_loader, test_loader, epochs=100)\n",
                "complex_results = train_model(ComplexModel(), train_loader, test_loader, epochs=100)"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Compare Results"],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
                "\n",
                "epochs_range = range(1, 101)\n",
                "\n",
                "# Loss comparison\n",
                "axes[0, 0].plot(epochs_range, simple_results[0], label='Simple Train')\n",
                "axes[0, 0].plot(epochs_range, simple_results[1], label='Simple Test')\n",
                "axes[0, 0].set_title('Simple Model (Underfitting)')\n",
                "axes[0, 0].set_ylabel('Loss')\n",
                "axes[0, 0].legend()\n",
                "axes[0, 0].grid(True)\n",
                "\n",
                "axes[0, 1].plot(epochs_range, medium_results[0], label='Medium Train')\n",
                "axes[0, 1].plot(epochs_range, medium_results[1], label='Medium Test')\n",
                "axes[0, 1].set_title('Medium Model (Well-fit)')\n",
                "axes[0, 1].set_ylabel('Loss')\n",
                "axes[0, 1].legend()\n",
                "axes[0, 1].grid(True)\n",
                "\n",
                "axes[1, 0].plot(epochs_range, complex_results[0], label='Complex Train')\n",
                "axes[1, 0].plot(epochs_range, complex_results[1], label='Complex Test')\n",
                "axes[1, 0].set_title('Complex Model (Overfitting)')\n",
                "axes[1, 0].set_ylabel('Loss')\n",
                "axes[1, 0].set_xlabel('Epoch')\n",
                "axes[1, 0].legend()\n",
                "axes[1, 0].grid(True)\n",
                "\n",
                "axes[1, 1].plot(epochs_range, simple_results[3], label='Simple Test Acc')\n",
                "axes[1, 1].plot(epochs_range, medium_results[3], label='Medium Test Acc')\n",
                "axes[1, 1].plot(epochs_range, complex_results[3], label='Complex Test Acc')\n",
                "axes[1, 1].set_title('Test Accuracy Comparison')\n",
                "axes[1, 1].set_ylabel('Accuracy (%)')\n",
                "axes[1, 1].set_xlabel('Epoch')\n",
                "axes[1, 1].legend()\n",
                "axes[1, 1].grid(True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
            "id": "cell-9"
        }
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def create_pytorch_transfer_learning_notebook() -> Dict:
    """Create Transfer Learning notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Transfer Learning with ImageNet Pre-trained Models"],
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
            "source": ["## What is Transfer Learning?\n\nTransfer learning leverages pre-trained models trained on large datasets (like ImageNet) to solve related tasks with limited data."],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load a pre-trained ResNet50\n",
                "model_pretrained = models.resnet50(pretrained=True)\n",
                "print(f'ResNet50 loaded with pretrained ImageNet weights')\n",
                "\n",
                "# Count parameters\n",
                "total_params = sum(p.numel() for p in model_pretrained.parameters())\n",
                "trainable_params = sum(p.numel() for p in model_pretrained.parameters() if p.requires_grad)\n",
                "print(f'Total parameters: {total_params / 1e6:.1f}M')\n",
                "print(f'Trainable parameters: {trainable_params / 1e6:.1f}M')"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Fine-tuning Strategy"],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Freeze all layers except the final classifier\n",
                "for param in model_pretrained.parameters():\n",
                "    param.requires_grad = False\n",
                "\n",
                "# Replace the final layer\n",
                "num_classes = 10  # e.g., CIFAR-10\n",
                "num_ftrs = model_pretrained.fc.in_features\n",
                "model_pretrained.fc = nn.Linear(num_ftrs, num_classes)\n",
                "\n",
                "# Only the new classifier layer will be trained\n",
                "trainable_params = sum(p.numel() for p in model_pretrained.parameters() if p.requires_grad)\n",
                "print(f'Trainable parameters after modification: {trainable_params / 1e6:.1f}M')"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Prepare Data"],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare data with ImageNet normalization\n",
                "transform = transforms.Compose([\n",
                "    transforms.Resize(224),\n",
                "    transforms.ToTensor(),\n",
                "    transforms.Normalize(mean=[0.485, 0.456, 0.406],\n",
                "                       std=[0.229, 0.224, 0.225])\n",
                "])\n",
                "\n",
                "train_dataset = datasets.CIFAR10(root='./data', train=True, \n",
                "                                  download=True, transform=transform)\n",
                "test_dataset = datasets.CIFAR10(root='./data', train=False, \n",
                "                                 download=True, transform=transform)\n",
                "\n",
                "# Use a subset for quick training\n",
                "indices = torch.randperm(len(train_dataset))[:1000]\n",
                "train_dataset = torch.utils.data.Subset(train_dataset, indices)\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
                "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Train the Fine-tuned Model"],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model_pretrained = model_pretrained.to(device)\n",
                "\n",
                "loss_fn = nn.CrossEntropyLoss()\n",
                "# Only optimize the parameters that require gradients\n",
                "optimizer = optim.Adam([p for p in model_pretrained.parameters() if p.requires_grad], lr=0.001)\n",
                "\n",
                "epochs = 5\n",
                "train_losses = []\n",
                "val_accs = []\n",
                "\n",
                "for epoch in range(epochs):\n",
                "    # Training\n",
                "    model_pretrained.train()\n",
                "    train_loss = 0\n",
                "    \n",
                "    for images, labels in train_loader:\n",
                "        images, labels = images.to(device), labels.to(device)\n",
                "        \n",
                "        outputs = model_pretrained(images)\n",
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
                "    model_pretrained.eval()\n",
                "    correct = 0\n",
                "    total = 0\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        for images, labels in test_loader:\n",
                "            images, labels = images.to(device), labels.to(device)\n",
                "            outputs = model_pretrained(images)\n",
                "            _, predicted = torch.max(outputs.data, 1)\n",
                "            total += labels.size(0)\n",
                "            correct += (predicted == labels).sum().item()\n",
                "    \n",
                "    val_acc = 100 * correct / total\n",
                "    val_accs.append(val_acc)\n",
                "    \n",
                "    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%')"
            ],
            "id": "cell-9"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Results"],
            "id": "cell-10"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
                "\n",
                "ax1.plot(train_losses)\n",
                "ax1.set_title('Training Loss')\n",
                "ax1.set_ylabel('Loss')\n",
                "ax1.set_xlabel('Epoch')\n",
                "ax1.grid(True)\n",
                "\n",
                "ax2.plot(val_accs)\n",
                "ax2.set_title('Validation Accuracy')\n",
                "ax2.set_ylabel('Accuracy (%)')\n",
                "ax2.set_xlabel('Epoch')\n",
                "ax2.grid(True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(f'\\nTransfer learning achieved {val_accs[-1]:.2f}% accuracy with minimal training!')"
            ],
            "id": "cell-11"
        }
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def create_pytorch_heat_maps_notebook() -> Dict:
    """Create visualization/heat maps notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Visualizing CNN Feature Maps and Heat Maps"],
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
                "import torch.nn.functional as F\n",
                "import torchvision.models as models\n",
                "import torchvision.transforms as transforms\n",
                "from torch.utils.data import DataLoader\n",
                "import torchvision.datasets as datasets\n",
                "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from matplotlib.cm import get_cmap"
            ],
            "id": "cell-1"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Load Pre-trained Model"],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load ResNet18 for visualization\n",
                "model = models.resnet18(pretrained=True)\n",
                "model.eval()\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = model.to(device)"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Visualize Feature Maps"],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Hook to capture intermediate features\n",
                "class FeatureExtractor:\n",
                "    def __init__(self, model, layer_name):\n",
                "        self.features = []\n",
                "        self.layer = dict(model.named_modules())[layer_name]\n",
                "        self.hook = self.layer.register_forward_hook(self.save_features)\n",
                "    \n",
                "    def save_features(self, module, input, output):\n",
                "        self.features = output.detach()\n",
                "    \n",
                "    def remove(self):\n",
                "        self.hook.remove()\n",
                "\n",
                "# Load test image\n",
                "transform = transforms.Compose([\n",
                "    transforms.Resize(224),\n",
                "    transforms.ToTensor(),\n",
                "    transforms.Normalize(mean=[0.485, 0.456, 0.406],\n",
                "                       std=[0.229, 0.224, 0.225])\n",
                "])\n",
                "\n",
                "test_dataset = datasets.CIFAR10(root='./data', train=False, \n",
                "                                 download=True, transform=transform)\n",
                "test_image, label = test_dataset[0]\n",
                "\n",
                "# Extract features from a middle layer\n",
                "extractor = FeatureExtractor(model, 'layer2')\n",
                "with torch.no_grad():\n",
                "    output = model(test_image.unsqueeze(0).to(device))\n",
                "\n",
                "features = extractor.features.cpu()\n",
                "extractor.remove()\n",
                "\n",
                "print(f'Feature map shape: {features.shape}')"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize the first 16 feature maps\n",
                "feature_maps = features[0].numpy()\n",
                "n_features = min(16, feature_maps.shape[0])\n",
                "\n",
                "fig, axes = plt.subplots(4, 4, figsize=(10, 10))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for i in range(n_features):\n",
                "    ax = axes[i]\n",
                "    fmap = feature_maps[i]\n",
                "    ax.imshow(fmap, cmap='viridis')\n",
                "    ax.set_title(f'Feature {i}')\n",
                "    ax.axis('off')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
            "id": "cell-6"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Gradient-based Heat Map (Class Activation Map)"],
            "id": "cell-7"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class GradCAM:\n",
                "    def __init__(self, model, target_layer):\n",
                "        self.model = model\n",
                "        self.target_layer = target_layer\n",
                "        self.features = None\n",
                "        self.gradients = None\n",
                "        \n",
                "        # Register hooks\n",
                "        self.target_layer.register_forward_hook(self.save_features)\n",
                "        self.target_layer.register_backward_hook(self.save_gradients)\n",
                "    \n",
                "    def save_features(self, module, input, output):\n",
                "        self.features = output.detach()\n",
                "    \n",
                "    def save_gradients(self, module, grad_input, grad_output):\n",
                "        self.gradients = grad_output[0].detach()\n",
                "    \n",
                "    def generate(self, x, target_class=None):\n",
                "        self.model.eval()\n",
                "        output = self.model(x)\n",
                "        \n",
                "        if target_class is None:\n",
                "            target_class = output.argmax(dim=1)\n",
                "        \n",
                "        self.model.zero_grad()\n",
                "        class_loss = output[0, target_class].sum()\n",
                "        class_loss.backward()\n",
                "        \n",
                "        # Compute Grad-CAM\n",
                "        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)\n",
                "        cam = (self.features * gradients).sum(dim=1, keepdim=True)\n",
                "        cam = F.relu(cam)\n",
                "        \n",
                "        return cam[0, 0].cpu().numpy()\n",
                "\n",
                "# Generate GradCAM\n",
                "grad_cam = GradCAM(model, model.layer4[1].conv2)\n",
                "test_tensor = test_image.unsqueeze(0).to(device)\n",
                "cam = grad_cam.generate(test_tensor)\n",
                "\n",
                "# Normalize CAM\n",
                "cam = (cam - cam.min()) / (cam.max() - cam.min())\n",
                "\n",
                "print(f'CAM shape: {cam.shape}')"
            ],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize the CAM\n",
                "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
                "\n",
                "# Original image\n",
                "orig_img = test_image.permute(1, 2, 0).numpy()\n",
                "orig_img = (orig_img * np.array([0.229, 0.224, 0.225]) + \n",
                "            np.array([0.485, 0.456, 0.406]))\n",
                "orig_img = np.clip(orig_img, 0, 1)\n",
                "\n",
                "axes[0].imshow(orig_img)\n",
                "axes[0].set_title('Original Image')\n",
                "axes[0].axis('off')\n",
                "\n",
                "# CAM overlay\n",
                "axes[1].imshow(orig_img)\n",
                "axes[1].imshow(cam, cmap='jet', alpha=0.4)\n",
                "axes[1].set_title('Grad-CAM Visualization')\n",
                "axes[1].axis('off')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
            "id": "cell-9"
        }
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def save_notebook(notebook: Dict, path: str):
    """Save a notebook as JSON to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Saved: {path}")


if __name__ == "__main__":
    base_path = "/Users/vkapoor/python_workspace/goodplacedeeplearning/PyTorch"

    print("Creating additional PyTorch notebooks...")

    # 6. CIFAR-10 Regularization
    notebook6 = create_pytorch_regularization_notebook()
    save_notebook(notebook6, f"{base_path}/04_Regularization/00_cifar10_regularization.ipynb")

    # 7. IMDB Overfitting/Underfitting
    notebook7 = create_pytorch_imdb_overfit_notebook()
    save_notebook(notebook7, f"{base_path}/04_Regularization/01_imdb_overfit_underfit.ipynb")

    # 8. Transfer Learning
    notebook8 = create_pytorch_transfer_learning_notebook()
    save_notebook(notebook8, f"{base_path}/05_Transfer_Learning/00_imagenet_transfer_learning.ipynb")

    # 9. Heat Maps / Visualization
    notebook9 = create_pytorch_heat_maps_notebook()
    save_notebook(notebook9, f"{base_path}/05_Transfer_Learning/01_visualize_heat_maps.ipynb")

    print("\nAdditional notebooks created successfully!")
