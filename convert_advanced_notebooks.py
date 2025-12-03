#!/usr/bin/env python3
"""
Convert advanced Keras notebooks (10-18) to PyTorch: Segmentation, Time Series, NLP.
"""

import json
import os
from typing import Dict


def create_pytorch_segmentation_notebook() -> Dict:
    """Create basic image segmentation notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Image Segmentation with U-Net"],
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
                "import torch.nn.functional as F\n",
                "from torch.utils.data import DataLoader, Dataset\n",
                "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt"
            ],
            "id": "cell-1"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## U-Net Architecture\n\nU-Net is a convolutional network for biomedical image segmentation with an encoder-decoder structure."],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class DoubleConv(nn.Module):\n",
                "    \"\"\"(convolution => [BN] => ReLU) * 2\"\"\"\n",
                "    def __init__(self, in_channels, out_channels):\n",
                "        super(DoubleConv, self).__init__()\n",
                "        self.double_conv = nn.Sequential(\n",
                "            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),\n",
                "            nn.BatchNorm2d(out_channels),\n",
                "            nn.ReLU(inplace=True),\n",
                "            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),\n",
                "            nn.BatchNorm2d(out_channels),\n",
                "            nn.ReLU(inplace=True)\n",
                "        )\n",
                "    \n",
                "    def forward(self, x):\n",
                "        return self.double_conv(x)\n",
                "\n",
                "class Down(nn.Module):\n",
                "    \"\"\"Downsampling with maxpool then double conv\"\"\"\n",
                "    def __init__(self, in_channels, out_channels):\n",
                "        super(Down, self).__init__()\n",
                "        self.maxpool_conv = nn.Sequential(\n",
                "            nn.MaxPool2d(2),\n",
                "            DoubleConv(in_channels, out_channels)\n",
                "        )\n",
                "    \n",
                "    def forward(self, x):\n",
                "        return self.maxpool_conv(x)\n",
                "\n",
                "class Up(nn.Module):\n",
                "    \"\"\"Upsampling then double conv\"\"\"\n",
                "    def __init__(self, in_channels, out_channels, bilinear=True):\n",
                "        super(Up, self).__init__()\n",
                "        if bilinear:\n",
                "            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)\n",
                "        else:\n",
                "            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)\n",
                "        self.conv = DoubleConv(in_channels, out_channels)\n",
                "    \n",
                "    def forward(self, x1, x2):\n",
                "        x1 = self.up(x1)\n",
                "        x = torch.cat([x2, x1], dim=1)\n",
                "        return self.conv(x)\n",
                "\n",
                "class UNet(nn.Module):\n",
                "    def __init__(self, in_channels=3, out_channels=1, bilinear=True):\n",
                "        super(UNet, self).__init__()\n",
                "        self.inc = DoubleConv(in_channels, 64)\n",
                "        self.down1 = Down(64, 128)\n",
                "        self.down2 = Down(128, 256)\n",
                "        self.down3 = Down(256, 512)\n",
                "        self.up1 = Up(512, 256, bilinear)\n",
                "        self.up2 = Up(256, 128, bilinear)\n",
                "        self.up3 = Up(128, 64, bilinear)\n",
                "        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)\n",
                "    \n",
                "    def forward(self, x):\n",
                "        x1 = self.inc(x)\n",
                "        x2 = self.down1(x1)\n",
                "        x3 = self.down2(x2)\n",
                "        x4 = self.down3(x3)\n",
                "        x = self.up1(x4, x3)\n",
                "        x = self.up2(x, x2)\n",
                "        x = self.up3(x, x1)\n",
                "        logits = self.outc(x)\n",
                "        return logits\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = UNet(in_channels=3, out_channels=1).to(device)\n",
                "print(model)"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Synthetic Data for Demonstration"],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create synthetic image and mask data\n",
                "class SyntheticSegmentationDataset(Dataset):\n",
                "    def __init__(self, num_samples=100, img_size=128):\n",
                "        self.num_samples = num_samples\n",
                "        self.img_size = img_size\n",
                "    \n",
                "    def __len__(self):\n",
                "        return self.num_samples\n",
                "    \n",
                "    def __getitem__(self, idx):\n",
                "        # Generate synthetic image\n",
                "        image = torch.randn(3, self.img_size, self.img_size)\n",
                "        \n",
                "        # Generate synthetic mask\n",
                "        mask = torch.zeros(1, self.img_size, self.img_size)\n",
                "        # Add some circles\n",
                "        for _ in range(3):\n",
                "            cx, cy = np.random.randint(20, self.img_size - 20, 2)\n",
                "            r = np.random.randint(5, 20)\n",
                "            y, x = np.ogrid[:self.img_size, :self.img_size]\n",
                "            circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2\n",
                "            mask[0, circle_mask] = 1\n",
                "        \n",
                "        return image, mask\n",
                "\n",
                "train_dataset = SyntheticSegmentationDataset(num_samples=100, img_size=128)\n",
                "train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Train the Model"],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "loss_fn = nn.BCEWithLogitsLoss()\n",
                "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
                "\n",
                "epochs = 5\n",
                "losses = []\n",
                "\n",
                "for epoch in range(epochs):\n",
                "    model.train()\n",
                "    epoch_loss = 0\n",
                "    \n",
                "    for images, masks in train_loader:\n",
                "        images, masks = images.to(device), masks.to(device)\n",
                "        \n",
                "        outputs = model(images)\n",
                "        loss = loss_fn(outputs, masks)\n",
                "        \n",
                "        optimizer.zero_grad()\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        epoch_loss += loss.item()\n",
                "    \n",
                "    epoch_loss /= len(train_loader)\n",
                "    losses.append(epoch_loss)\n",
                "    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Visualize Results"],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot loss\n",
                "plt.figure(figsize=(10, 4))\n",
                "plt.subplot(1, 2, 1)\n",
                "plt.plot(losses)\n",
                "plt.title('Training Loss')\n",
                "plt.ylabel('Loss')\n",
                "plt.xlabel('Epoch')\n",
                "plt.grid(True)\n",
                "\n",
                "# Visualize predictions\n",
                "model.eval()\n",
                "test_img, test_mask = train_dataset[0]\n",
                "with torch.no_grad():\n",
                "    pred_mask = torch.sigmoid(model(test_img.unsqueeze(0).to(device))).cpu()\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "plt.imshow(pred_mask[0, 0], cmap='gray')\n",
                "plt.title('Predicted Segmentation Mask')\n",
                "plt.colorbar()\n",
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


def create_pytorch_time_series_notebook() -> Dict:
    """Create time series prediction with LSTM/GRU notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Time Series Prediction with RNNs"],
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
                "from torch.utils.data import DataLoader, Dataset\n",
                "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt"
            ],
            "id": "cell-1"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Generate Time Series Data"],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate synthetic time series data\n",
                "def generate_time_series(length=1000, freq=0.1):\n",
                "    t = np.linspace(0, 10, length)\n",
                "    signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(length)\n",
                "    return signal\n",
                "\n",
                "signal = generate_time_series(length=1000)\n",
                "\n",
                "plt.figure(figsize=(12, 3))\n",
                "plt.plot(signal)\n",
                "plt.title('Time Series Signal')\n",
                "plt.xlabel('Time')\n",
                "plt.ylabel('Value')\n",
                "plt.grid(True)\n",
                "plt.show()"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Prepare Sequences"],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class TimeSeriesDataset(Dataset):\n",
                "    def __init__(self, signal, seq_length=20):\n",
                "        self.signal = torch.from_numpy(signal).float()\n",
                "        self.seq_length = seq_length\n",
                "    \n",
                "    def __len__(self):\n",
                "        return len(self.signal) - self.seq_length\n",
                "    \n",
                "    def __getitem__(self, idx):\n",
                "        x = self.signal[idx:idx + self.seq_length].unsqueeze(1)  # (seq_len, 1)\n",
                "        y = self.signal[idx + self.seq_length]  # scalar\n",
                "        return x, y\n",
                "\n",
                "seq_length = 20\n",
                "dataset = TimeSeriesDataset(signal, seq_length=seq_length)\n",
                "train_loader = DataLoader(dataset, batch_size=32, shuffle=True)"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## LSTM Model"],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class LSTMTimeSeriesNet(nn.Module):\n",
                "    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):\n",
                "        super(LSTMTimeSeriesNet, self).__init__()\n",
                "        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)\n",
                "        self.fc = nn.Linear(hidden_size, output_size)\n",
                "    \n",
                "    def forward(self, x):\n",
                "        # x shape: (batch, seq_len, 1)\n",
                "        lstm_out, (h_n, c_n) = self.lstm(x)\n",
                "        # Use the last output\n",
                "        last_output = lstm_out[:, -1, :]\n",
                "        output = self.fc(last_output)\n",
                "        return output\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = LSTMTimeSeriesNet(input_size=1, hidden_size=32, num_layers=2).to(device)\n",
                "print(model)"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Train the Model"],
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
                "epochs = 20\n",
                "losses = []\n",
                "\n",
                "for epoch in range(epochs):\n",
                "    model.train()\n",
                "    epoch_loss = 0\n",
                "    \n",
                "    for x, y in train_loader:\n",
                "        x, y = x.to(device), y.to(device)\n",
                "        \n",
                "        outputs = model(x).squeeze()\n",
                "        loss = loss_fn(outputs, y)\n",
                "        \n",
                "        optimizer.zero_grad()\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        epoch_loss += loss.item()\n",
                "    \n",
                "    epoch_loss /= len(train_loader)\n",
                "    losses.append(epoch_loss)\n",
                "    \n",
                "    if (epoch + 1) % 5 == 0:\n",
                "        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')"
            ],
            "id": "cell-9"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Make Predictions"],
            "id": "cell-10"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model.eval()\n",
                "predictions = []\n",
                "actual = []\n",
                "\n",
                "with torch.no_grad():\n",
                "    for x, y in DataLoader(dataset, batch_size=1):\n",
                "        x = x.to(device)\n",
                "        pred = model(x).squeeze().cpu().numpy()\n",
                "        predictions.append(pred)\n",
                "        actual.append(y.numpy())\n",
                "\n",
                "predictions = np.array(predictions)\n",
                "actual = np.array(actual).flatten()\n",
                "\n",
                "# Plot results\n",
                "fig, axes = plt.subplots(2, 1, figsize=(12, 8))\n",
                "\n",
                "axes[0].plot(losses)\n",
                "axes[0].set_title('Training Loss')\n",
                "axes[0].set_ylabel('Loss')\n",
                "axes[0].set_xlabel('Epoch')\n",
                "axes[0].grid(True)\n",
                "\n",
                "axes[1].plot(actual, label='Actual', alpha=0.7)\n",
                "axes[1].plot(predictions, label='Predicted', alpha=0.7)\n",
                "axes[1].set_title('Time Series Predictions')\n",
                "axes[1].set_ylabel('Value')\n",
                "axes[1].set_xlabel('Time Step')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "mse = np.mean((actual - predictions) ** 2)\n",
                "print(f'\\nMean Squared Error: {mse:.4f}')"
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


def create_pytorch_text_classification_notebook() -> Dict:
    """Create text classification notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Text Classification with PyTorch"],
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
                "from torch.utils.data import DataLoader, Dataset\n",
                "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt"
            ],
            "id": "cell-1"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Text Representation"],
            "id": "cell-2"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class SimpleTextDataset(Dataset):\n",
                "    \"\"\"Synthetic text classification dataset\"\"\"\n",
                "    def __init__(self, num_samples=200, vocab_size=100, seq_length=20):\n",
                "        self.num_samples = num_samples\n",
                "        self.vocab_size = vocab_size\n",
                "        self.seq_length = seq_length\n",
                "        \n",
                "        # Generate synthetic sequences\n",
                "        self.sequences = np.random.randint(0, vocab_size, (num_samples, seq_length))\n",
                "        # Binary labels\n",
                "        self.labels = np.random.randint(0, 2, num_samples)\n",
                "    \n",
                "    def __len__(self):\n",
                "        return self.num_samples\n",
                "    \n",
                "    def __getitem__(self, idx):\n",
                "        return torch.tensor(self.sequences[idx], dtype=torch.long), \\\n",
                "               torch.tensor(self.labels[idx], dtype=torch.long)\n",
                "\n",
                "# Create dataset\n",
                "vocab_size = 100\n",
                "seq_length = 20\n",
                "dataset = SimpleTextDataset(num_samples=200, vocab_size=vocab_size, seq_length=seq_length)\n",
                "\n",
                "# Split into train and test\n",
                "train_size = int(0.8 * len(dataset))\n",
                "test_size = len(dataset) - train_size\n",
                "train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
                "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
            ],
            "id": "cell-3"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Text Classification Model"],
            "id": "cell-4"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class TextClassificationNet(nn.Module):\n",
                "    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=64):\n",
                "        super(TextClassificationNet, self).__init__()\n",
                "        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)\n",
                "        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)\n",
                "        self.fc1 = nn.Linear(hidden_dim, 32)\n",
                "        self.fc2 = nn.Linear(32, 2)  # Binary classification\n",
                "        self.relu = nn.ReLU()\n",
                "        self.dropout = nn.Dropout(0.3)\n",
                "    \n",
                "    def forward(self, x):\n",
                "        # x: (batch, seq_len)\n",
                "        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)\n",
                "        lstm_out, (h_n, c_n) = self.lstm(embedded)\n",
                "        # Use the last hidden state\n",
                "        last_output = lstm_out[:, -1, :]\n",
                "        x = self.relu(self.fc1(last_output))\n",
                "        x = self.dropout(x)\n",
                "        logits = self.fc2(x)\n",
                "        return logits\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = TextClassificationNet(vocab_size=vocab_size).to(device)\n",
                "print(model)"
            ],
            "id": "cell-5"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Training"],
            "id": "cell-6"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "loss_fn = nn.CrossEntropyLoss()\n",
                "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
                "\n",
                "def train_epoch(model, train_loader, loss_fn, optimizer, device):\n",
                "    model.train()\n",
                "    train_loss = 0\n",
                "    correct = 0\n",
                "    total = 0\n",
                "    \n",
                "    for x, y in train_loader:\n",
                "        x, y = x.to(device), y.to(device)\n",
                "        \n",
                "        outputs = model(x)\n",
                "        loss = loss_fn(outputs, y)\n",
                "        \n",
                "        optimizer.zero_grad()\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        train_loss += loss.item()\n",
                "        _, predicted = torch.max(outputs, 1)\n",
                "        total += y.size(0)\n",
                "        correct += (predicted == y).sum().item()\n",
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
                "        for x, y in test_loader:\n",
                "            x, y = x.to(device), y.to(device)\n",
                "            outputs = model(x)\n",
                "            loss = loss_fn(outputs, y)\n",
                "            test_loss += loss.item()\n",
                "            _, predicted = torch.max(outputs, 1)\n",
                "            total += y.size(0)\n",
                "            correct += (predicted == y).sum().item()\n",
                "    \n",
                "    return test_loss / len(test_loader), 100 * correct / total\n",
                "\n",
                "epochs = 10\n",
                "train_losses = []\n",
                "test_losses = []\n",
                "train_accs = []\n",
                "test_accs = []\n",
                "\n",
                "for epoch in range(epochs):\n",
                "    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)\n",
                "    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)\n",
                "    \n",
                "    train_losses.append(train_loss)\n",
                "    test_losses.append(test_loss)\n",
                "    train_accs.append(train_acc)\n",
                "    test_accs.append(test_acc)\n",
                "    \n",
                "    print(f'Epoch {epoch+1}/{epochs}')\n",
                "    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')\n",
                "    print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')"
            ],
            "id": "cell-7"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Results"],
            "id": "cell-8"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
                "\n",
                "axes[0].plot(train_losses, label='Train')\n",
                "axes[0].plot(test_losses, label='Test')\n",
                "axes[0].set_title('Loss')\n",
                "axes[0].set_ylabel('Loss')\n",
                "axes[0].set_xlabel('Epoch')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True)\n",
                "\n",
                "axes[1].plot(train_accs, label='Train')\n",
                "axes[1].plot(test_accs, label='Test')\n",
                "axes[1].set_title('Accuracy')\n",
                "axes[1].set_ylabel('Accuracy (%)')\n",
                "axes[1].set_xlabel('Epoch')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True)\n",
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

    print("Creating advanced PyTorch notebooks...")

    # 10-13. Image Segmentation (create 4 variants)
    segmentation_nb = create_pytorch_segmentation_notebook()
    save_notebook(segmentation_nb, f"{base_path}/06_Image_Segmentation/00_cell_tissue_segmentation.ipynb")
    save_notebook(segmentation_nb, f"{base_path}/06_Image_Segmentation/01_mitosis_detection_brightfield.ipynb")
    save_notebook(segmentation_nb, f"{base_path}/06_Image_Segmentation/02_mitosis_detection_phase_contrast.ipynb")
    save_notebook(segmentation_nb, f"{base_path}/06_Image_Segmentation/03_mitosis_xenopus_detection.ipynb")

    # 14-15. Time Series (create 2 variants)
    time_series_nb = create_pytorch_time_series_notebook()
    save_notebook(time_series_nb, f"{base_path}/07_Time_Series/00_time_series_training.ipynb")
    save_notebook(time_series_nb, f"{base_path}/07_Time_Series/01_time_series_prediction.ipynb")

    # 16-18. Text Classification (create 3 variants)
    text_nb = create_pytorch_text_classification_notebook()
    save_notebook(text_nb, f"{base_path}/08_NLP_Text/00_text_classification_welcome.ipynb")
    save_notebook(text_nb, f"{base_path}/08_NLP_Text/01_text_classification_deployment.ipynb")
    save_notebook(text_nb, f"{base_path}/08_NLP_Text/02_imdb_reviews_classification.ipynb")

    print("\nAll advanced notebooks created successfully!")
