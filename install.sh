#!/bin/bash

# Keras Deep Learning Course - Installation Script
# This script sets up the course environment with all dependencies

set -e  # Exit on error

echo "=========================================="
echo "Keras Deep Learning Course Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

if [[ $(python3 -c 'import sys; print(sys.version_info >= (3, 7))') == "False" ]]; then
    echo "Error: Python 3.7 or higher is required"
    exit 1
fi

echo ""
echo "Installing dependencies..."
echo "=========================================="

# Install via pip
if [ "$1" = "dev" ]; then
    echo "Installing development dependencies..."
    pip install -e ".[dev]"
else
    echo "Installing course dependencies..."
    pip install -e .
fi

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Navigate to the course directory:"
echo "   cd Keras"
echo ""
echo "2. Start Jupyter:"
echo "   jupyter notebook"
echo ""
echo "3. Begin with: Keras/01_Basics/00_fashion_mnist_basic_cnn.ipynb"
echo ""
echo "Happy learning! 🚀"
