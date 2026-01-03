# Neural Network Implementation

This project implements a neural network from scratch for digit classification using the MNIST dataset.

## Overview

A three-layer neural network implementation that learns to classify handwritten digits (0-9) from the MNIST dataset. The network uses:
- Input layer: 784 neurons (28x28 pixel images flattened)
- Hidden layers: 32 neurons each
- Output layer: 10 neurons (one for each digit)

## Files

- **main.py** - Core neural network implementation with forward propagation, backpropagation, and training logic
- **Testing.ipynb** - Jupyter notebook for testing and evaluating the model

## Requirements

- Python 3.x
- TensorFlow/Keras (for MNIST dataset)
- NumPy
- Pandas
- SciPy

## Key Functions

- `rand_parameters()` - Initialize random network weights and biases
- `write_new_par()` - Save parameters to Excel file
- `read_par()` - Load parameters from Excel file

## Usage

1. Initialize network parameters: `write_new_par()`
2. Run training and testing in the Jupyter notebook (`Testing.ipynb`)


## Notes

- Network parameters are stored in `Neural Network_Parameters.xlsx`

