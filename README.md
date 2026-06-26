# Autograd Engine & Simple Neural Network

This repository contains a scalar-valued automatic differentiation engine and a simple neural network library built entirely from scratch in Python[cite: 1]. It was implemented by following along with Andrej Karpathy's "zero-to-hero" neural network tutorials.

## Features

* **Scalar Autograd Engine**: Implements a custom `Value` class that tracks data and dynamically builds a computational graph to compute gradients[cite: 1].
* **Mathematical Operations**: Supports core operations including addition, subtraction, multiplication, division, and power operations (`+`, `-`, `*`, `/`, `**`), as well as `tanh` and `exp` functions[cite: 1].
* **Topological Sort**: Utilizes a topological sorting algorithm to perform backpropagation and correctly chain gradients through complex expressions[cite: 1].
* **Neural Network Library**: Includes a minimal API for building multi-layer perceptrons (MLPs) from scratch, featuring custom `Neuron`, `Layer`, and `MLP` classes[cite: 1].
* **Graph Visualization**: Uses `graphviz` to generate clear, visual representations of the computational graph, showing nodes, operations, weights, and gradients[cite: 2].
* **PyTorch Validation**: Includes a dedicated section that rebuilds the computational graph using PyTorch to validate the accuracy of the manual forward and backward passes[cite: 2].

## Code Structure

The implementation is contained within a Jupyter Notebook and demonstrates the following workflow:
1. Defining the core `Value` object and its derivatives[cite: 1].
2. Manually calculating slopes and gradients for simple expressions[cite: 2].
3. Building the `trace` and `draw_dot` functions to visualize the graph layout using Graphviz[cite: 2].
4. Constructing a basic neural network architecture[cite: 1].
5. Performing a training loop with forward pass, loss calculation (mean squared error), backward pass, and gradient descent optimization[cite: 1].

## Dependencies

To run the notebook, you will need the following libraries:
* `numpy`[cite: 1]
* `matplotlib`[cite: 1]
* `graphviz`[cite: 2]
* `torch` (for validation only)[cite: 2]
