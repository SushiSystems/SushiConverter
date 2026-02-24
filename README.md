# SushiConverter

Simple tool to convert and optimize neural network models for deployment. It supports multiple input and output formats with a focus on static shapes and compatibility for edge devices.

## Features

- Converts Darknet (.cfg/.weights) to ONNX, Caffe, PyTorch, and standalone Python source.
- Converts PyTorch (.pt) models to ONNX.
- Enforces ONNX Opset 11 for hardware compatibility.
- Provides standalone source code generation for Darknet models.
- Includes a numerical validation pipeline to compare output results.
- Supports raw tensor export or decoded YOLO layers.
- Optimizes ONNX graphs for NPU deployment.

## Environment

A dedicated virtual environment is required for stability. The program is validated with specific library versions listed in the requirements file.
Recommended Python version: 3.9.25

## Installation

pip install -r requirements.txt

## Usage

### Darknet to ONNX
python main.py --mode darknet --graph model.cfg --weights model.weights --output model.onnx --validate

### PyTorch to ONNX
python main.py --mode pytorch --weights model.pt --output model.onnx --validate

### Darknet to PyTorch (.pt)
python main.py --mode darknet --graph model.cfg --weights model.weights --output-mode pytorch --output model.pt

### Darknet to Caffe
python main.py --mode darknet --graph model.cfg --weights model.weights --output-mode caffe

### Standalone Python Source (.py + .pth)
python main.py --mode darknet --graph model.cfg --weights model.weights --output-mode source --output standalone_model

## Arguments

*--mode*:
Input model type (darknet, pytorch, onnx).

*--graph*:
Path to the model graph definition (e.g., .cfg file).

*--weights*:
Path to the weight file (.weights, .pt, or .pth).

*--output-mode*:
Target format (onnx, pytorch, caffe, source, pth).

*--validate*:
Compare source and output results using numerical tests.

*--yolo-layer*:
Include decoded YOLO predictions in the output. Default is raw tensors.

*--no-simplify*:
Skip the ONNX simplification step.

*--shape*:
Define input shape B C H W (default: 1 3 416 416).

## Status

This project is a prototype. It is tested on basic darknet tiny models and is intended for model conversion and optimization for edge devices.
