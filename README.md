# Project: Exploring Model Optimization Techniques for ResNet Architectures

## Introduction

This repository contains a series of experiments focused on optimizing deep learning models for efficiency without significant compromises in performance. The primary goal is to explore and quantify the impact of various optimization techniques on standard ResNet architectures. The techniques investigated include pruning, quantization, knowledge distillation, and the use of TensorRT for inference acceleration.

The experiments are conducted on ResNet18 and ResNet50 models, providing a comprehensive analysis of how these optimization methods affect model size, accuracy, and inference speed on both CPU and GPU platforms.

## Model Architectures

The following baseline models are used for the experiments:

*   **ResNet18:** A 18-layer residual network.
*   **ResNet50:** A 50-layer residual network.

## Optimization Techniques Explored

This project investigates several state-of-the-art model optimization techniques. These methods aim to reduce the computational and memory requirements of deep learning models.

### Pruning
Pruning involves removing redundant or less important parameters from a neural network to reduce its size and improve inference speed. The following pruning strategies were explored:
*   **Structured Pruning:** This method removes entire groups of weights, such as filters or channels. The experiments used L1-norm based filter pruning in both one-shot and iterative scenarios.
*   **Unstructured Pruning:** This technique removes individual weights based on their magnitude. The experiments implemented one-shot and iterative L1 unstructured pruning.
*   **N:M Sparsity:** A fine-grained structured pruning technique that prunes N out of every M consecutive weights. This project specifically uses 2:4 sparsity.

### Quantization
Quantization reduces the memory footprint and computation time by using lower-precision numerical formats for model weights and/or activations.
*   **Post-Training Quantization (PTQ):** This technique quantizes a pre-trained model without requiring re-training. Both per-tensor and per-channel INT8 quantization were applied.
*   **Quantization Aware Training (QAT):** QAT simulates the effects of quantization during the training process, which can lead to better performance compared to PTQ.
*   **K-Means Quantization:** This method groups model weights into a predefined number of clusters and uses the cluster centroids as the quantized values.

### Knowledge Distillation
Knowledge distillation is a technique where a smaller "student" model is trained to mimic the behavior of a larger, pre-trained "teacher" model. This can transfer the knowledge from the larger model to a more compact one. Our experiments use a ResNet50 as the teacher to train a ResNet18 student.

### TensorRT Optimization
NVIDIA TensorRT is a high-performance deep learning inference optimizer and runtime library that accelerates AI workloads on NVIDIA GPUs. It applies various optimizations, including layer fusion, kernel auto-tuning, and precision calibration. Experiments were run using both FP32 and FP16 precision.

### Combined Approaches
Several experiments explore the benefits of combining multiple optimization techniques, such as applying TensorRT to a distilled model or quantizing a distilled model.

## Experiment Results

The following tables summarize the results of the various optimization experiments, detailing their impact on accuracy, model size, and performance metrics.

### Accuracy and Model Size Analysis

This table provides a comprehensive overview of the trade-offs between model accuracy and size for each experiment.

| Model ID | Base Model | Optimization Category | Specific Technique | Top-1 Acc. | Top-5 Acc. | Model Size (MB) | Params (M) | Non-Zero Params (M) |
|---|---|---|---|---|---|---|---|---|
| `R18-Baseline` | ResNet18 | Baseline | Baseline | 0.5009 | 0.7632 | 44.67 | 11.69 | 11.69 |
| `R50-Baseline` | ResNet50 | Baseline | Baseline | 0.6495 | 0.8748 | 97.80 | 25.56 | 25.56 |
| `R18-Distill-Quant-KMeans`| ResNet18 | Combined | KMeans Quant (256) | 0.5368 | 0.8032 | 44.67 | 11.69 | 11.69 |
| `R18-Distill-PTQ-PerChannel`| ResNet18 | Combined | PTQ INT8 (Per-Channel)| 0.5078 | 0.7808 | 11.30 | N/A | N/A |
| `R18-Distill-PTQ-PerTensor`| ResNet18 | Combined | PTQ INT8 (Per-Tensor) | 0.5335 | 0.8027 | 11.30 | N/A | N/A |
| `R18-Distill-QAT` | ResNet18 | Combined | QAT INT8 | 0.5356 | 0.8045 | 11.30 | N/A | N/A |
| `R50>R18-KD-Pre` | ResNet18 | Knowledge Distillation| T:resnet50->S:resnet18 | 0.5381 | 0.8027 | 44.67 | 11.69 | 11.69 |
| `R50>R18-KD-Scratch` | ResNet18 | Knowledge Distillation| T:resnet50->S:resnet18 | 0.1764 | 0.4017 | 44.67 | 11.69 | 11.69 |
| `R50-Prune-NM(2:4)` | ResNet50 | Pruning | N:M Sparsity (2:4) | 0.6426 | 0.8731 | 97.80 | 25.56 | 13.83 |
| `R50-P.Struct-IT-sp50` | ResNet50 | Pruning | Iter. Struct. (L1, 50%)| 0.4994 | 0.7675 | 49.60 | 12.94 | 12.94 |
| `R50-P.Struct-IT-sp75` | ResNet50 | Pruning | Iter. Struct. (L1, 75%)| 0.2982 | 0.5799 | 25.49 | 6.63 | 6.63 |
| `R50-P.Struct-IT-sp90` | ResNet50 | Pruning | Iter. Struct. (L1, 90%)| 0.1308 | 0.3406 | 10.21 | 2.63 | 2.63 |
| `R50-P.Struct-OS-sp50` | ResNet50 | Pruning | One-Shot Struct. (L1, 50%)|0.5478 | 0.8101 | 49.60 | 12.94 | 12.94 |
| `R50-P.Struct-OS-sp75` | ResNet50 | Pruning | One-Shot Struct. (L1, 75%)|0.2207 | 0.4841 | 21.86 | 5.68 | 5.68 |
| `R50-P.Struct-OS-sp90` | ResNet50 | Pruning | One-Shot Struct. (L1, 90%)|0.0571 | 0.1947 | 10.56 | 2.72 | 2.72 |
| `R50-P.Unstruct-IT-sp50`| ResNet50 | Pruning | Iter. Unstruct. (L1, 50%)|0.6606 | 0.8531 | 97.79 | 25.56 | 12.75 |
| `R50-P.Unstruct-IT-sp75`| ResNet50 | Pruning | Iter. Unstruct. (L1, 75%)|0.6195 | 0.7928 | 97.79 | 25.56 | 6.38 |
| `R50-P.Unstruct-IT-sp90`| ResNet50 | Pruning | Iter. Unstruct. (L1, 90%)|0.4650 | 0.7520 | 97.80 | 25.56 | 2.55 |
| `R50-P.Unstruct-OS-sp50`| ResNet50 | Pruning | One-Shot Unstruct. (L1, 50%)|0.6645 | 0.8861 | 97.80 | 25.56 | 12.75 |
| `R50-P.Unstruct-OS-sp75`| ResNet50 | Pruning | One-Shot Unstruct. (L1, 75%)|0.6240 | 0.8598 | 97.80 | 25.56 | 6.38 |
| `R50-P.Unstruct-OS-sp90`| ResNet50 | Pruning | One-Shot Unstruct. (L1, 90%)|0.4183 | 0.7122 | 97.80 | 25.56 | 2.55 |
| `R50-Quant-KMeans` | ResNet50 | Quantization | KMeans Quant (256) | 0.6492 | 0.8731 | 97.80 | 25.56 | 25.56 |
| `R50-Quant-PTQ-PerChannel`| ResNet50 | Quantization | PTQ INT8 (Per-Channel)| 0.6069 | 0.8404 | 24.94 | N/A | N/A |
| `R50-Quant-PTQ-PerTensor` | ResNet50 | Quantization | PTQ INT8 (Per-Tensor) | 0.6498 | 0.8736 | 24.94 | N/A | N/A |
| `R50-Quant-QAT` | ResNet50 | Quantization | QAT INT8 | 0.6635 | 0.8807 | 24.94 | N/A | N/A |
| `R50-TRT-FP16` | ResNet50 | TensorRT | TensorRT FP16 | 0.6505 | 0.8746 | 67.38 | 25.56 | 25.56 |
| `R50-TRT-FP32` | ResNet50 | TensorRT | TensorRT FP32 | 0.6492 | 0.8751 | 162.20| 25.56 | 25.56 |
| `R18-Distill-TRT-FP16` | ResNet18 | Combined | Distill + TRT FP16 | 0.5384 | 0.8027 | 31.02 | 11.69 | 11.69 |
| `R18-Distill-TRT-FP32` | ResNet18 | Combined | Distill + TRT FP32 | 0.5379 | 0.8027 | 96.01 | 11.69 | 11.69 |

### Performance Benchmarks

The following tables show the latency and throughput of the optimized models on both GPU and CPU.

#### GPU Performance

| Model | Optimization Category | Latency (ms) | Throughput (FPS) | Size (MB) | GPU Mem (MB) |
|---|---|---|---|---|---|
| R18-Distill-TRT-FP16 | TensorRT (Distilled)| 2.49 | 12865.97 | 31.02 | 18.31 |
| R50-TRT-FP16 | TensorRT (Baseline)| 6.53 | 4903.96 | 67.38 | 18.31 |
| R18-Distill-TRT-FP32 | TensorRT (Distilled)| 8.55 | 3744.62 | 96.01 | 27.50 |
| R50-P.Struct-IT-sp90 | Pruning (Structured)| 9.55 | 3349.67 | 10.21 | 423.72 |
| `R18-Distill-Quant-KMeans`| Combined (Distill+Quant)| 11.46 | 2791.43 | 44.67 | 268.07 |
| R50>R18-KD-Scratch | Knowledge Distillation | 11.78 | 2717.11 | 44.67 | 268.07 |
| R50>R18-KD-Pre | Knowledge Distillation | 11.95 | 2677.82 | 44.67 | 268.07 |
| R18-Baseline | Baseline | 12.50 | 2559.52 | 44.67 | 268.07 |
| R50-P.Struct-OS-sp75 | Pruning (Structured)| 13.76 | 2326.09 | 21.86 | 372.17 |
| R50-P.Struct-OS-sp90 | Pruning (Structured)| 14.30 | 2237.90 | 10.56 | 314.92 |
| R50-P.Struct-IT-sp75 | Pruning (Structured)| 15.24 | 2099.46 | 25.49 | 500.91 |
| R50-TRT-FP32 | TensorRT (Baseline)| 18.10 | 1768.23 | 162.20 | 27.50 |
| R50-P.Struct-IT-sp50 | Pruning (Structured)| 24.67 | 1297.02 | 49.60 | 482.58 |
| R50-P.Struct-OS-sp50 | Pruning (Structured)| 25.12 | 1273.91 | 49.60 | 482.14 |
| R50-P.Unstruct-OS-sp90| Pruning (Unstructured)| 34.83 | 918.85 | 97.80 | 443.60 |
| `R50-Quant-KMeans` | Quantization (KMeans)| 34.86 | 917.88 | 97.80 | 443.60 |
| R50-P.Unstruct-OS-sp75| Pruning (Unstructured)| 35.01 | 913.99 | 97.80 | 443.60 |
| R50-Baseline | Baseline | 35.02 | 913.78 | 97.80 | 443.60 |
| R50-P.Unstruct-IT-sp90| Pruning (Unstructured)| 35.08 | 912.20 | 97.80 | 443.60 |
| R50-P.Unstruct-IT-sp75| Pruning (Unstructured)| 35.09 | 911.89 | 97.79 | 443.60 |
| R50-P.Unstruct-OS-sp50| Pruning (Unstructured)| 35.37 | 904.60 | 97.80 | 443.60 |
| R50-P.Unstruct-IT-sp50| Pruning (Unstructured)| 35.49 | 901.76 | 97.79 | 446.10 |
| `R50-Prune-NM(2:4)` | Pruning (N:M Sparsity)| 37.40 | 855.55 | 97.80 | 443.60 |

#### CPU Performance

| Model | Optimization Category | Latency (ms) | Throughput (FPS) | Size (MB) |
|---|---|---|---|---|
| R50-P.Struct-IT-sp90 | Pruning (Structured)| 385.34 | 83.04 | 10.21 |
| R50-P.Struct-OS-sp90 | Pruning (Structured)| 393.90 | 81.24 | 10.56 |
| R50-P.Struct-OS-sp75 | Pruning (Structured)| 730.68 | 43.79 | 21.86 |
| R50-P.Struct-IT-sp75 | Pruning (Structured)| 791.62 | 40.42 | 25.49 |
| R50>R18-KD-Pre | Knowledge Distillation | 844.32 | 37.90 | 44.67 |
| `R18-Distill-Quant-KMeans`| Combined (Distill+Quant)| 844.44 | 37.90 | 44.67 |
| R50>R18-KD-Scratch | Knowledge Distillation | 848.13 | 37.73 | 44.67 |
| R18-Baseline | Baseline | 850.19 | 37.64 | 44.67 |
| R50-P.Struct-OS-sp50 | Pruning (Structured)| 1401.12 | 22.84 | 49.60 |
| R50-P.Struct-IT-sp50 | Pruning (Structured)| 1403.07 | 22.81 | 49.60 |
| `R50-Prune-NM(2:4)` | Pruning (N:M Sparsity)| 2313.70 | 13.83 | 97.80 |
| R50-P.Unstruct-IT-sp75| Pruning (Unstructured)| 2355.32 | 13.59 | 97.79 |
| R50-P.Unstruct-OS-sp50| Pruning (Unstructured)| 2358.78 | 13.57 | 97.80 |
| R50-P.Unstruct-OS-sp90| Pruning (Unstructured)| 2365.86 | 13.53 | 97.80 |
| R50-P.Unstruct-IT-sp90| Pruning (Unstructured)| 2368.08 | 13.51 | 97.80 |
| R50-P.Unstruct-IT-sp50| Pruning (Unstructured)| 2368.85 | 13.51 | 97.79 |
| R50-Baseline | Baseline | 2369.79 | 13.50 | 97.80 |
| R50-P.Unstruct-OS-sp75| Pruning (Unstructured)| 2373.12 | 13.48 | 97.80 |
| `R50-Quant-KMeans` | Quantization (KMeans)| 2383.02 | 13.43 | 97.80 |

## How to Reproduce

To reproduce the results in this repository, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Set up the environment:**
    *(Please provide instructions on setting up the environment, e.g., using `pip` or `conda`)*
    ```bash
    # Example:
    pip install -r requirements.txt
    ```

3.  **Download Datasets:**
    *(Please provide instructions on how to download and prepare any necessary datasets.)*

4.  **Run Experiments:**
    *(Please provide clear instructions or scripts to run the different optimization experiments.)*
    ```bash
    # Example for running a pruning experiment
    python run_pruning.py --model resnet50 --method iterative_structured --sparsity 0.9

    # Example for running a quantization experiment
    python run_quantization.py --model resnet18 --method qat
    ```
