# RNN based Spoken Digits Classifier 

This repository contains my submission for the **Innatera Staff Embedded Neuromorphic Applications Engineer** home assignment. The goal was to train and optimize an RNN-based classifier on the **Free Spoken Digits Dataset (FSDD)** using PyTorch, under increasingly strict hardware constraints.

The project follows the three tasks outlined in the assignment:

| Task | Objective |
|------|-----------|
| **Task A** | Train a high-performing RNN model on FSDD with any preprocessing or architecture choice |
| **Task B** | Adapt the network to fit within **36kB of memory** and run in **fixed-point precision** |
| **Task C** | Further constrain the model so that all weight parameters are **powers of two**, mimicking hardware-friendly computation |


## Features & Methods Used

- **Samples metadata inspection** using `plotly` (channels, sample rates, durations)
- **MFCC feature extraction** using `torchaudio.transforms.MFCC`
- **Custom `torch.utils.data.Dataset`** for real-time MFCC computation and preprocessing
- **LSTM-based RNN architecture** with configurable depth, dropout, and hidden size
- **Analytical memory-constrained design**, using a derived quadratic expression to solve for `hidden_dim` under a 36 KB constraint
- **Post-training static quantization** with `QuantStub`, and calibrated activations
- **Power-of-2 weight quantization** (an inspired implementation)
- **Model size analysis** using `model.named_parameters()`
- **Classification report and confusion matrix visualization** with `sklearn` and `plotly`


## Sample Results

| Model Variant                     | Accuracy | Model Size | Inference time 
|-----------------------------------|----------|------------|----------|
| Baseline (float32)                | 99.00%   | 807.04 KB   | --   |
| Memory Constrained (float32)      | 93.33%   | 13.00 KB   | --   |
| Quantized (int8)                  | 93.33%   | 13.00 KB   | --   |
| Power-of-2 Quantized              | *Tuned*  | *< 36 KB*  | --  |


---

## Detailed Implementations

To validate the results shown above and demonstrate how each task was tackled, refer to the following Jupyter notebooks:

- **Task A:**  
  [`Training Without Constraints`](notebooks/innatera-task-a.ipynb)  

- **Task B:**  
  [`Memory-Constrained Training (≤36 KB)`](notebooks/TASK-B.ipynb)  

- **Task B:**  
  [`No Floating Point / Int8 Quantized Model:`](notebooks/TASK-B-Part-2.ipynb)  

- **Task C:**  
  [`Power of Two Quantized Model:`](notebooks/TASK-B-Part-2.ipynb)


These notebooks include full training loops, model summaries, memory profiling, and visualizations to support the results presented.

---

## Installation & Setup

To set up the environment and install dependencies, please follow the step-by-step instructions in:

[`docs/install.md`](docs/install.md)


## Key Learnings

## Future Work


