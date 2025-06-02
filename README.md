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

