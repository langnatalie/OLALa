# Online Learned Adaptive Lattice Codes for Heterogeneous Federated Learning
PyTorch implementation of Online Learned Adaptive Lattice Codes for Heterogeneous Federated Learning.

<img width="1496" height="268" alt="image" src="https://github.com/user-attachments/assets/a0cfa59d-20c3-4f31-a983-d8b777f804c6" />

## Introduction
In this work we propose OLALa, an online adaptive lattice quantization framework for communication-efficient heterogeneous federated learning. OLALa learns user-specific lattice generator matrices during federated training and uses the resulting lattice quantizers to compress model updates before server aggregation. This repository contains a basic PyTorch implementation of OLALa. Please refer to our [paper](https://arxiv.org/abs/2506.20297) for more details.

## Usage
This code has been tested on Python 3.10.20, PyTorch 2.7.1+cu118, Torchvision 0.22.1+cu118, NumPy 2.2.6, and Linux 5.14.0.

### Prerequisite
1. PyTorch: https://pytorch.org
2. torchvision
3. numpy
4. tqdm
5. TensorboardX: https://github.com/lanpa/tensorboardX
6. Pillow
7. datasets
8. torchinfo

### Training
```
python main.py --exp_name=olala --mechanism olala --data mnist --model cnn2 --R 3 --lattice_dim 2
```

### Testing
```
python main.py --exp_name=olala --eval
```
