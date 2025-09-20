# MNIST Digit Classification using Convolutional Neural Network


## Overview
```
- CNN based model trained over MNIST digit data to classify handwritten digits.
- Test accuracy of >99.4% was achieved within 20 epochs with 10,218 parameters.
- Various techniques were used in the architecture.
```

## Data Overview

```
- MNIST database consists of handwritten digit images each of 28x28 pixel size. 
- It contains 60,000 training images and 10,000 testing images.
- As they contains of handwritten digits or numbers, the number of classes or labels possible are 10.
```

- Distribution of each category in the training dataset.

# link to image

## Network Architecture

```
- The network consists of three convolution blocks and two transition blocks.
- The network flow is as following
    - Convolution Block 1 ----> Transition Block 1 ----> Convolution Block 2 ----> Transition Block 2 ----> Convolution Block 3 ----> Output
```
```
  1> Convolution Block 1
    - The network begins with a block of three convolutional layers, each followed by a Batch Normalization layer (nn.BatchNorm2d) and a Rectified Linear Unit (ReLU) activation function.
  2> Transition Block 1
    - Following the initial convolutions, the feature maps are downsampled. Max pool, Drop out and 1x1 convolution is applied.
    
  3> Convolution Block 2
    - Intermediate Convolutional Block with two convolutional layers.

  4> Transition Block 2
    - Similar to the first downsampling stage, this section further reduces the feature map size with max pool layer. The number of kernels is reduced by making use of 1x1 convolution.

  5> Convolution Block 3
    - The network concludes with layers designed to output the final class probabilities. Consists of two convolution layers.

  6> Output
    - GAP & softmax to output the final class values.
```
### Network Architecture and Receptive Field Details

| **Layer**           | **Input Size (HxW)** | **Output Size (HxW)** | **Input Channels** | **Output Channels** | **Receptive Field (RF)**   | **Details**                                   |
|----------------------|----------------------|------------------------|---------------------|----------------------|------------------------|-----------------------------------------------|
| **Input**            | 28x28               | 28x28                 | 1                   | 1                    | 1                        | Initial image                                 |
| **Conv1**            | 28x28               | 28x28                 | 1                   | 8                    | 3                        | Kernel: 3x3, Padding: 1                      |
| **Conv2**            | 28x28               | 28x28                 | 8                   | 16                   | 5                        | Kernel: 3x3, Padding: 1                      |
| **Conv3**            | 28x28               | 28x28                 | 16                  | 16                   | 7                        | Kernel: 3x3, Padding: 1                      |
| **MaxPool1**         | 28x28               | 14x14                 | 16                  | 16                   | 8                        | Kernel: 2x2, Stride: 2                       |
| **Transition1 (1x1)**| 14x14               | 14x14                 | 16                  | 8                    | 8                        | Kernel: 1x1, No change to RF                 |
| **Conv4**            | 14x14               | 14x14                 | 8                   | 16                   | 12                       | Kernel: 3x3, Padding: 1                      |
| **Conv5**            | 14x14               | 14x14                 | 16                  | 16                   | 16                       | Kernel: 3x3, Padding: 1                      |
| **MaxPool2**         | 14x14               | 7x7                   | 16                  | 16                   | 18                       | Kernel: 2x2, Stride: 2                       |
| **Transition2 (1x1)**| 7x7                 | 7x7                   | 16                  | 8                    | 18                       | Kernel: 1x1, No change to RF                 |
| **Conv6**            | 7x7                 | 7x7                   | 8                   | 16                   | 26                       | Kernel: 3x3, Padding: 1                      |
| **Conv7**            | 7x7                 | 5x5                   | 32                  | 10                   | 34                       | Kernel: 3x3, No Padding                      |
| **GAP**              | 5x5                 | 1x1                   | 10                  | 10                   | 34                       | Adaptive Average Pooling to 1x1              |
| **Output**           | 1x1                 | 1x1                   | 10                  | 10                   | 34                       | Final output after GAP                       |


### PyTorch Summary of the network
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
              ReLU-3            [-1, 8, 28, 28]               0
            Conv2d-4           [-1, 16, 28, 28]           1,168
       BatchNorm2d-5           [-1, 16, 28, 28]              32
              ReLU-6           [-1, 16, 28, 28]               0
            Conv2d-7           [-1, 16, 28, 28]           2,320
       BatchNorm2d-8           [-1, 16, 28, 28]              32
              ReLU-9           [-1, 16, 28, 28]               0
        MaxPool2d-10           [-1, 16, 14, 14]               0
      BatchNorm2d-11           [-1, 16, 14, 14]              32
        Dropout2d-12           [-1, 16, 14, 14]               0
           Conv2d-13            [-1, 8, 14, 14]             136
      BatchNorm2d-14            [-1, 8, 14, 14]              16
           Conv2d-15           [-1, 16, 14, 14]           1,168
      BatchNorm2d-16           [-1, 16, 14, 14]              32
             ReLU-17           [-1, 16, 14, 14]               0
           Conv2d-18           [-1, 16, 14, 14]           2,320
      BatchNorm2d-19           [-1, 16, 14, 14]              32
             ReLU-20           [-1, 16, 14, 14]               0
        MaxPool2d-21             [-1, 16, 7, 7]               0
      BatchNorm2d-22             [-1, 16, 7, 7]              32
           Conv2d-23              [-1, 8, 7, 7]             136
      BatchNorm2d-24              [-1, 8, 7, 7]              16
           Conv2d-25             [-1, 16, 7, 7]           1,168
      BatchNorm2d-26             [-1, 16, 7, 7]              32
             ReLU-27             [-1, 16, 7, 7]               0
           Conv2d-28             [-1, 10, 5, 5]           1,450
AdaptiveAvgPool2d-29             [-1, 10, 1, 1]               0
================================================================
Total params: 10,218
Trainable params: 10,218
Non-trainable params: 0
----------------------------------------------------------------
```

## Techniques Used

### Random Data Augmentaton
### Use of MaxPool Layers         - to increase the receptive field.
### Use of Batch Normalization
### Use of Dropout
### Use of GAP layer

```
```

## Training Logs
```
```

## Training Loss & Accuracy Graphs

