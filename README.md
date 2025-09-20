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
This custom PyTorch transforms class RandomAugmentation is designed to apply a series of random data augmentations to MNIST grayscale images. The primary goal of this augmentation pipeline is to enhance the model's ability to generalize by exposing it to a wider variety of image variations, preventing overfitting, and improving its robustness to common image distortions.

```
class RandomAugmentation:
    """
    Custom augmentation that randomly applies a sequence of transformations
    to grayscale images, such as MNIST digits.
    """
    def __init__(self, p=0.7):
        self.p = p  # Probability of applying any augmentation

        # Define all available individual transforms
        self.transforms_list = [
            # 1. Randomly rotate the image by a specified degree range
            transforms.RandomRotation(degrees=10),

            # 2. Apply random affine transformations (rotations, translations, scaling)
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),  # Up to 10% translation
                scale=(0.9, 1.1),      # Scaling between 90% and 110%
                shear=10               # Shearing by 10 degrees
            ),

            # 3. Randomly shift the image by cropping a random portion
            transforms.RandomResizedCrop(
                size=28,  # Maintain the original image size for MNIST
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),

            # 4. Apply a random perspective transform
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0),

            # 5. Simulate a change in brightness/contrast (effective even on grayscale)
            transforms.ColorJitter(brightness=0.2, contrast=0.2),

            # 6. Randomly erase a small portion of the image to force the model to learn from partial data
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ]

    def add_noise(self, img):
        """Add random Gaussian noise to a PyTorch tensor."""
        noise = torch.randn_like(img) * 0.1
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0., 1.)

    def __call__(self, img):
        """Applies a random number of augmentations (1 or 2) from the list."""

        # With probability (1-p), return the original image
        if random.random() > self.p:
            return img

        # Randomly decide how many augmentations to apply (1 or 2)
        num_augmentations = random.randint(1, 2)

        # Select a random subset of transformations without replacement
        transforms_to_apply = random.sample(self.transforms_list, num_augmentations)

        # Add noise as a separate, potential augmentation, not a core choice
        if random.random() < 0.5: # 50% chance to also add noise
            transforms_to_apply.append(self.add_noise)

        # Apply the selected sequence of transformations
        for transform in transforms_to_apply:
            if callable(transform):
                img = transform(img)
            else:
                img = transform(img)

        return img
```

```
The RandomAugmentation class applies a random selection of the following transformations to each image with a probability defined by the p parameter (set to 0.7 by default):

    Random Rotation: Rotates the image randomly by up to 10∘.

    Random Affine Transformations: Applies a combination of random rotations, translations, and scaling.

    Random Resized Crop: Simulates slight shifts in the digit's position by cropping and resizing the image.

    Random Perspective: Adds a random perspective distortion to the images.

    Color Jitter: Randomly adjusts the brightness and contrast.

    Random Erasing: Hides a small, random portion of the image to train the model to learn from incomplete data.

By combining these diverse augmentations, the model is forced to focus on the essential features of the digits rather than relying on their exact position or orientation within the image.
```

### Use of MaxPool Layers         - to increase the receptive field.
This is applied twice in our network to boost the receptive field and to reduce the size of the image in the network.

### Use of Batch Normalization
Applied throughout the network for faster convergence and reduce dependency on dropout for regularization effect.

### Use of Dropout
Applied for regularization.

### Use of GAP layer
Spatial Reduction from 5x5 to 1x1 for the final layer to predict the correct class.

### 1x1 Convolution
Applied to reduce the number of kernels in subsequent layers thereby reducing the total number of parameters in the network but at the same time preserving the most important feature during the convolution.


```
```

## Training Logs
```
```

## Training Loss & Accuracy Graphs

