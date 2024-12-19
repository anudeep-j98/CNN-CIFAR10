# Project Title

## Overview
This project implements a convolutional neural network (CNN) architecture designed to achieve high accuracy on image classification tasks. The architecture is specifically tailored to meet the following requirements and optimizations.

## Achievements

### Architecture
- Implemented a CNN architecture consisting of layers C1, C2, C3, and C4.
- **No MaxPooling**: Instead of using MaxPooling layers, the architecture utilizes three 3x3 convolutional layers with a stride of 2 after each convolution block.
- **Total Receptive Field**: The total receptive field (RF) of the network exceeds 44, ensuring that the model captures sufficient context from the input images.
- **Depthwise Separable Convolution**: One of the convolutional layers employs Depthwise Separable Convolution.
- **Dilated Convolution**: Another layer incorporates Dilated Convolution to expand the receptive field without increasing the number of parameters.

### Global Average Pooling (GAP)
- Implemented Global Average Pooling (GAP) as a compulsory layer.
- An optional Fully Connected (FC) layer follows GAP to target the specified number of classes.

### Data Augmentation
Utilized an augmentation library to enhance the training dataset with the following techniques:
- **Horizontal Flip**: Randomly flips images horizontally to increase variability.
- **ShiftScaleRotate**: Applies random shifts, scaling, and rotations to the images.
- **Coarse Dropout**: Implemented with the following parameters:
  - `max_holes = 1`
  - `max_height = 16px`
  - `max_width = 1`
  - `min_holes = 1`
  - `min_height = 16px`
  - `min_width = 16px`
  - `fill_value`: Mean of the dataset
  - `mask_fill_value`: None

### Performance
- Achieved an accuracy of **85%** on the validation dataset.
- The model was trained for a sufficient number of epochs to reach the desired accuracy.
- The total number of parameters in the model is less than **200,000**.

## Conclusion
This project demonstrates the effective use of advanced convolutional techniques and data augmentation strategies to build a robust image classification model. The architecture is optimized for performance while adhering to the specified constraints.

## Model Architecture
````
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 32, 15, 15]           9,216
             ReLU-10           [-1, 32, 15, 15]               0
      BatchNorm2d-11           [-1, 32, 15, 15]              64
          Dropout-12           [-1, 32, 15, 15]               0
           Conv2d-13           [-1, 32, 15, 15]             288
           Conv2d-14           [-1, 64, 15, 15]           2,048
DepthwiseSeparableConv-15           [-1, 64, 15, 15]               0
             ReLU-16           [-1, 64, 15, 15]               0
      BatchNorm2d-17           [-1, 64, 15, 15]             128
          Dropout-18           [-1, 64, 15, 15]               0
           Conv2d-19           [-1, 64, 15, 15]             576
           Conv2d-20           [-1, 64, 15, 15]           4,096
DepthwiseSeparableConv-21           [-1, 64, 15, 15]               0
             ReLU-22           [-1, 64, 15, 15]               0
      BatchNorm2d-23           [-1, 64, 15, 15]             128
          Dropout-24           [-1, 64, 15, 15]               0
           Conv2d-25           [-1, 64, 11, 11]             576
           Conv2d-26           [-1, 32, 11, 11]           2,048
DepthwiseSeparableConv-27           [-1, 32, 11, 11]               0
             ReLU-28           [-1, 32, 11, 11]               0
      BatchNorm2d-29           [-1, 32, 11, 11]              64
          Dropout-30           [-1, 32, 11, 11]               0
           Conv2d-31           [-1, 64, 11, 11]          18,432
             ReLU-32           [-1, 64, 11, 11]               0
      BatchNorm2d-33           [-1, 64, 11, 11]             128
          Dropout-34           [-1, 64, 11, 11]               0
           Conv2d-35           [-1, 64, 11, 11]          36,864
             ReLU-36           [-1, 64, 11, 11]               0
      BatchNorm2d-37           [-1, 64, 11, 11]             128
          Dropout-38           [-1, 64, 11, 11]               0
           Conv2d-39             [-1, 32, 5, 5]          18,432
             ReLU-40             [-1, 32, 5, 5]               0
      BatchNorm2d-41             [-1, 32, 5, 5]              64
          Dropout-42             [-1, 32, 5, 5]               0
           Conv2d-43             [-1, 64, 5, 5]          18,432
             ReLU-44             [-1, 64, 5, 5]               0
      BatchNorm2d-45             [-1, 64, 5, 5]             128
          Dropout-46             [-1, 64, 5, 5]               0
           Conv2d-47             [-1, 64, 5, 5]          36,864
             ReLU-48             [-1, 64, 5, 5]               0
      BatchNorm2d-49             [-1, 64, 5, 5]             128
          Dropout-50             [-1, 64, 5, 5]               0
           Conv2d-51             [-1, 64, 3, 3]          36,864
AdaptiveAvgPool2d-52             [-1, 64, 1, 1]               0
           Linear-53                   [-1, 10]             640
================================================================
Total params: 191,472
Trainable params: 191,472
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.79
Params size (MB): 0.73
Estimated Total Size (MB): 4.53
----------------------------------------------------------------
````


## calculation of receptive field
````
| Layer                           | Rin  | Rout | Jin  | Jout |
|---------------------------------|-------|------|------|------|
| **Input Image**                 | 1     | 1    | 1    | 1    |
| **Block 1 (conv1)**             |       |      |      |      |
| Conv2d(3, 16, K=3, P=1, S=1)    | 1     | 3    | 1    | 1    |
| Conv2d(16, 32, K=3, P=1, S=1)   | 3     | 5    | 1    | 1    |
| Conv2d(32, 32, K=3, P=0, S=2)   | 5     | 7    | 1    | 2    |
| **Block 2 (conv2: Depthwise)**  |       |      |      |      |
| DepthwiseSeparableConv(K=3, P=1)| 7     | 11   | 2    | 2    |
| DepthwiseSeparableConv(K=3, P=1)| 11    | 15   | 2    | 2    |
| DepthwiseSeparableConv(K=3, D=2)| 15    | 27   | 2    | 2    |
| **Block 3 (conv3)**             |       |      |      |      |
| Conv2d(32, 64, K=3, P=1, S=1)   | 27    | 31   | 2    | 2    |
| Conv2d(64, 64, K=3, P=1, S=1)   | 31    | 35   | 2    | 2    |
| Conv2d(64, 32, K=3, P=0, S=2)   | 35    | 39   | 2    | 4    |
| **Block 4 (conv4)**             |       |      |      |      |
| Conv2d(32, 64, K=3, P=1, S=1)   | 39    | 47   | 4    | 4    |
| Conv2d(64, 64, K=3, P=1, S=1)   | 47    | 55   | 4    | 4    |
| Conv2d(64, 64, K=3, P=1, S=2)   | 55    | 63   | 4    | 8    |
| **GAP (Global Avg Pool)**       | 63    | Full | 8    | -    |
````