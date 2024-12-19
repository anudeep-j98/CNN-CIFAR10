import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Convolution Block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False, dilation = 1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias, dilation = dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Network Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block: Standard Convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),  # Input: 1 channel -> 8 channels
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),  # Input: 1 channel -> 8 channels
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0, bias=False),  # Transition with stride=2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )

        # Convolution Block 1: Depthwise Separable Convolution
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, padding=1, bias=False),  # 8 -> 16 channels
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            DepthwiseSeparableConv(64, 64, padding=1, bias=False),  # 8 -> 16 channels
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            DepthwiseSeparableConv(64, 32, kernel_size=3, dilation=2, padding=0, bias=False),  # Transition with stride=2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )

        # Convolution Block 3: Standard Convolution with Down-sampling
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),  # Transition with stride=2
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # Transition with stride=2
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)  # Transition with stride=2
        )

        # Global Average Pooling (GAP) + Fully Connected Layer
        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP reduces to 1x1
        # self.fc1 = nn.Linear(512, 256, bias=False)
        # self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)  # Fully Connected Layer for 10 classes

    def forward(self, x):
        # input image size => [-1, 3, 64, 64]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 64) # Flatten to match input of FC layer
        x = self.fc3(x)     # Fully Connected Layer
        return F.log_softmax(x, dim=1)  # Output probabilities (log-softmax for classification)