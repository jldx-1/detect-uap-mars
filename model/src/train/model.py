"""
model.py

This module defines a simplified YOLOv5-based model for UAP detection.
It implements a lightweight classification/ROI network that outputs bounding box
predictions and class scores. Note that this is a simplified version intended as a shell,
which you can expand as needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class YOLOv5S(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(YOLOv5S, self).__init__()
        # Simplified backbone: a few convolutional layers with batch norm and ReLU
        self.conv1 = nn.Conv2d(config.INPUT_CHANNELS, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
        # Additional conv layers could be added here to mimic YOLOv5's CSP structure
        
        # Detection head: outputs for each anchor box: [objectness, x, y, w, h, class scores]
        # For simplicity, we assume a single scale with 3 anchors.
        # Output channels = 3 * (num_classes + 5)
        self.detector = nn.Conv2d(128, 3 * (num_classes + 5), kernel_size=1)
    
    def forward(self, x):
        # Backbone forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Detection head
        x = self.detector(x)
        # Note: In a full implementation, you would reshape and decode predictions,
        # apply anchor boxes, and perform non-max suppression.
        return x

def build_model():
    """
    Constructs and returns the YOLOv5S model.
    """
    model = YOLOv5S(num_classes=config.NUM_CLASSES)
    return model

if __name__ == "__main__":
    # Quick test of the model architecture
    model = build_model()
    print(model)
    # Create a dummy input tensor with shape: (batch_size, channels, height, width)
    x = torch.randn(1, config.INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    y = model(x)
    print("Output shape:", y.shape)
