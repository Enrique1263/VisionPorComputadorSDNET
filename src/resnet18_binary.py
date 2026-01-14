import torch
import torch.nn as nn
from torchvision import models


class ResNet18Binary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Replace classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

        # IMPORTANT: expose layer4 for Grad-CAM
        self.layer4 = self.backbone.layer4

    def forward(self, x):
        return self.backbone(x)
