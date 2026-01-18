import torch
import torch.nn as nn
import torch.nn.functional as F


class CrackCNNCustom(nn.Module):
    """
    Custom CNN optimized for crack-like patterns.
    Fully Streamlit + Grad-CAM compatible.
    """

    def __init__(self):
        super().__init__()

        # Exposed as `features` for Grad-CAM
        self.features = nn.Sequential(
            # Low-level edges
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112

            # Texture aggregation
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56

            # Crack continuity
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28

            # High-level structure
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
