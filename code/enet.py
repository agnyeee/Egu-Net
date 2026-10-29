"""E-Net: 1×1 spectral convolutions and shared late abundance layers."""
import torch
from torch import nn


class SpectralStem(nn.Module):
    def __init__(self, bands, dropout_rate=0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(bands, 256, 1)
        self.bn1 = nn.BatchNorm2d(256, momentum=0.1)
        self.drop = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(256, 128, 1)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.1)

    def forward(self, x):
        x = torch.tanh(self.drop(self.bn1(self.conv1(x))))
        return torch.tanh(self.bn2(self.conv2(x)))


class ENet(nn.Module):
    def __init__(self, stem, shared_conv3, shared_bn3, shared_conv4, materials):
        super().__init__()
        self.stem = stem
        self.shared_conv3 = shared_conv3
        self.shared_bn3 = shared_bn3
        self.shared_conv4 = shared_conv4
        self.materials = materials

    def forward(self, pure_spectra):
        # Inputs may be N×B or N×B×1×1.
        if pure_spectra.ndim == 2:
            pure_spectra = pure_spectra[:, :, None, None]
        x = self.stem(pure_spectra)
        x = torch.relu(self.shared_bn3(self.shared_conv3(x)))
        logits = self.shared_conv4(x)
        return logits.flatten(1), torch.softmax(logits.flatten(1), dim=1)
