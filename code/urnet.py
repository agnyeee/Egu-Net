"""Spatial-spectral UR-Net with nonlinear transposed-convolution decoder."""
import torch
import torch.nn.functional as F
from torch import nn


class URNet(nn.Module):
    def __init__(self, bands, materials, dropout_rate, shared_conv3,
                 shared_bn3, decoder_layers):
        super().__init__()
        self.spatial_conv1 = nn.Conv2d(bands, 256, 5, padding=2)
        self.spatial_bn1 = nn.BatchNorm2d(256, momentum=0.1)
        # TensorFlow's tf.nn.dropout is elementwise; use Dropout, not Dropout2d.
        self.drop = nn.Dropout(dropout_rate)
        self.spatial_conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.spatial_bn2 = nn.BatchNorm2d(128, momentum=0.1)
        self.shared_conv3 = shared_conv3
        self.shared_bn3 = shared_bn3
        # A pointwise head is applied after smooth full-resolution upsampling.
        self.abundance_projection = nn.Conv2d(32, materials, kernel_size=1)
        self.decoder = decoder_layers

    def forward(self, cube, return_features=False):
        h, w = cube.shape[-2:]
        f1 = torch.tanh(F.avg_pool2d(self.drop(self.spatial_bn1(self.spatial_conv1(cube))), 2, ceil_mode=False))
        f2 = torch.tanh(F.avg_pool2d(self.spatial_bn2(self.spatial_conv2(f1)), 2, ceil_mode=False))
        f3 = torch.relu(F.avg_pool2d(self.shared_bn3(self.shared_conv3(f2)), 2, ceil_mode=False))
        full_resolution = F.interpolate(
            f3, size=(h, w), mode="bilinear", align_corners=False
        )
        logits = self.abundance_projection(full_resolution)
        abundance = torch.softmax(logits, dim=1)
        d1 = torch.sigmoid(self.decoder[0](abundance))
        d2 = torch.sigmoid(self.decoder[1](d1))
        d3 = torch.sigmoid(self.decoder[2](d2))
        reconstruction = torch.sigmoid(self.decoder[3](d3))
        if return_features:
            return abundance, reconstruction, {"spatial_stage1": f1, "spatial_stage2": f2,
                                               "shared_abundance_features": f3, "abundance": abundance,
                                               "decoder_stage1": d1, "decoder_stage2": d2}
        return abundance, reconstruction
