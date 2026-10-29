"""Joint EGU-Net-ss model, with genuine shared module objects."""
from torch import nn
from .enet import SpectralStem, ENet
from .urnet import URNet


class EGUNetSS(nn.Module):
    def __init__(self, bands=156, materials=3, dropout_rate=0.1):
        super().__init__()
        self.bands, self.materials = int(bands), int(materials)
        stem = SpectralStem(self.bands, dropout_rate)
        shared_conv3 = nn.Conv2d(128, 32, 1)
        shared_bn3 = nn.BatchNorm2d(32, momentum=0.1)
        shared_conv4 = nn.Conv2d(32, self.materials, 1)
        self.enet = ENet(stem, shared_conv3, shared_bn3, shared_conv4, self.materials)
        decoder = nn.ModuleList([
            nn.ConvTranspose2d(self.materials, 32, 1),
            nn.ConvTranspose2d(32, 128, 1),
            nn.ConvTranspose2d(128, 256, 3, padding=1),
            nn.ConvTranspose2d(256, self.bands, 5, padding=2),
        ])
        self.urnet = URNet(self.bands, self.materials, dropout_rate,
                           shared_conv3, shared_bn3, decoder)
        # Match the official source's Xavier initialization for convolutional
        # and transposed-convolutional filters; its biases start at zero.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pure_spectra, cube, return_features=False):
        logits, pure_abundances = self.enet(pure_spectra)
        if return_features:
            mixed_abundances, reconstruction, features = self.urnet(cube, return_features=True)
            return {"class_logits": logits, "pure_abundances": pure_abundances,
                    "abundances": mixed_abundances, "reconstruction": reconstruction,
                    "features": features}
        mixed_abundances, reconstruction = self.urnet(cube)
        return {"class_logits": logits, "pure_abundances": pure_abundances,
                "abundances": mixed_abundances, "reconstruction": reconstruction}

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sharing_is_valid(self):
        return (self.enet.shared_conv3 is self.urnet.shared_conv3 and
                self.enet.shared_bn3 is self.urnet.shared_bn3 and
                self.urnet.abundance_projection is not self.enet.shared_conv4)
