"""
EfficientNet-B0 UNet for 4-class vegetation segmentation.

Camera-robustness architecture:
- InputNormalization: per-image channel standardization that adapts to any camera's
  output statistics at inference (conceptually Instance Normalization applied to input;
  Ulyanov et al., 2016).
- Kaiming He initialization for PReLU decoder layers (He et al., 2015).
- Modality Dropout (ModDrop) for NIR robustness (Neverova et al., 2015).
- Optional Gradient Reversal Layer for domain adaptation (Ganin et al., 2016).
- Input: 4ch (RGB + NIR).
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (Ganin & Lempitsky, 2016)
# ---------------------------------------------------------------------------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    """Binary domain classifier head attached after encoder bottleneck."""
    def __init__(self, in_features, hidden=128):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x, alpha=1.0):
        self.grl.alpha = alpha
        return self.classifier(self.grl(x))


# ---------------------------------------------------------------------------
# Input Normalization for Camera Robustness
# ---------------------------------------------------------------------------
class InputNormalization(nn.Module):
    """Per-image, per-channel zero-mean unit-variance normalization with
    learnable affine rescaling.

    Why this helps camera robustness:
    Different cameras produce different absolute pixel value distributions
    due to different sensor gains, exposure settings, gamma curves, and
    white balance pipelines. BatchNorm in the encoder operates on batch
    statistics which are frozen to training-camera stats at inference.
    Instance-level normalization on the INPUT removes per-image camera
    bias before the encoder sees it.

    The learnable gamma/beta let the network map normalized values back
    to whatever range the pretrained encoder expects, preserving weight
    compatibility.

    """
    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-6
        x_norm = (x - mean) / std
        return self.gamma * x_norm + self.beta


#
# modality Drop -out (Neverova et al., 2015)
# ---------------------------------------------------------------------------
class ModalityDropout(nn.Module):
    """Drops entire channel groups independently during training."""
    def __init__(self, channel_groups, drop_probs):
        super().__init__()
        assert len(channel_groups) == len(drop_probs)
        self.channel_groups = channel_groups
        self.drop_probs = drop_probs

    def forward(self, x):
        if not self.training:
            return x

        B, C, H, W = x.shape
        mask = torch.ones(B, C, 1, 1, device=x.device, dtype=x.dtype)

        for (start, end), p in zip(self.channel_groups, self.drop_probs):
            if p > 0:
                # Per-sample dropout: each sample gets independent dropout decision
                drop_mask = torch.rand(B, 1, 1, 1, device=x.device) < p
                mask[:, start:end, :, :] = (~drop_mask).float()

        return x * mask



class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class EfficientUNet4Class(nn.Module):
    """EfficientNet-B0 encoder + UNet decoder with camera-robust input pipeline.

    Args:
        in_channels: 4 (RGB + NIR).
        num_classes: 4 (dead, severe, moderate, healthy).
        use_domain_head: attach GRL + domain classifier for domain adaptation.
        moddrop_nir_prob: probability of dropping NIR during training.
    """
    def __init__(self, in_channels=4, num_classes=4,
                 use_domain_head=False, moddrop_nir_prob=0.3):
        super().__init__()

        self.input_norm = InputNormalization(in_channels)

        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).features

        old_conv = base[0][0]
        old_weights = old_conv.weight.data.clone()
        new_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2,
                             padding=1, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_weights
            avg_w = old_weights.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:, :, :] = avg_w.repeat(1, in_channels - 3, 1, 1)
        base[0][0] = new_conv

        self.moddrop = ModalityDropout(
            channel_groups=[(0, 3), (3, in_channels)],
            drop_probs=[0.0, moddrop_nir_prob],
        )

        # --- Encoder ---
        self.enc0 = base[0:2]
        self.enc1 = base[2:3]
        self.enc2 = base[3:4]
        self.enc3 = base[4:6]
        self.bottleneck = base[6:]

        # --- Bottleneck compression ---
        self.bottleneck_compress = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(num_parameters=256),
        )

        # --- Decoder ---
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = UNetBlock(256 + 112, 160)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = UNetBlock(160 + 40, 80)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = UNetBlock(80 + 24, 40)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = UNetBlock(40 + 16, 32)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, num_classes, 1),
        )

        # --- Optional domain head ---
        self.use_domain_head = use_domain_head
        if use_domain_head:
            self.domain_head = DomainClassifier(in_features=256)

        # --- Kaiming init for decoder ---
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        decoder_modules = [
            self.bottleneck_compress,
            self.dec4, self.dec3, self.dec2, self.dec1,
            self.final,
        ]
        for module in decoder_modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    
                    nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_in',
                                            nonlinearity='leaky_relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_domain=False, domain_alpha=1.0):
        """
        Args:
            x: [B, 4, H, W] — RGB + NIR in [0,1].
               At RGB-only inference, zero-fill channel 3.
        """
        x = self.input_norm(x)
        x = self.moddrop(x)

        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        b = self.bottleneck_compress(b)

        domain_logits = None
        if return_domain and self.use_domain_head:
            domain_logits = self.domain_head(b, alpha=domain_alpha)

        x = self.up4(b)
        x = self.dec4(torch.cat([x, e3], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, e2], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, e1], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, e0], dim=1))
        logits = self.final(x)

        if return_domain:
            return logits, domain_logits
        return logits