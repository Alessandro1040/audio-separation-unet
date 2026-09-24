"""U-Net: the image-segmentation network doing all the work.

Input : log-magnitude spectrogram of the mixture, shape (B, 2, F, T)  ("the photo")
Output: one 2-D map per source and channel channel, i.e. a *mask* per source
        (B, 4 sources * 2 channels * 2 (real/imag), F, T)  ("the segmentation")

The classic U-Net shape (Ronneberger et al., 2015) is used: an encoder that
downsamples and captures context, a decoder that upsamples back to the original
resolution, and skip connections that inject the fine detail (note onsets, harmonics)
lost while downsampling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(norm: str, channels: int, groups: int = 8) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        return nn.GroupNorm(min(groups, channels), channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"unknown norm '{norm}'")


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with normalisation and ReLU (the U-Net workhorse)."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=norm == "instance"),
            make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=norm == "instance"),
            make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Strided convolution (learned pooling) followed by a ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            ConvBlock(out_ch, out_ch, norm, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upsample to the size of the matching encoder feature map and fuse it (skip)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "batch",
                 up_mode: str = "bilinear", dropout: float = 0.0):
        super().__init__()
        if up_mode == "transpose":
            self.up: nn.Module = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        else:
            # 1x1 projection so the upsampled tensor can be concatenated with the skip
            self.up = nn.Conv2d(in_ch, out_ch, 1)
        self.up_mode = up_mode
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, norm, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.up_mode == "bilinear":
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        else:
            x = self.up(x)
            if x.shape[-2:] != skip.shape[-2:]:  # odd frequency bins
                x = x[..., : skip.shape[-2], : skip.shape[-1]]
        x = self.up(x) if self.up_mode == "bilinear" else x
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """U-Net that predicts one mask per (source, channel)."""

    def __init__(
        self,
        in_channels: int = 2,
        n_sources: int = 4,
        out_per_source: int = 4,   # 2 channels x (real, imag)
        base_channels: int = 32,
        depth: int = 5,
        norm: str = "batch",
        up_mode: str = "bilinear",
        mask: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert depth >= 2, "need at least one down and one up stage"
        self.n_sources = n_sources
        self.out_per_source = out_per_source
        self.mask = mask
        self.out_channels = n_sources * out_per_source

        widths = [base_channels * 2**min(i, 3) for i in range(depth)]
        self.stem = ConvBlock(in_channels, widths[0], norm, dropout)

        self.downs = nn.ModuleList(
            [Down(widths[i], widths[i + 1], norm, dropout) for i in range(depth - 1)]
        )
        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2, norm, dropout)
        widths[-1] = widths[-1] * 2

        self.ups = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            in_ch = widths[i + 1]
            self.ups.append(
                Up(in_ch, widths[i], widths[i], norm, up_mode, dropout)
            )

        self.head = nn.Conv2d(widths[0], self.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.stem(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        masks = self.head(x)
        if self.mask == "tanh":
            return torch.tanh(masks)
        if self.mask == "sigmoid":
            return torch.sigmoid(masks)
        return masks
