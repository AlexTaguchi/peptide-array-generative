import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_multiple(x, multiple=8):
    """Pad image height and width to be divisible by set multiple.

    Args:
        x (torch.Tensor): Input image, (N, K, H, W).
        multiple (int, optional): Multiple to pad to. Defaults to 8.

    Returns:
        tuple: Tuple containing the padded image, the height padding, and the width padding
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), pad_h, pad_w


def unpad(x, pad_h, pad_w):
    """Unpad image height and width to be divisible by set multiple.

    Args:
        x (torch.Tensor): Input image, (N, K, H, W).
        pad_h (int): Height padding.
        pad_w (int): Width padding.

    Returns:
        torch.Tensor: Unpadded image, (N, K, H, W).
    """
    return x[:, :, :-pad_h if pad_h > 0 else None, :-pad_w if pad_w > 0 else None]


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.residual = residual
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        return F.gelu(x + self.double_conv(x)) if self.residual else self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        x += self.embedding_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        x += self.embedding_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x


class UNet(nn.Module):
    def __init__(self, channels=3, time_embedding_dim=256, conditional_dim=None):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.inc = DoubleConv(channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)

        self.bot1 = DoubleConv(256, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 256)

        self.up1 = Up(256 + 256, 128)
        self.up2 = Up(128 + 128, 64)
        self.up3 = Up(64 + 64, 64)

        self.outc = nn.Conv2d(64, channels, kernel_size=1)

        if conditional_dim is not None:
            self.conditional_embedding = nn.Linear(conditional_dim, time_embedding_dim)

    def positionally_encode_time(self, t):
        """Positionally encode time data.

        Args:
            t (torch.Tensor): Time data, (N,).

        Returns:
            torch.Tensor: Positionally encoded time data, (N, channels).
        """
        model_device = next(self.parameters()).device
        time_range = torch.arange(0, self.time_embedding_dim, 2, device=model_device).float()
        inverse_frequencies = 1.0 / (10000 ** (time_range / self.time_embedding_dim))
        encoding_a = torch.sin(t.repeat(1, self.time_embedding_dim // 2) * inverse_frequencies)
        encoding_b = torch.cos(t.repeat(1, self.time_embedding_dim // 2) * inverse_frequencies)
        encoding = torch.cat([encoding_a, encoding_b], dim=-1)
        return encoding

    def forward(self, x, t, y=None):
        """Forward pass of UNet with optional conditioning.

        Args:
            x (torch.Tensor): Input image, (N, H, W, K).
            t (torch.Tensor): Time data, (N,).
            y (torch.Tensor, optional): Conditional data, (N, C). Defaults to None.

        Returns:
            torch.Tensor: Output image, (N, H, W, K)
        """
        x = x.permute(0, 3, 1, 2)

        # Embed time data
        t = t.unsqueeze(-1)
        t = self.positionally_encode_time(t)

        # Embed conditional data
        if y is not None:
            t += self.conditional_embedding(y.float())

        # Pad input to nearest multiple of 8
        x, pad_h, pad_w = pad_to_multiple(x, multiple=8)

        # Encode image by downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        # Build latent representation
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Decode image by upsampling
        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)

        # Output decoded image
        x = self.outc(x)
        x = unpad(x, pad_h, pad_w)
        x = x.permute(0, 2, 3, 1)

        # Apply softmax for categorical data
        x = torch.softmax(x, dim=-1)

        return x