"""Offline class-conditional diffusion sampler for 32×32 RGB images."""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


NOISE_STEPS = 1000
NULL_LABEL = 2
SAMPLE_BATCH = 32
CHECKPOINT = Path(__file__).with_name("weights.pt")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(embed_dim, 2 * out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x, embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(F.silu(embedding))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        return self.skip(x) + self.conv2(F.silu(h))


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x):
        batch, channels, height, width = x.shape
        seq = self.norm(x).flatten(2).transpose(1, 2)
        qkv = self.qkv(seq).view(batch, height * width, 3, 4, channels // 4)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, height * width, channels)
        return x + self.out(attn).transpose(1, 2).reshape_as(x)


class Diffusion(nn.Module):
    def __init__(self, width=64, sampling_steps=32, guidance_scale=1.5):
        super().__init__()
        self.sampling_steps = sampling_steps
        self.guidance_scale = guidance_scale
        width8, width4, embed_dim = 2 * width, 3 * width, 4 * width
        freq = torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2))
        self.register_buffer("frequencies", freq, persistent=False)
        self.register_buffer("alpha_bar", cosine_schedule(), persistent=False)
        self.time_mlp = nn.Sequential(nn.Linear(width, embed_dim), nn.SiLU(),
                                      nn.Linear(embed_dim, embed_dim))
        self.classes = nn.Embedding(3, embed_dim)
        self.stem = nn.Conv2d(12, width, 3, padding=1)
        self.enc16 = ResidualBlock(width, width, embed_dim)
        self.down8 = nn.Conv2d(width, width8, 4, stride=2, padding=1)
        self.enc8 = ResidualBlock(width8, width8, embed_dim)
        self.down4 = nn.Conv2d(width8, width4, 4, stride=2, padding=1)
        self.middle1 = ResidualBlock(width4, width4, embed_dim)
        self.attention = SpatialAttention(width4)
        self.middle2 = ResidualBlock(width4, width4, embed_dim)
        self.up8 = nn.Conv2d(width4, width8, 3, padding=1)
        self.dec8 = ResidualBlock(2 * width8, width8, embed_dim)
        self.up16 = nn.Conv2d(width8, width, 3, padding=1)
        self.dec16 = ResidualBlock(2 * width, width, embed_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 12, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, noisy, timesteps, labels):
        phase = timesteps[:, None].float() * self.frequencies[None]
        embedding = self.time_mlp(torch.cat((phase.sin(), phase.cos()), dim=1)) + self.classes(labels)
        high = self.enc16(self.stem(noisy), embedding)
        low = self.enc8(self.down8(high), embedding)
        h = self.middle1(self.down4(low), embedding)
        h = self.middle2(self.attention(h), embedding)
        h = self.up8(F.interpolate(h, size=(8, 8), mode="nearest"))
        h = self.dec8(torch.cat((h, low), dim=1), embedding)
        h = self.up16(F.interpolate(h, size=(16, 16), mode="nearest"))
        return self.out(self.dec16(torch.cat((h, high), dim=1), embedding))


def cosine_schedule():
    t = torch.arange(NOISE_STEPS + 1, dtype=torch.float64) / NOISE_STEPS
    target = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    betas = (1 - target[1:] / target[:-1]).clamp(max=0.999)
    return torch.cat((torch.ones(1, dtype=torch.float64), (1 - betas).cumprod(0))).float()


def noisy_with_velocity(clean, noise, alpha_bar):
    """Map (clean, noise) to the noisy image and its velocity target."""
    a, b = alpha_bar.sqrt(), (1 - alpha_bar).sqrt()
    return a * clean + b * noise, a * noise - b * clean


def guided_velocity(model, noisy, t, labels):
    if model.guidance_scale == 1:
        return model(noisy, t, labels)
    unconditional, conditional = model(
        torch.cat((noisy, noisy)), torch.cat((t, t)),
        torch.cat((torch.full_like(labels, NULL_LABEL), labels)),
    ).chunk(2)
    return torch.lerp(unconditional, conditional, model.guidance_scale)


def ddim(model, noisy, labels, time_grid):
    for current, previous in zip(time_grid[:-1], time_grid[1:], strict=True):
        t = torch.full_like(labels, current)
        a = model.alpha_bar[current].sqrt()
        b = (1 - model.alpha_bar[current]).sqrt()
        velocity = guided_velocity(model, noisy, t, labels)
        clean = (a * noisy - b * velocity).clamp(-1, 1)
        noise = (noisy - a * clean) / b
        prior_alpha = model.alpha_bar[previous]
        noisy = prior_alpha.sqrt() * clean + (1 - prior_alpha).sqrt() * noise
    return F.pixel_shuffle(noisy, 2).mul_(0.5).add_(0.5).clamp_(0, 1)


def load_model(device):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = Diffusion(checkpoint["width"], checkpoint["sampling_steps"], checkpoint["guidance_scale"])
    model.load_state_dict(checkpoint["state"])
    return model.to(device).eval()


@torch.no_grad()
def generate(model, labels, seed):
    if labels.numel() == 0:
        return torch.empty((0, 3, 32, 32), device=labels.device)
    time_grid = torch.linspace(NOISE_STEPS, 0, model.sampling_steps + 1).round().long().tolist()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = []
    for batch in labels.split(SAMPLE_BATCH):
        # Fixed chunk sizes preserve the noise stream and per-image numerical
        # results when unrelated labels are changed or appended.
        noise = torch.randn((SAMPLE_BATCH, 12, 16, 16), generator=generator, device="cpu")
        padded = F.pad(batch, (0, SAMPLE_BATCH - len(batch)), value=NULL_LABEL)
        images.append(ddim(model, noise.to(batch.device), padded, time_grid)[:len(batch)])
    return torch.cat(images)
