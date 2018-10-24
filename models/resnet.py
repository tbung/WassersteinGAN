import torch.nn as nn
import torch.nn.functional as F

DIM_G = 128
DIM_D = 128


class GenResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenResBlock, self).__init__()

        self.layers_pre = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.layers_post = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = F.interpolate(x, scale_factor=2)
        residual = self.residual(residual)

        out = self.layers_pre(x)

        out = F.interpolate(out, scale_factor=2)

        out = self.layers_post(out)

        return out + residual


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()

        self.nz = nz

        self.linear = nn.Linear(nz, nz * 4 * 4)

        self.layers = nn.Sequential(
            GenResBlock(nz, nz),
            GenResBlock(nz, nz),
            GenResBlock(nz, nz),
            nn.BatchNorm2d(nz),
            nn.ReLU(),
            nn.Conv2d(nz, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.reshape(-1, self.nz)
        out = self.linear(z)

        out = out.reshape(-1, self.nz, 4, 4)

        out = self.layers(out)

        return out


class DiscResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super(DiscResBlock, self).__init__()
        self.pool = pool
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels:
            residual = self.residual(residual)
        if self.pool:
            residual = F.avg_pool2d(residual, kernel_size=2)

        out = self.layers(x)

        if self.pool:
            out = F.avg_pool2d(out, kernel_size=2)

        return out + residual


class Discriminator(nn.Module):
    def __init__(self, nz):
        super(Discriminator, self).__init__()

        self.initial_res = nn.Sequential(
            nn.Conv2d(3, nz, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nz, nz, kernel_size=3, padding=1),
            nn.AvgPool2d(2)
        )
        self.init_skip = nn.Sequential(
            nn.Conv2d(3, nz, kernel_size=1),
            nn.AvgPool2d(2)
        )

        self.layers = nn.Sequential(
            DiscResBlock(nz, nz, pool=True),
            DiscResBlock(nz, nz, pool=False),
            DiscResBlock(nz, nz, pool=False),
            nn.ReLU(),
        )

        self.linear = nn.Linear(nz, 1)

    def forward(self, x):
        out = self.initial_res(x)
        out += self.init_skip(x)

        out = self.layers(out)

        out = out.mean(dim=-1).mean(dim=-1)

        return self.linear(out)
