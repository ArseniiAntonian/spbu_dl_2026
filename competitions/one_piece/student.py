"""Собственная компактная CNN (ученик): residual-блоки из двух 3x3 свёрток с Squeeze-and-Excitation.

Около 1.2M параметров, веса в fp32 занимают < 5 МБ. Depthwise-свёртки (MobileNet) не используются:
на Apple MPS они в ~8 раз медленнее обычных при том же числе параметров.
"""

import torch
from torch import nn


class SqueezeExcite(nn.Module):
    """Взвешивает каналы по глобальному контексту: GAP -> узкий MLP -> sigmoid."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


def conv_bn(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, act: bool = True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False), nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """3x3 -> 3x3 -> SE + shortcut (1x1 свёртка, если меняется размер)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, drop_path: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(conv_bn(in_ch, out_ch, 3, stride), conv_bn(out_ch, out_ch, 3, act=False), SqueezeExcite(out_ch))
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.AvgPool2d(stride) if stride > 1 else nn.Identity(), conv_bn(in_ch, out_ch, 1, act=False))
        self.act = nn.SiLU(inplace=True)
        self.drop_path = drop_path

    def forward(self, x):
        out = self.body(x)
        if self.training and self.drop_path > 0:
            keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path
            out = out * keep / (1 - self.drop_path)
        return self.act(out + self.shortcut(x))


# (out_channels, blocks, stride)
STAGES = [(40, 1, 2), (80, 2, 2), (128, 2, 2), (144, 1, 2)]


class OnePieceNet(nn.Module):
    def __init__(self, num_classes: int, drop_path: float = 0.05, dropout: float = 0.2):
        super().__init__()
        n_blocks = sum(stage[1] for stage in STAGES)
        layers = [conv_bn(3, 24, 3, stride=2), conv_bn(24, 32, 3)]
        in_ch, idx = 32, 0
        for out_ch, blocks, stride in STAGES:
            for i in range(blocks):
                layers.append(ResidualBlock(in_ch, out_ch, stride if i == 0 else 1, drop_path * idx / n_blocks))
                in_ch, idx = out_ch, idx + 1
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        # последний BN в ветке блока обнуляем: в начале обучения блок близок к shortcut
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.zeros_(m.body[1][1].weight)

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    model = OnePieceNet(18)
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    print(f"params={n_params / 1e6:.3f}M, fp32 state_dict ~ {(n_params + n_buffers) * 4 / 2**20:.2f} MB")
