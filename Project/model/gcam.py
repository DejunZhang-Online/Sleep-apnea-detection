import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(self.avg_pool(x).view(x.size(0), -1)).view(x.size(0), x.size(1), 1)
        return x * weights


class GCAM(nn.Module):

    def __init__(self, channel, group=8):
        super().__init__()

        self.cov1 = nn.Conv1d(channel, channel, kernel_size=1)
        self.cov2 = nn.Conv1d(channel, channel, kernel_size=1)
        self.group = group  # 分组数
        cam = []

        for i in range(self.group):
            cam_ = ChannelAttention(channel // group, reduction_ratio=8)
            cam.append(cam_)

        self.cam = nn.ModuleList(cam)
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x = self.cov1(x)

        y = torch.split(x, x.size(1) // self.group, dim=1)

        mask = []

        for y_, cam in zip(y, self.cam):
            y_ = cam(y_)
            y_ = self.sigomid(y_)

            mean = torch.mean(y_, [1, 2])
            mean = mean.view(-1, 1, 1)

            gate = torch.ones_like(y_) * mean
            mk = torch.where(y_ > gate, 1, y_)
            mask.append(mk)
        mask = torch.cat(mask, dim=1)

        x = x * mask
        x = self.cov2(x)
        x = x + x0
        return x


