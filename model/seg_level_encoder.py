import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcam import GCAM


class ConvBlock1D(nn.Module):
    """
    可选 BatchNorm + RevIN + Dropout 的一维卷积块
    顺序：Conv -> BN? -> ReLU -> Pool -> RevIN -> Dropout
    """

    def __init__(self, in_ch, out_ch, kernel_size=11, stride=1, pool_k=3, pool_s=3, p_drop=0.2):
        super().__init__()
        pad = kernel_size // 2
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, stride=stride), nn.BatchNorm1d(out_ch)]
        layers += [nn.GELU()]
        if pool_k is not None and pool_s is not None:
            layers.append(nn.MaxPool1d(pool_k, pool_s))
        layers.append(nn.Dropout(p_drop))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Seg_Encoder(nn.Module):
    def __init__(self):
        super(Seg_Encoder, self).__init__()
        # 下采样与特征提取
        self.RRI_conv1 = nn.Sequential(
            ConvBlock1D(1, 16, kernel_size=11, stride=1, pool_k=None, pool_s=None, p_drop=0.0),
            ConvBlock1D(16, 24, kernel_size=11, stride=2, pool_k=3, pool_s=3, p_drop=0.0),
            # SpeKAN(24, 1.0)
        )
        self.RPA_conv1 = nn.Sequential(
            ConvBlock1D(1, 16, kernel_size=11, stride=1, pool_k=None, pool_s=None, p_drop=0.0),
            ConvBlock1D(16, 24, kernel_size=11, stride=2, pool_k=3, pool_s=3, p_drop=0.0),
            # SpeKAN(24)
        )

        self.RRI_conv2 = nn.Sequential(
            ConvBlock1D(24, 32, kernel_size=11, stride=1, pool_k=5, pool_s=5, p_drop=0.0),
            # SpeKAN(32)
        )
        self.RPA_conv2 = nn.Sequential(
            ConvBlock1D(24, 32, kernel_size=11, stride=1, pool_k=5, pool_s=5, p_drop=0.0),
            # SpeKAN(32)
        )

        self.FSN_conv2 = nn.Sequential(
            ConvBlock1D(48, 32, kernel_size=11, stride=1, pool_k=5, pool_s=5, p_drop=0.0),
            # SpeKAN(32)
        )

        self.RRI_RPA_fsn = nn.Sequential(
            ConvBlock1D(32, 32, kernel_size=11, stride=1, pool_k=5, pool_s=5, p_drop=0.0),
            # SpeKAN(32)
        )

        # Match the paper setting by expanding the fused feature map to 128 channels
        # before GCAM, then applying grouped channel attention with g=4.
        self.pre_gcam_conv = ConvBlock1D(64, 128, kernel_size=1, stride=1, pool_k=None, pool_s=None, p_drop=0.0)
        self.gcbam = GCAM(channel=128, group=4)

    def forward(self, x1, x2):
        x1 = self.RRI_conv1(x1)  # [B, 24, L1]
        x2 = self.RPA_conv1(x2)  # [B, 24, L1]

        fsn1 = torch.cat([x1, x2], dim=1)  # [B, 48, L1]
        fsn2 = self.FSN_conv2(fsn1)  # [B, 32, L2]

        x1 = self.RRI_conv2(x1)  # [B, 32, L2]
        x2 = self.RPA_conv2(x2)  # [B, 32, L2]

        # 稀疏注意力
        fsn3 = F.scaled_dot_product_attention(fsn2, fsn2, fsn2)
        x1 = F.scaled_dot_product_attention(fsn3, x1, x1)
        x2 = F.scaled_dot_product_attention(fsn3, x2, x2)

        concat = torch.cat([x1, x2], dim=1)  # [B, 64, L2]
        concat = self.pre_gcam_conv(concat)  # [B, 128, L2]
        scale = self.gcbam(concat)  # [B, 128, L2]

        return scale


if __name__ == '__main__':
    testData1 = torch.randn(2, 1, 900)
    testData2 = torch.randn(2, 1, 900)
    model = Seg_Encoder()
    out = model(testData1, testData2)
    print(out.shape)
