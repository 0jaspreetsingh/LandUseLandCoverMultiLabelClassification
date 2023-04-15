import torch
import torch.nn as nn
from .efficientnet import EfficientNet
from .efficientnetutils import get_channel_sizes
from .cbam import CBAMBlock
import numpy as np

class EsambleModel(nn.Module):
    def __init__(self, device, variant='efficientnet-b5', lbl_corr_vec=None, n_classes=19):
        super(EsambleModel, self).__init__()
        self.device = device
        variant_B10, variant_B20, variant_B60 = variant, 'efficientnet-b0', 'efficientnet-b0'
        channel_sizes_B10, channel_sizes_B20, channel_sizes_B60 = get_channel_sizes(
            variant_B10), get_channel_sizes(variant_B20), get_channel_sizes(variant_B60)
        self.b10_model = EfficientNet.from_name(variant_B10)
        self.b10_model._change_in_channels(4)
        self.b20_model = EfficientNet.from_name(variant_B20)
        self.b20_model._change_in_channels(6)
        # self.b60_model = Net()
        self.convReduction1 = nn.Sequential(
            nn.Conv2d(channel_sizes_B10[2], channel_sizes_B10[2], kernel_size=(
                4, 4), stride=(2, 2)),
            nn.BatchNorm2d(channel_sizes_B10[2]),
            nn.ReLU()
        )
        self.convReduction_B60 = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.cbam = CBAMBlock(
            channel=channel_sizes_B10[2] + channel_sizes_B20[2] +128, reduction=16, kernel_size=7)  # Kernel size is a
        # hyper-paramater
        self.register_buffer('lbl_corr_vec', lbl_corr_vec, persistent=False)
        self.lbl_layers = nn.Sequential(
            nn.BatchNorm1d(19*19 + channel_sizes_B10[2] + channel_sizes_B20[2]+128),
            nn.Linear(19*19 + channel_sizes_B10[2] + channel_sizes_B20[2] + 128, 64),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(
            channel_sizes_B10[2] + channel_sizes_B20[2]+128 + 128, n_classes)
        

    def forward(self, x):
        B10 , B20, B60 = x
        B10 = B10.to(self.device)
        B20 = B20.to(self.device)
        B60 = B60.to(self.device)
        eps_B10 = self.b10_model.extract_endpoints(B10)
        B10 = eps_B10.get('reduction_6')
        B10 = self.convReduction1(B10)
        eps_B20 = self.b20_model.extract_endpoints(B20)
        B20 = eps_B20.get('reduction_6')
        B60 = self.convReduction_B60(B60)
        x = torch.concat((B10, B20, B60), dim=1)
        x = self.cbam(x)
        x_pool = self.avg_pooling(x).squeeze()
        lc_in = torch.concat(
            (x_pool, self.lbl_corr_vec.repeat(x_pool.shape[0], 1)), dim=1)
        lc_out = self.lbl_layers(lc_in)
        x_pool = torch.concat((x_pool, lc_out), dim=1)
        x = self.dropout(x_pool).squeeze()
        x = self.fc(x)
        return x