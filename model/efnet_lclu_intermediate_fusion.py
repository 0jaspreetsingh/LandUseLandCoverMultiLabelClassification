import torch
import torch.nn as nn
from .efficientnet import EfficientNet
from .efficientnetutils import get_channel_sizes
from .cbam import CBAMBlock
import numpy as np

class IntermediateFustionModel(nn.Module):
    def __init__(self,device, variant='efficientnet-b5', lbl_corr_vec=None, n_channels=4, n_classes=19):
        super(IntermediateFustionModel, self).__init__()
        self.device = device
        channel_sizes = get_channel_sizes(variant)
        self.model = EfficientNet.from_name(variant)
        self.model._change_in_channels(n_channels)
        self.convReduction1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0] + 6, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.convReduction2 = nn.Sequential(
            nn.Conv2d(channel_sizes[1] + 64 + 2, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        in_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear( in_ftrs + 128 + 128, n_classes)
        self.cbam = CBAMBlock(channel=channel_sizes[2] + 128, reduction=16, kernel_size=7)  # Kernel size is a
        # hyper-paramater

        self.register_buffer('lbl_corr_vec', lbl_corr_vec, persistent=False)
        # self.lbl_corr_vec = lbl_corr_vec

        self.lbl_layers = nn.Sequential(
            nn.BatchNorm1d(19*19 + in_ftrs + 128),
            nn.Linear(19*19 + in_ftrs + 128, 64),
            nn.Linear(64, 128),
            nn.ReLU()
        )


    def forward(self, x):
        B10 , B20, B60 = x
        B10 = B10.to(self.device)
        B20 = B20.to(self.device)
        B60 = B60.to(self.device)

        eps = self.model.extract_endpoints(B10)
        x, x1half, x2half = eps.get('reduction_6'), eps.get('reduction_1'), eps.get('reduction_2')
        x1half = torch.concat((x1half , B20), dim=1)
        x2half = torch.concat((x2half, self.convReduction1(x1half)), dim=1)
        x2half = torch.concat((x2half , B60), dim=1)
        x = torch.concat((x, self.convReduction2(x2half)), dim=1)

        x = self.cbam(x)
        x_pool = self.model._avg_pooling(x).squeeze()
        lc_in = torch.concat((x_pool, self.lbl_corr_vec.repeat(x_pool.shape[0], 1)), dim=1)
        lc_out = self.lbl_layers(lc_in)
        x_pool = torch.concat((x_pool, lc_out), dim=1)
        x = self.model._dropout(x_pool).squeeze()
        x = self.model._fc(x)
        return x
