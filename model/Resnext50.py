import torch.nn as nn
import torchvision
import torch

# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (19) and a Sigmoid instead of a default Softmax.


class Resnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = torchvision.models.resnext50_32x4d(
            weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
        resnet.fc = torch.nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features,
                      out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))
