import torch.nn as nn
import torchvision
import torch
# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (19) and a Sigmoid instead of a default Softmax.


class Resnet(nn.Module):
    def __init__(self, device, n_classes, n_channels):
        super().__init__()
        self.device = device
        resnet = torchvision.models.resnet18()
        resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        resnet.fc = torch.nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features,
                      out_features=n_classes)
        )
        self.base_model = resnet

    def forward(self, x):
        x = x.to(self.device)
        return self.base_model(x)
