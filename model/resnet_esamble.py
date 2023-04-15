import torch.nn as nn
import torchvision
import torch

# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (19) and a Sigmoid instead of a default Softmax.


class ResnetEsamble(nn.Module):
    def __init__(self, device, n_classes):
        super().__init__()
        self.device = device
        self.resnet_B10 = torchvision.models.resnet18()
        self.resnet_B10.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.resnet_B10.fc = torch.nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=self.resnet_B10.fc.in_features,
        #               out_features=n_classes)
        # )

        self.resnet_B20 = torchvision.models.resnet18()
        self.resnet_B20.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.resnet_B20.fc = torch.nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=self.resnet_B20.fc.in_features,
        #               out_features=n_classes)
        # )

        self.convReduction_B60 = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc = torch.nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.resnet_B10.fc.out_features + self.resnet_B20.fc.out_features + 128,
                      out_features=n_classes)
        )


    def forward(self, x):
        B10 , B20, B60 = x
        B10 = B10.to(self.device)
        B20 = B20.to(self.device)
        B60 = B60.to(self.device)

        B10 = self.resnet_B10(B10)
        B20 = self.resnet_B20(B20)
        B60 = self.convReduction_B60(B60)

        B60_dims = B60.shape
        B60 = B60.view(B60_dims[0],-1)

        output = torch.concat((B10, B20, B60), dim=1)

        return self.fc(output)


