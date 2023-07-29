import torch
from torch import nn
from torchvision import models
import torchvision.transforms as T


class InceptionV3Head(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, progress=False)
        self.net = nn.Sequential(
            pretrained.Conv2d_1a_3x3,
            pretrained.Conv2d_2a_3x3,
            pretrained.Conv2d_2b_3x3,
            pretrained.maxpool1,
            pretrained.Conv2d_3b_1x1,
            pretrained.Conv2d_4a_3x3,
            pretrained.maxpool2,
            pretrained.Mixed_5b,
            pretrained.Mixed_5c,
            pretrained.Mixed_5d,
            pretrained.Mixed_6a,
            pretrained.Mixed_6b,
            pretrained.Mixed_6c,
            pretrained.Mixed_6d,
            pretrained.Mixed_6e,
            pretrained.Mixed_7a,
            pretrained.Mixed_7b
        )

    def forward(self, x):
        x = self.net(x)
        return x


class InceptionV3Tail(nn.Module):
    def __init__(self, number_of_images, dropout=0.5):
        super(InceptionV3Tail, self).__init__()

        self.input_channels = number_of_images * 2048 + 1
        # self.Mixed_7b = models.inception.InceptionE(self.input_channels)
        self.Mixed_7c = models.inception.InceptionE(self.input_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, 1)

        self.metric_input = nn.Linear(1, 64)

    def _forward(self, x):
        # x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, images, metric):
        images = images.reshape(-1, self.input_channels - 1, 8, 8)
        metric_to_input = self.metric_input(metric)
        metric_to_input = metric_to_input.view(-1, 1, 8, 8)
        concat = torch.cat((images, metric_to_input), dim=1)
        outputs = self._forward(concat)
        return outputs
