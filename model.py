import torch.nn as nn
import torch


""" Optional conv block """
def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            Inception(64, 64),
            nn.ReLU(inplace=True),
        )
        self.maxpool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            Inception(192, 128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            Inception(384, 128),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.maxpool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.maxpool3(out)
        embedding_vector = out.view(out.size(0), -1)
        return embedding_vector
        

class Inception(nn.Module):
    def __init__(self, ch_in=1, ch_out=8):
        super(Inception, self).__init__()
        self.conv_d1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=2, dilation=2)
        self.conv_d3 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=3, dilation=3)

    def forward(self, x):
        x_d1 = self.conv_d1(x)
        x_d2 = self.conv_d2(x)
        x_d3 = self.conv_d3(x)
        out = torch.cat((x_d1, x_d2, x_d3), 1)
        return out