import torch
from torch import nn


class FullyConnectedNet(nn.Module):
    def __init__(self, channels_in=1066):
        super(FullyConnectedNet, self).__init__()
        channels = 3000
        self.hid1 = nn.Linear(channels_in, channels)
        self.block1 = Block(channels)
        self.block2 = Block(channels)
        self.block3 = Block(channels)
        self.block4 = Block(channels)
        self.block5 = Block(channels)
        self.block6 = Block(channels)
        self.block7 = Block(channels)
        self.block8 = Block(channels)
        self.block9 = Block(channels)
        self.block10 = Block(channels)
        self.block11 = Block(channels)
        self.block12 = Block(channels)
        self.block13 = Block(channels)
        self.block14 = Block(channels)
        self.block15 = Block(channels)
        self.block16 = Block(channels)
        self.block17 = Block(channels)
        self.output = nn.Linear(channels, 1)
        self.output_aux = nn.Linear(channels, 1)
        self.dropout = nn.Dropout(0.15)
    
    def forward(self, x):
        x = torch.relu(self.hid1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        aux = self.output_aux(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.output(x)
        return x, aux
    
    def my_net_name(self):
        return "nn-with-skip-connection"


class Block(nn.Module):
    def __init__(self, channels_in, channels_out=None):
        super(Block, self).__init__()
        if channels_out is None:
            channels_out = channels_in
        self.block = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(channels_in, channels_out),
        )
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        out = torch.relu(out)
        return out
