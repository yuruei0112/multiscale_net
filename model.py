import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, groups, stride=1, padding=1, dw=True):
        super(BasicBlock, self).__init__()
        if dw:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=(3, 3), stride=stride, padding=padding, bias=False,
                          groups=groups),
                nn.Conv2d(mid_ch, out_ch, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.Hardswish(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class SeBlock(nn.Module):
    def __init__(self, inplane, plane):
        super(SeBlock, self).__init__()

        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplane, plane, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(plane, inplane, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.squeeze(x)


class MultiscaleBlock(nn.Module):
    def __init__(self, inchannel, stride=1, padding=1, dw=True, scale=4):
        super(MultiscaleBlock, self).__init__()
        self.vertical = []
        self.horizontal = []
        self.vertical.append(nn.Sequential(
            BasicBlock(in_ch=inchannel, mid_ch=inchannel, out_ch=inchannel, groups=inchannel, stride=stride,
                        padding=padding, dw=dw),
            BasicBlock(in_ch=inchannel, mid_ch=inchannel, out_ch=inchannel, groups=inchannel, stride=stride,
                        padding=padding, dw=dw)))
        for _ in range(1, scale):
            self.vertical.append(
                BasicBlock(in_ch=2 * inchannel, mid_ch=inchannel, out_ch=inchannel, groups=inchannel, stride=stride,
                            padding=padding, dw=dw))
        for _ in range(scale):
            self.horizontal.append(
                BasicBlock(in_ch=inchannel, mid_ch=inchannel, out_ch=inchannel, groups=inchannel, stride=stride,
                            padding=padding, dw=dw))
        self.vertical = nn.ModuleList(self.vertical)
        self.horizontal = nn.ModuleList(self.horizontal)
        self.scale = scale
        self.final = BasicBlock(in_ch=2 * inchannel, mid_ch=2 * inchannel, out_ch=2 * inchannel, groups=inchannel,
                                 stride=stride, padding=padding, dw=dw)
        self.squeeze = SeBlock(inplane=2 * inchannel, plane=6)
        self.maxpool = nn.MaxPool2d(kernel_size=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        out = x
        for i in range(self.scale):
            vertical = self.vertical[i](out)
            horizontal = self.horizontal[i](x)
            out = torch.concat((vertical, horizontal), 1)
        out = self.final(out)
        out = self.squeeze(out)
        out = self.maxpool(out)
        return out


class Multiscale_Net(nn.Module):
    def __init__(self, in_ch=3, plane=16, out_ch=6, multiscale_block=MultiscaleBlock, block_num=3):
        super(Multiscale_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, plane, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU(inplace=True),
        )
        multi_layers = [multiscale_block(inchannel=plane * (2 ** i), dw=True) for i in range(block_num)]
        self.multiblock = nn.Sequential(*multi_layers)

        self.global_depthwise_conv = nn.Sequential(
            nn.Conv2d(plane * (2 ** block_num), plane * (2 ** block_num), kernel_size=(16, 16), stride=(1, 1),
                      padding=0, groups=128, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01)
        )
        self.fc = nn.Sequential(
            nn.Linear(plane * (2 ** block_num), out_ch, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.multiblock(x)
        print(x.size())
        x = self.global_depthwise_conv(x)
        print(x.size())
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

