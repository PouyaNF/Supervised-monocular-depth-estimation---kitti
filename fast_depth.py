import torch
import torch.nn.functional as F
import torch.nn as nn


class MobileNet(nn.Module):
    def __init__(self, relu6=True):
        super(MobileNet, self).__init__()

        def relu(relu6):
            if relu6:
                return nn.ReLU6(inplace=True)
            else:
                return nn.ReLU(inplace=True)

        def conv_bn(inp, oup, stride, relu6):  # conv_bn(  3,  32, 2, relu6),
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False), # bias is included in batch norm
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        def conv_dw(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(relu6),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2, relu6),
            conv_dw(32, 64, 1, relu6),
            conv_dw(64, 128, 2, relu6),
            conv_dw(128, 128, 1, relu6),
            conv_dw(128, 256, 2, relu6),
            conv_dw(256, 256, 1, relu6),
            conv_dw(256, 512, 2, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 1024, 2, relu6),
            conv_dw(1024, 1024, 1, relu6),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def depthwise(in_channels, kernel_size):
    padding = (kernel_size - 1) // 2
    assert 2 * padding == kernel_size - 1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size,
                  stride=1, padding=padding, bias=False, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
    )


def pointwise(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MobileNetSkipAdd(nn.Module):
    def __init__(self):
        super(MobileNetSkipAdd, self).__init__()

        mobilenet = MobileNet()
        path = './fast depth-pre_trained_encoder/mobilenet.pth'
        mobilenet.load_state_dict(torch.load(path))
        print('All keys matched successfully : fast_depth encoder')

        for i in range(14):
            setattr(self, 'conv{}'.format(i),
                    mobilenet.model[i])  # setattr(object, name, value) sets the value of the attribute of an object

        kernel_size = 5
        # self.decode_conv1 = conv(1024, 512, kernel_size)
        # self.decode_conv2 = conv(512, 256, kernel_size)
        # self.decode_conv3 = conv(256, 128, kernel_size)
        # self.decode_conv4 = conv(128, 64, kernel_size)
        # self.decode_conv5 = conv(64, 32, kernel_size)
        self.decode_conv1 = nn.Sequential(
            depthwise(1024, kernel_size),
            pointwise(1024, 512))
        self.decode_conv2 = nn.Sequential(
            depthwise(512, kernel_size),
            pointwise(512, 256))
        self.decode_conv3 = nn.Sequential(
            depthwise(256, kernel_size),
            pointwise(256, 128))
        self.decode_conv4 = nn.Sequential(
            depthwise(128, kernel_size),
            pointwise(128, 64))
        self.decode_conv5 = nn.Sequential(
            depthwise(64, kernel_size),
            pointwise(64, 32))
        self.decode_conv6 = pointwise(32, 1)

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            # print("{}: {}".format(i, x.size()))
            if i == 1:
                x1 = x  # save output of 2 , 4 and 6 layers for upsampling in decoder
            elif i == 3:
                x2 = x
            elif i == 5:
                x3 = x
        for i in range(1, 6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            # up samples the input to the given scale_factor . spatial size would be doubled
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i == 4:
                x = x + x1
            elif i == 3:
                x = x + x2
            elif i == 2:
                x = x + x3
            # print("{}: {}".format(i, x.size()))
        x = self.decode_conv6(x)
        return x


