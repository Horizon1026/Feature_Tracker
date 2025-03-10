import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if stride != 1 or in_channels != out_channels else nn.Identity()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        channel_step = out_channels // 4
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, channel_step, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
        )
        self.resnet_1 = nn.Sequential(
            ResNetBlock(channel_step, channel_step),
            ResNetBlock(channel_step, channel_step * 2, stride = 2),
        )
        self.resnet_2 = nn.Sequential(
            ResNetBlock(channel_step * 2, channel_step * 2),
            ResNetBlock(channel_step * 2, channel_step * 3, stride = 2),
        )
        self.resnet_3 = nn.Sequential(
            ResNetBlock(channel_step * 3, channel_step * 3),
            ResNetBlock(channel_step * 3, out_channels, stride = 2),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, img):
        x = self.conv_in(img)
        x = self.resnet_1(x)
        x = self.resnet_2(x)
        x = self.resnet_3(x)
        x = self.conv_out(x)
        return x

class ContextEncoder(nn.Module):
    def __init__(self, in_channels, context_channels, hidden_channels):
        super().__init__()
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        self.net = FeatureEncoder(in_channels, context_channels + hidden_channels)
    def forward(self, img):
        x = self.net(img)
        # Split the output on the channel dimension.
        context, hidden = torch.split(x, [self.context_channels, self.hidden_channels], dim = -3)
        return context, hidden

# Test the model.
if __name__ == '__main__':
    img = torch.randn(5, 1, 224, 224)

    print('>> Test FeatureEncoder:')
    feature_encoder = FeatureEncoder(1, 128)
    out = feature_encoder(img)
    print('out.size():', out.size())

    print('>> Test ContextEncoder:')
    context_encoder = ContextEncoder(1, 64, 64)
    context, hidden = context_encoder(img)
    print('context.size():', context.size())
    print('hidden.size():', hidden.size())
