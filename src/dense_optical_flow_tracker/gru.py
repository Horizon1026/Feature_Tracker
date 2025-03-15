import torch

# Standard GRU.
class Gru(torch.nn.Module):
    def __init__(self, x_channels, h_channels):
        super(Gru, self).__init__()
        self.fc_z = torch.nn.Linear(x_channels + h_channels, h_channels)
        self.fc_r = torch.nn.Linear(x_channels + h_channels, h_channels)
        self.fc_q = torch.nn.Linear(x_channels + h_channels, h_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
    def forward(self, x, h):
        # x: input tensor.
        # h: hidden state tensor, i.e., previous output.
        xh = torch.cat([x, h], dim = 1)
        z = self.sigmoid(self.fc_z(xh))
        r = self.sigmoid(self.fc_r(xh))
        xh = torch.cat([x, r * h], dim = 1)
        q = self.tanh(self.fc_q(xh))
        h = (1 - z) * h + z * q
        return h

# Standard convolutional GRU.
class ConvGru(torch.nn.Module):
    def __init__(self, x_channels, h_channels, kernel_size):
        super(ConvGru, self).__init__()
        assert kernel_size % 2 == 1, 'Kernel size must be odd.'
        padding = kernel_size // 2
        self.conv_z = torch.nn.Conv2d(x_channels + h_channels, h_channels, kernel_size, stride = 1, padding = padding)
        self.conv_r = torch.nn.Conv2d(x_channels + h_channels, h_channels, kernel_size, stride = 1, padding = padding)
        self.conv_q = torch.nn.Conv2d(x_channels + h_channels, h_channels, kernel_size, stride = 1, padding = padding)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
    def forward(self, x, h):
        # x: input tensor.
        # h: hidden state tensor, i.e., previous output.
        xh = torch.cat([x, h], dim = 1)
        z = self.sigmoid(self.conv_z(xh))
        r = self.sigmoid(self.conv_r(xh))
        xh = torch.cat([x, r * h], dim = 1)
        q = self.tanh(self.conv_q(xh))
        h = (1 - z) * h + z * q
        return h

# Separable convolutional GRU.
class SepConvGru(torch.nn.Module):
    def __init__(self, x_channels, h_channels, kernel_size):
        super(SepConvGru, self).__init__()
        assert kernel_size % 2 == 1, 'Kernel size must be odd.'
        padding = kernel_size // 2
        self.conv_z_horizontal = torch.nn.Conv2d(x_channels + h_channels, h_channels, (1, kernel_size), stride = 1, padding = (0, padding))
        self.conv_r_horizontal = torch.nn.Conv2d(x_channels + h_channels, h_channels, (1, kernel_size), stride = 1, padding = (0, padding))
        self.conv_q_horizontal = torch.nn.Conv2d(x_channels + h_channels, h_channels, (1, kernel_size), stride = 1, padding = (0, padding))
        self.conv_z_vertical = torch.nn.Conv2d(x_channels + h_channels, h_channels, (kernel_size, 1), stride = 1, padding = (padding, 0))
        self.conv_r_vertical = torch.nn.Conv2d(x_channels + h_channels, h_channels, (kernel_size, 1), stride = 1, padding = (padding, 0))
        self.conv_q_vertical = torch.nn.Conv2d(x_channels + h_channels, h_channels, (kernel_size, 1), stride = 1, padding = (padding, 0))
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
    def forward(self, x, h):
        # x: input tensor.
        # h: hidden state tensor, i.e., previous output.
        # Process horizontal direction.
        xh = torch.cat([x, h], dim = 1)
        z_horizontal = self.sigmoid(self.conv_z_horizontal(xh))
        r_horizontal = self.sigmoid(self.conv_r_horizontal(xh))
        xh_horizontal = torch.cat([x, r_horizontal * h], dim = 1)
        q_horizontal = self.tanh(self.conv_q_horizontal(xh_horizontal))
        h = (1 - z_horizontal) * h + z_horizontal * q_horizontal
        # Process vertical direction.
        xh = torch.cat([x, h], dim = 1)
        z_vertical = self.sigmoid(self.conv_z_vertical(xh))
        r_vertical = self.sigmoid(self.conv_r_vertical(xh))
        xh_vertical = torch.cat([x, r_vertical * h], dim = 1)
        q_vertical = self.tanh(self.conv_q_vertical(xh_vertical))
        h = (1 - z_vertical) * h + z_vertical * q_vertical
        return h


if __name__ == '__main__':
    batch_size = 5
    x_channels = 3
    h_channels = 16

    x = torch.randn(batch_size, x_channels)
    h = torch.randn(batch_size, h_channels)
    print('>> Test Gru:')
    gru = Gru(x_channels = x_channels, h_channels = h_channels)
    print('Gru output size:', gru(x, h).size())

    x = torch.randn(batch_size, x_channels, 224, 224)
    h = torch.randn(batch_size, h_channels, 224, 224)
    print('>> Test ConvGru:')
    conv_gru = ConvGru(x_channels = x_channels, h_channels = h_channels, kernel_size = 5)
    print('ConvGru output size:', conv_gru(x, h).size())
    print('>> Test SepConvGru:')
    sep_conv_gru = SepConvGru(x_channels = x_channels, h_channels = h_channels, kernel_size = 5)
    print('SepConvGru output size:', sep_conv_gru(x, h).size())
    print('Done.')
