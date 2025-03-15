import torch
from gru import *

class FlowHead(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(FlowHead, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, hidden_channels, kernel_size = 3, stride = 1, padding = 1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(hidden_channels, 2, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class MotionEncoder(torch.nn.Module):
    def __init__(self, correlation_in_channels, correlation_hidden_channels, correlation_out_channels,
                 flow_hidden_channels, flow_out_channels, out_channels):
        super(MotionEncoder, self).__init__()
        self.correlation_conv = torch.nn.Sequential(
            torch.nn.Conv2d(correlation_in_channels, correlation_hidden_channels, kernel_size = 1, padding = 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(correlation_hidden_channels, correlation_out_channels, kernel_size = 3, padding = 1),
            torch.nn.ReLU(),
        )
        self.flow_conv = torch.nn.Sequential(
            torch.nn.Conv2d(2, flow_hidden_channels, kernel_size = 7, padding = 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(flow_hidden_channels, flow_out_channels, kernel_size = 3, padding = 1),
            torch.nn.ReLU(),
        )
        self.out_conv = torch.nn.Sequential(
            torch.nn.Conv2d(correlation_out_channels + flow_out_channels, out_channels - 2, kernel_size = 3, padding = 1),
            torch.nn.ReLU(),
        )
    def forward(self, correlation, flow):
        temp_correlation = self.correlation_conv(correlation)
        temp_flow = self.flow_conv(flow)
        temp = torch.cat([temp_correlation, temp_flow], dim = 1)
        out = self.out_conv(temp)
        cat_out = torch.cat([out, flow], dim = 1)
        # Return channels of 'out_channels'.
        return cat_out

class UpdateBlock(torch.nn.Module):
    def __init__(self, net_in_channels, inp_in_channels, corr_in_channels,
                 corr_hidden_channels, corr_out_channels,
                 flow_hidden_channels, flow_out_channels,
                 encoder_out_channels, mask_hidden_channels):
        super(UpdateBlock, self).__init__()
        self.encoder = MotionEncoder(corr_in_channels, corr_hidden_channels, corr_out_channels,
                                     flow_hidden_channels, flow_out_channels, encoder_out_channels)
        self.gru = SepConvGru(x_channels = inp_in_channels + encoder_out_channels,
                              h_channels = net_in_channels, kernel_size = 5)
        self.flow_head = FlowHead(net_in_channels, flow_out_channels)
        self.mask = torch.nn.Sequential(
            torch.nn.Conv2d(net_in_channels, mask_hidden_channels, kernel_size = 3, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(mask_hidden_channels, 8 * 8 * 9, kernel_size = 1, padding = 0),
        )
    def forward(self, net, inp, correlation, flow):
        motion = self.encoder(correlation, flow)
        inp_motion = torch.cat([inp, motion], dim = 1)
        new_net = self.gru(inp_motion, net)
        delta_flow = self.flow_head(new_net)
        mask = .25 * self.mask(new_net)
        return new_net, mask, delta_flow


if __name__ == '__main__':
    print('>> Test UpdateBlock:')
    net = torch.randn(5, 256, 32, 32)
    inp = torch.randn(5, 3, 32, 32)
    correlation = torch.randn(5, (2 * 3 + 1) * (2 * 3 + 1) * 4, 32, 32)
    flow = torch.randn(5, 2, 32, 32)
    model = UpdateBlock(
        net_in_channels = 256,
        inp_in_channels = 3,
        corr_in_channels = (2 * 3 + 1) * (2 * 3 + 1) * 4,
        corr_hidden_channels = 256,
        corr_out_channels = 192,
        flow_hidden_channels = 128,
        flow_out_channels = 64,
        encoder_out_channels = 128,
        mask_hidden_channels = 256,
    )
    new_net, mask, delta_flow = model(net, inp, correlation, flow)
    print('new_net.size():', new_net.size())
    print('mask.size():', mask.size())
    print('delta_flow.size():', delta_flow.size())
    print('Done.')
