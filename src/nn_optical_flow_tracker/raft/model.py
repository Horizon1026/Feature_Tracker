import torch
from encoder import *
from correlation_volumes import *
from update_block import *

class Raft(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, feature_channels, context_channels,
                 correlation_pyramid_levels, correlation_radius,
                 correlation_hidden_channels, correlation_out_channels,
                 flow_hidden_channels, flow_out_channels,
                 motion_out_channels, mask_hidden_channels,
                 max_iterations = 12):
        super(Raft, self).__init__()
        self.hidden_dim = hidden_channels   # Dimension of h.
        self.context_dim = context_channels # Dimension of x.
        self.correlation_pyramid_levels = correlation_pyramid_levels
        self.correlation_radius = correlation_radius
        self.max_iterations = max_iterations

        self.feature_encoder = FeatureEncoder(in_channels, feature_channels)
        self.context_encoder = ContextEncoder(in_channels, context_channels, hidden_channels)
        self.update_block = UpdateBlock(
            net_in_channels = hidden_channels,
            inp_in_channels = context_channels,
            corr_in_channels = correlation_pyramid_levels * ((2 * correlation_radius + 1) ** 2),
            corr_hidden_channels = correlation_hidden_channels,
            corr_out_channels = correlation_out_channels,
            flow_hidden_channels = flow_hidden_channels,
            flow_out_channels = flow_out_channels,
            motion_out_channels = motion_out_channels,
            mask_hidden_channels = mask_hidden_channels,
        )

    def InitializeFlow(self, flow_size):
        batch_size, _, height, width = flow_size
        # Initialize pixel locations of the reference image. Iterating each pixel over the height and width of the image.
        ref_pixel_locations = torch.meshgrid([torch.arange(height), torch.arange(width)], indexing = 'ij')
        ref_pixel_locations = torch.stack(ref_pixel_locations[::-1], dim = 0).float()
        ref_pixel_locations = ref_pixel_locations[None].repeat(batch_size, 1, 1, 1)
        # Repeat for pixel locations of the current image.
        cur_pixel_locations = torch.meshgrid([torch.arange(height), torch.arange(width)], indexing = 'ij')
        cur_pixel_locations = torch.stack(cur_pixel_locations[::-1], dim = 0).float()
        cur_pixel_locations = cur_pixel_locations[None].repeat(batch_size, 1, 1, 1)
        # Flow = cur_pixel_locations - ref_pixel_locations.
        return ref_pixel_locations, cur_pixel_locations

    # Upsample flow from [H/8, W/8, 2] to [H, W, 2].
    def UpsampleFlow(self, flow, mask):
        batch_size, _, height, width = flow.size()
        # mask: [batch_size, 8 * 8 * 9 = 576, height, width] -> [batch_size, 1, 9, 8, 8, height, width]
        mask = mask.view(batch_size, 1, 9, 8, 8, height, width)
        mask = torch.softmax(mask, dim = 2)
        # flow: [batch_size, 2, height, width] -> [batch_size, 2 * 3 * 3, height, width]
        up_flow = torch.nn.functional.unfold(8 * flow, [3, 3], padding = 1)
        up_flow = up_flow.view(batch_size, 2, 9, 1, 1, height, width)
        # flow: [batch_size, 2, 9, 1, 1, height, width] * [batch_size, 1, 9, 8, 8, height, width]
        #    -> [batch_size, 2, 9, 8, 8, height, width]
        #    -> (after do sum on dim 2) [batch_size, 2, 8, 8, height, width]
        up_flow = torch.sum(up_flow * mask, dim = 2)
        # flow: [batch_size, 2, 8, 8, height, width] -> [batch_size, 2, height, 8, width, 8]
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        # flow: [batch_size, 2, height, 8, width, 8] -> [batch_size, 2, height * 8, width * 8]
        up_flow = up_flow.reshape(batch_size, 2, height * 8, width * 8)
        return up_flow

    def forward(self, ref_image, cur_image):
        # Validate inputs.
        assert ref_image.size() == cur_image.size(), 'The size of the reference and current images should be the same.'
        # Normalize the images.
        ref_image = 2.0 * (ref_image / 255.0) - 1.0
        cur_image = 2.0 * (cur_image / 255.0) - 1.0
        ref_image = ref_image.contiguous()
        cur_image = cur_image.contiguous()
        # Extract image features.
        ref_feature = self.feature_encoder(ref_image).float()
        cur_feature = self.feature_encoder(cur_image).float()
        # Generate correlation pyramid.
        correlation_map = CorrelationPyramid(ref_feature, cur_feature, self.correlation_pyramid_levels, self.correlation_radius)
        # Extract image context.
        ref_inp, ref_net = self.context_encoder(ref_image)
        # Initialize flow.
        ref_pixel_locations, cur_pixel_locations = self.InitializeFlow(ref_feature.size())
        # Update the flow.
        all_predictions = []
        for iter in range(self.max_iterations):
            # Extract correlation volumes.
            correlation_volume = correlation_map(cur_pixel_locations)
            correlation = torch.cat(correlation_volume, dim = -1).permute(0, 3, 1, 2).contiguous().float()
            # Update the flow.
            flow = cur_pixel_locations - ref_pixel_locations
            ref_net, up_mask, delta_flow = self.update_block(ref_net, ref_inp, correlation, flow)
            cur_pixel_locations = cur_pixel_locations + delta_flow
            # Upsample flow. Record prediction at each iteration.
            up_flow = self.UpsampleFlow(cur_pixel_locations - ref_pixel_locations, up_mask)
            all_predictions.append(up_flow)

        return all_predictions

# Test the model.
if __name__ == '__main__':
    ref_image = torch.randn(5, 1, 60, 60)
    cur_image = torch.randn(5, 1, 60, 60)

    print('>> Test Raft:')
    raft = Raft(in_channels = 1,
                hidden_channels = 64,
                feature_channels = 128,
                context_channels = 128,
                correlation_pyramid_levels = 3,
                correlation_radius = 3,
                correlation_hidden_channels = 64,
                correlation_out_channels = 32,
                flow_hidden_channels = 32,
                flow_out_channels = 16,
                motion_out_channels = 32,
                mask_hidden_channels = 64,
                max_iterations = 5)
    all_predictions = raft(ref_image, cur_image)
    for i, prediction in enumerate(all_predictions):
        print(f'prediction_{i}.size():', prediction.size())
    print('Done.')
