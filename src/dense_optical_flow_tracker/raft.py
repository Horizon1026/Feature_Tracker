import torch
from encoder import *
from correlation_volumes import *
from update_block import *

class Raft(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, feature_channels, context_channels,
                 correlation_pyramid_levels, correlation_radius):
        super(Raft, self).__init__()
        self.hidden_dim = hidden_channels
        self.context_dim = context_channels
        self.correlation_pyramid_levels = correlation_pyramid_levels
        self.correlation_radius = correlation_radius

        self.feature_encoder = FeatureEncoder(in_channels, feature_channels)
        self.context_encoder = ContextEncoder(in_channels, context_channels, hidden_channels)

    def InitializeFlow(self, image):
        batch_size, _, height, width = image.size()
        height = height // 8
        width = width // 8
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

    def forward(self, ref_image, cur_image):
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
        for i, correlation in enumerate(correlation_map.correlation_pyramid):
            print(f'correlation_{i}.size():', correlation.size())
        # Extract image context.
        ref_net, ref_inp = self.context_encoder(ref_image)
        print('ref_net size:', ref_net.size())
        print('ref_inp size:', ref_inp.size())
        # Initialize flow.
        ref_pixel_locations, cur_pixel_locations = self.InitializeFlow(ref_image)
        # Update the flow.
        for iter in range(10):
            cur_pixel_locations = cur_pixel_locations.detach()
            # TODO: Implement the update block.

        return ref_image, cur_image

# Test the model.
if __name__ == '__main__':
    ref_image = torch.randn(5, 1, 224, 224)
    cur_image = torch.randn(5, 1, 224, 224)

    print('>> Test Raft:')
    raft = Raft(1, 64, 128, 64, 2, 3)
    ref_image, cur_image = raft(ref_image, cur_image)
