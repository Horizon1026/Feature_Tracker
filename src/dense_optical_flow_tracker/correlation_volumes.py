import torch

def BilinearSampler(fmap, pixel_locations):
    height, width = fmap.shape[-2:]
    # Divide x and y coordinates. Which means [..., 2] -> [..., 1], [..., 1].
    x_grid, y_grid = pixel_locations.split([1, 1], dim = -1)
    # Normalize the coordinates to [-1, 1].
    x_grid = 2 * x_grid / (width - 1) - 1
    y_grid = 2 * y_grid / (height - 1) - 1
    # Concatenate x and y coordinates. Which means [..., 1], [..., 1] -> [..., 2].
    grid = torch.cat([x_grid, y_grid], dim = -1)
    # Sample the image with the coordinates.
    # fmap: [batch_size * height * width, channels, height, width]
    # grid: [batch_size * height * width, window_height, window_width, 2]
    # sampled_fmap: [batch_size * height * width, channels, window_height, window_width]
    sampled_fmap = torch.nn.functional.grid_sample(fmap, grid, align_corners = True)
    return sampled_fmap

class CorrelationPyramid():
    def __init__(self, fmap0, fmap1, num_levels, radius):
        assert fmap0.size() == fmap1.size()
        self.num_levels = num_levels
        self.radius = radius
        self.correlation_pyramid = []
        # Compute basic correlation.
        correlation_0 = self.ComputeCorrelation(fmap0, fmap1)
        b, h, w, dim, h, w = correlation_0.size()
        correlation_0 = correlation_0.view(b * h * w, dim, h, w)
        self.correlation_pyramid.append(correlation_0)
        # Compute correlation pyramid.
        correlation = correlation_0
        for _ in range(self.num_levels - 1):
            correlation = torch.nn.functional.avg_pool2d(correlation, kernel_size = 2, stride = 2)
            self.correlation_pyramid.append(correlation)

    def ComputeCorrelation(self, fmap0, fmap1):
        batch_size, channels, height, width = fmap0.size()
        # Transform fmaps to be cols of vectors. Size of vector is height * width. Number of vectors is channels.
        fmap0 = fmap0.view(batch_size, channels, height * width)
        fmap1 = fmap1.view(batch_size, channels, height * width)
        # Compute correlation.
        # [batch_size, channels, height * width] * [batch_size, height * width, channels] -> [batch_size, height * width, height * width]
        correlation = torch.matmul(fmap0.transpose(1, 2), fmap1)
        correlation = correlation.view(batch_size, height, width, 1, height, width)
        return correlation / torch.sqrt(torch.tensor(channels).float())

    def __call__(self, pixel_locations):
        # pixel_locations: [batch_size, 2, height, width] -> [batch_size, height, width, 2]
        # 'pixel_locations' comes from torch.meshgrid() function.
        assert pixel_locations.size(1) == 2, 'The size of pixel_locations should be [batch_size, 2, height, width].'
        pixel_locations = pixel_locations.permute(0, 2, 3, 1)
        batch_size, height, width, _ = pixel_locations.size()
        r = self.radius
        out_pyramid = []
        for i, correlation in enumerate(self.correlation_pyramid):
            # Create search window with size of [2 * r + 1, 2 * r + 1, 2].
            # window_height = window_width = 2 * r + 1.
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing = 'ij'), dim = -1).to(correlation.device)
            # Compute location of all pixels in search window with respect to the center pixel.
            # Scale the pixel locations to the current level of the pyramid.
            scale = 2 ** i
            centriod = pixel_locations / scale
            # centriod: [batch_size, height, width, 2] -> [batch_size * height * width, 1, 1, 2]
            centriod = centriod.reshape(batch_size * height * width, 1, 1, 2)
            # delta: [2 * r + 1, 2 * r + 1, 2] -> [1, 2 * r + 1, 2 * r + 1, 2]
            delta = delta.reshape(1, 2 * r + 1, 2 * r + 1, 2)
            # For each center pixel (with number of batch_size * height * width),
            # search all pixels (with number of (2r+1)^2) in the search window centered at the pixel.
            # [batch_size * height * width, 1, 1, 2] + [1, 2 * r + 1, 2 * r + 1, 2] ->
            # [batch_size * height * width, 2 * r + 1, 2 * r + 1, 2]
            scaled_pixel_locations = centriod + delta
            # Search neibor features for each pixel in correlation volume.
            sampled_correlation = BilinearSampler(correlation, scaled_pixel_locations)
            # sampled_correlation: [batch_size * height * width, channels, window_height, window_width]
            # -> [batch_size, height, width, (2 * r + 1) ** 2]
            sampled_correlation = sampled_correlation.view(batch_size, height, width, (2 * r + 1) ** 2)
            out_pyramid.append(sampled_correlation)
        return out_pyramid


# Test the model.
if __name__ == '__main__':
    fmap0 = torch.randn(5, 128, 8, 8)
    fmap1 = torch.randn(5, 128, 8, 8)
    model = CorrelationPyramid(fmap0, fmap1, num_levels = 3, radius = 3)
    # Print the correlation pyramid.
    for i, correlation in enumerate(model.correlation_pyramid):
        print(f'correlation_{i}.size():', correlation.size())

    pixel_locations = torch.randn(5, 2, 8, 8)
    out_pyramid = model(pixel_locations)
    # Print the output pyramid.
    for i, out in enumerate(out_pyramid):
        print(f'out_{i}.size():', out.size())
    sample_output = torch.cat(out_pyramid, dim = -1).permute(0, 3, 1, 2).contiguous().float()
    print('sample_output.size():', sample_output.size())
    print('Done.')
