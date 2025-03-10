import torch
import torch.nn as nn

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

    def __call__(self):
        # TODO: https://blog.csdn.net/qq_39546227/article/details/115005833
        print('>> CorrelationPyramid.__call__():')
        return 0


# Test the model.
if __name__ == '__main__':
    fmap0 = torch.randn(5, 256, 32, 32)
    fmap1 = torch.randn(5, 256, 32, 32)
    model = CorrelationPyramid(fmap0, fmap1, num_levels = 4, radius = 4)
    # Print the correlation pyramid.
    for i, correlation in enumerate(model.correlation_pyramid):
        print(f'correlation_{i}.size():', correlation.size())
    model()
