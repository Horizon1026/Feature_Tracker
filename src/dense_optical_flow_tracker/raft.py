# 代码由 Deepseek R1 生成，仅供参考
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEncoder(nn.Module):
    """特征编码器模块，用于提取图像特征"""
    def __init__(self, output_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 下采样
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)

class ContextEncoder(nn.Module):
    """上下文编码器，提取上下文特征和隐藏状态"""
    def __init__(self, feature_dim=256, hidden_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # 输出隐藏状态和上下文特征
        self.hidden_state = nn.Conv2d(256, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.conv_layers(x)
        hidden = self.hidden_state(x)
        return features, hidden

class CorrelationPyramid(nn.Module):
    """修正后的相关金字塔模块"""
    def __init__(self, num_levels=4):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, fmap1, fmap2):
        # 重新设计维度处理逻辑
        batch, dim, ht, wd = fmap1.shape

        # 计算初始相关体积
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)  # [batch, ht*wd, ht*wd]
        corr = corr.view(batch, ht, wd, ht, wd)  # 调整为4D相关体积

        # 转换为适合3D卷积的格式 [batch, 1, d, h, w]
        corr = corr.reshape(batch, ht, wd, ht, wd)
        corr = corr.unsqueeze(1)  # 添加通道维度 [batch, 1, ht, wd, ht, wd]

        # 调整维度顺序为 [batch, channels, depth, height, width]
        corr = corr.permute(0, 1, 4, 2, 5, 3)  # 调整为可池化形式
        corr = corr.reshape(batch, 1, ht*ht, wd, wd)  # 合并空间维度

        # 构建金字塔
        pyramid = []
        for _ in range(self.num_levels):
            # 使用3D平均池化进行下采样（调整核尺寸为立方体）
            corr = F.avg_pool3d(corr, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            pyramid.append(corr)

        return pyramid

class MotionEncoder(nn.Module):
    """运动编码器，编码光流和相关特征"""
    def __init__(self, corr_dim=1, flow_dim=2, output_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(corr_dim + flow_dim, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1)
        )

    def forward(self, flow, corr):
        corr = F.relu(corr)
        motion_input = torch.cat([flow, corr], dim=1)
        return self.conv_layers(motion_input)

class UpdateBlock(nn.Module):
    """迭代更新模块（GRU结构）"""
    def __init__(self, hidden_dim=128, corr_dim=256):
        super().__init__()
        # GRU的更新门计算
        self.convz = nn.Conv2d(hidden_dim + corr_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + corr_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + corr_dim, hidden_dim, 3, padding=1)

    def forward(self, hidden, context, corr_features):
        # GRU更新步骤
        hidden_context = torch.cat([hidden, context, corr_features], dim=1)

        z = torch.sigmoid(self.convz(hidden_context))
        r = torch.sigmoid(self.convr(hidden_context))
        q = torch.tanh(self.convq(torch.cat([r * hidden, context, corr_features], dim=1)))

        hidden = (1 - z) * hidden + z * q
        return hidden

class RAFT(nn.Module):
    """完整的RAFT模型"""
    def __init__(self, num_iters=12, hidden_dim=128):
        super().__init__()
        self.num_iters = num_iters

        # 特征提取模块
        self.feature_encoder = FeatureEncoder(256)
        self.context_encoder = ContextEncoder(256, hidden_dim)

        # 相关金字塔模块
        self.corr_pyramid = CorrelationPyramid(4)

        # 更新模块组件
        self.motion_encoder = MotionEncoder()
        self.update_block = UpdateBlock(hidden_dim)

        # 光流预测头
        self.flow_head = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)

    def forward(self, image1, image2):
        # 特征提取
        fmap1 = self.feature_encoder(image1)
        fmap2 = self.feature_encoder(image2)

        # 构建相关金字塔
        pyramid = self.corr_pyramid(fmap1, fmap2)

        # 初始化光流和隐藏状态
        batch, _, h, w = fmap1.shape
        flow = torch.zeros(batch, 2, h, w, device=image1.device)
        hidden = torch.zeros(batch, 128, h, w, device=image1.device)

        # 迭代更新
        flow_predictions = []
        for _ in range(self.num_iters):
            # 从金字塔采样相关特征
            corr = self.sample_correlation(pyramid, flow)

            # 运动特征编码
            motion_features = self.motion_encoder(flow, corr)

            # GRU更新隐藏状态
            hidden = self.update_block(hidden, motion_features)

            # 预测光流增量
            delta_flow = self.flow_head(hidden)
            flow = flow + delta_flow
            flow_predictions.append(flow)

        return flow_predictions

    def sample_correlation(self, pyramid, flow):
        """简化的特征采样方法（实际应实现双线性采样）"""
        # 取第一个金字塔层并调整维度
        return pyramid[0][:, :, ::8, ::8, ::8]  # 示例采样方式


# 使用示例
if __name__ == "__main__":
    # 输入形状: [batch, channel, height, width]
    image1 = torch.randn(2, 3, 256, 256)
    image2 = torch.randn(2, 3, 256, 256)

    model = RAFT()
    predictions = model(image1, image2)
    final_flow = predictions[-1]
    print("Output flow shape:", final_flow.shape)  # 应该输出 [2, 2, 64, 64]
