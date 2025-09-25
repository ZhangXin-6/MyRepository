import torch.nn as nn
import torch


class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // 4, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_planes // 4, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))

        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()

        self.SamConv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print(avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(max_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        # print(x.shape)
        x = self.SamConv(x)
        # print(x.shape)
        return self.sigmoid(x)


class CAE(nn.Module):
    def __init__(self, in_planes):
        super(CAE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 4, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))

        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out
        return self.sigmoid(out)


if __name__ == '__main__':
    # tensor1 = torch.randn(2, 16, 128, 128, 32)  # B, C, H, W, D
    # CAE = ChannelAttentionEnhancement(16)
    # SAE = SpatialAttentionExtractor()
    # out1 = CAE(tensor1)
    # print(out1.shape)
    # out1 = tensor1 * out1
    # print(out1.shape)
    # out2 = SAE(out1)
    # print(out2.shape)
    # out3 = tensor1 * out2 + tensor1 * (1 - out2)
    # print(out3.shape)
    tensor1 = torch.randn(2, 16, 32, 32)
    CE = CAE(16)
    t2 = CE(tensor1)
    print(t2.shape)
