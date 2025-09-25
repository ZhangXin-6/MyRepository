import torch
from torch import nn
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler


def _make_layer(dim1, dim2, kernel_size, stride, Trans=True):
    if Trans:
        layer1 = nn.ConvTranspose3d(dim1, dim2, kernel_size=kernel_size, stride=stride, padding=1)
    else:
        layer1 = nn.Conv3d(dim1, dim2, kernel_size=kernel_size, stride=stride, padding=1)
    layer2 = nn.BatchNorm3d(dim2)
    layer3 = nn.LeakyReLU()
    layers = (layer1, layer2, layer3)

    return nn.Sequential(*layers)


def _make_layer_1(dim1, dim2):
    layer1 = nn.Conv3d(dim1, dim2, kernel_size=1)
    layer2 = nn.BatchNorm3d(dim2)
    layer3 = nn.LeakyReLU()
    layers = (layer1, layer2, layer3)

    return nn.Sequential(*layers)


class Hourglass(nn.Module):
    def __init__(self, dim1, dim2):
        super(Hourglass, self).__init__()

        self.conv1 = _make_layer_1(dim1, dim2)
        self.conv2 = _make_layer_1(dim2, dim1)

        self.layer1 = _make_layer(dim2, dim2, kernel_size=3, stride=1, Trans=False)
        self.layer2 = _make_layer(dim2, dim2, kernel_size=3, stride=1, Trans=False)
        self.layer3 = _make_layer(dim2, 2 * dim2, kernel_size=3, stride=2, Trans=False)
        self.layer4 = _make_layer(2 * dim2, 2 * dim2, kernel_size=3, stride=1, Trans=False)
        self.layer5 = _make_layer(2 * dim2, 2 * dim2, kernel_size=3, stride=2, Trans=False)
        self.layer6 = _make_layer(2 * dim2, 2 * dim2, kernel_size=3, stride=1, Trans=False)
        self.layer7 = _make_layer(2 * dim2, 2 * dim2, kernel_size=4, stride=2, Trans=True)
        self.layer8 = _make_layer(2 * dim2, dim2, kernel_size=4, stride=2, Trans=True)

    def forward(self, x):
        x = self.conv1(x)  # [B, C, H, W, D]

        x1 = self.layer1(x)

        x1 = self.layer2(x1)

        x2 = self.layer3(x1)

        x2 = self.layer4(x2)

        x3 = self.layer5(x2)

        x3 = self.layer6(x3)

        x4 = self.layer7(x3)
        # print(x4.shape)
        # print(x2.shape)

        x4 += x2

        x5 = self.layer8(x4)

        x5 += x1

        x5 = self.conv2(x5)

        return x5.contiguous()  # [B, C, H, W, D]


class getDisp(nn.Module):
    def __init__(self, min_disp, max_disp, dim):
        super(getDisp, self).__init__()
        self.min_disp = min_disp // 4
        self.max_disp = max_disp // 4
        self.Conv = nn.Conv3d(dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.contiguous()  # [B, C, H, W, D]
        x = self.Conv(x)  # [N, 1, H, W, D]
        x = x.squeeze(1)  # [N, H, W, D]
        assert x.shape[-1] == self.max_disp - self.min_disp
        disp_candidates = torch.linspace(1.0 * self.min_disp, 1.0 * self.max_disp - 1.0,
                                         self.max_disp - self.min_disp).to(x.device)
        prob_volume = torch.softmax(-1.0 * x, -1)
        disparity = torch.sum(disp_candidates * prob_volume, dim=-1, keepdim=True)

        return x.permute(0, 3, 1, 2).contiguous(), disparity.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]


class Combined_Cost_Volume:
    def __init__(self, cost, num_levels=3, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.cost_volume_pyramid = []

        b, c, h, w, d = cost.shape

        cost_volume = cost.permute(0, 2, 3, 1, 4).reshape(b * h * w, c, 1, d)
        self.cost_volume_pyramid.append(cost_volume)
        for i in range(self.num_levels - 1):
            cost_volume = F.avg_pool2d(cost_volume, [1, 2], stride=[1, 2])
            self.cost_volume_pyramid.append(cost_volume)

    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.cost_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dx = dx.view(1, 1, 2 * r + 1, 1).to(disp.device)
            x0 = dx + disp.reshape(b * h * w, 1, 1, 1) / 2 ** i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0, y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
        out = torch.cat(out_pyramid, dim=-1)
        # print(out.shape)
        return out.permute(0, 3, 1, 2).contiguous().float()


if __name__ == '__main__':
    t = torch.randn(2, 3, 4, 4)
    print(t)
    classfication = torch.argmax(t, dim=1)
    print(classfication)

