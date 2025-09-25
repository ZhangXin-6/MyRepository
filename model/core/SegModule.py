import torch
import torch.nn as nn
import torch.nn.functional as F


class SegModule(nn.Module):
    def __init__(self, dim, category_nums):
        super(SegModule, self).__init__()

        self.avg_layer1 = nn.AvgPool2d(5, stride=2, padding=2)
        self.avg_layer1 = nn.AvgPool2d(5, stride=2, padding=2)

        self.conv_layer0 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(inplace=True))

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(3 * dim, 3 * dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(3 * dim),
            nn.ReLU(inplace=True))

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(5 * dim, 3 * dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(3 * dim),
            nn.ReLU(inplace=True))

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(4 * dim, 2 * dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(inplace=True))

        self.deconv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(3 * dim, 3 * dim, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(3 * dim),
            nn.ReLU(inplace=True)
        )
        self.deconv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(3 * dim, 3 * dim, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(3 * dim),
            nn.ReLU(inplace=True)
        )

        self.output_layer = nn.Conv2d(2 * dim, category_nums, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_0 = self.conv_layer0(x[0])  # C H W
        x_1 = self.avg_layer1(x[0])  # C H//2 W//2
        x_1 = torch.cat((x_1, x[1]), dim=1)  # 2C H//2 W//2
        x_1 = self.conv_layer1(x_1)  # 2C H//2 W//2

        x_2 = self.avg_layer1(x_1)  # 2C H//4 W//4
        x_2 = torch.cat((x_2, x[2]), dim=1)     # 3C H//4 W//4
        x_2 = self.conv_layer2(x_2)     # 3C H//4 W//4

        x_3 = self.deconv_layer1(x_2)   # 3C H//2 W//2
        x_3 = torch.cat((x_3, x_1), dim=1)  # 5C H//2 W//2
        x_3 = self.conv_layer3(x_3)     # 3C H//2 W//2

        x_4 = self.deconv_layer2(x_3)   # 3C H//2 W//2
        x_4 = torch.cat((x_4, x_0), dim=1)  # 4C H//2 W//2
        x_4 = self.conv_layer4(x_4)


        output = self.output_layer(x_4)

        return output


if __name__ == '__main__':
    net = SegModule(3, 10)
    t = torch.ones(2, 1, 32, 32)
    print(t)
    t0 = torch.ones(2, 1, 32, 32)
    print(t0)
    t1 = torch.randn(1, 3, 16, 16)
    t2 = torch.randn(1, 3, 8, 8)
    # list_t = (t, t1, t2)
    # x1 = net(list_t)
    # print(x1.size())
    # print(x1)
    seg_loss = F.cross_entropy(t, t0, reduction='none').mean()
    print(seg_loss)
