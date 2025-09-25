import torch
import torch.nn as nn
import torch.nn.functional as F


class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256,
                 out_dim=1,
                 ):
        super(DispHead, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))

        return out


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256,
                 kernel_size=3,
                 ):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class ConvGSC(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256,
                 kernel_size=3,
                 ):
        super(ConvGSC, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convt = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(hx))
        t = (1 - z) * h + q * z
        t = torch.tanh(self.convt(t))
        h = r * t
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_fn):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(cor_fn, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 1, 3, padding=1)

    def forward(self, disp, cost):
        cost = F.relu(self.convc1(cost))
        cost = F.relu(self.convc2(cost))
        dis = F.relu(self.convf1(disp))
        dis = F.relu(self.convf2(dis))

        cost_disp = torch.cat([cost, dis], dim=1)
        out = F.relu(self.conv(cost_disp))
        return torch.cat([out, disp], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, cor,
                 hidden_dim=128,
                 context_dim=128,
                 downsample_factor=4,
                 bilinear_up=False,
                 ):
        super(BasicUpdateBlock, self).__init__()

        self.encoder = BasicMotionEncoder(cor_fn=cor)

        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=context_dim + hidden_dim)

        self.disp_head = DispHead(hidden_dim, 256, 1)

        if bilinear_up:
            self.mask = None
        else:
            self.mask = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, (downsample_factor ** 2) * 9, 1, padding=0))

    def forward(self, net, inp, cost, disp):

        motion_features = self.encoder(disp, cost)

        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_disp = self.disp_head(net)

        if self.mask is not None:
            mask = .25 * self.mask(net)
        else:
            mask = None

        return net, mask, delta_disp


if __name__ == '__main__':
    tensor1 = torch.randn(1, 128, 32, 128)
    max_disp = 96
    disp_range = 96 // 4
    disp_range = disp_range * 2
    cost_volume = torch.randn(1, disp_range, 32, 128)
    disp = torch.randn(1, 1, 32, 128)
    gru = BasicUpdateBlock()
