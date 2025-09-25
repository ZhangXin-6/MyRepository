import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import torch.nn as nn

from core.position import PositionEmbeddingSine


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4 or x.ndim == 3) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    # print(f"img dim = ", img.ndim)
    # print('img dim4 = ', img.shape[-1])
    # print('img dim3 = ', img.shape[-2])
    # print('img dim2 = ', img.shape[-3])
    # print('img dim1 = ', img.shape[-4])
    H, W = img.shape[-2:]
    # print('H, W = ', H, W)
    xgrid, ygrid = coords.split([1, 1], dim=-1)

    # print(f"xgrid dim = ", xgrid.ndim)
    # print('xgrid dim4 = ', xgrid.shape[-1])
    # print('xgrid dim3 = ', xgrid.shape[-2])
    # print('xgrid dim2 = ', xgrid.shape[-3])
    # print('xgrid dim1 = ', xgrid.shape[-4])

    xgrid = 2 * xgrid / (W - 1) - 1
    if H > 1:
        ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    # print(f"grid dim = ", grid.ndim)
    # print('grid dim4 = ', grid.shape[-1])
    # print('grid dim3 = ', grid.shape[-2])
    # print('grid dim2 = ', grid.shape[-3])
    # print('grid dim1 = ', grid.shape[-4])
    img = F.grid_sample(img, grid, align_corners=True)
    # print("img dim = ", img.ndim)
    # print('img dim4 = ', img.shape[-1])
    # print('img dim3 = ', img.shape[-2])
    # print('img dim2 = ', img.shape[-3])
    # print('img dim1 = ', img.shape[-4])
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def updisp16(disp, mode='bilinear'):
    new_size = (16 * disp.shape[2], 16 * disp.shape[3])
    return 16 * F.interpolate(disp, size=new_size, mode=mode, align_corners=True)


def updisp8(disp, mode='bilinear'):
    new_size = (8 * disp.shape[2], 8 * disp.shape[3])
    return 8 * F.interpolate(disp, size=new_size, mode=mode, align_corners=True)


def updisp4(disp, mode='bilinear'):
    new_size = (4 * disp.shape[2], 4 * disp.shape[3])
    return 4 * F.interpolate(disp, size=new_size, mode=mode, align_corners=True)


def upseg4(seg, mode='bilinear'):
    return F.interpolate(seg, scale_factor=4, mode=mode)


def feature_corr(fmap1, fmap2):
    B, D, H, W1 = fmap1.shape
    _, _, _, W2 = fmap2.shape
    fmap1 = fmap1.view(B, D, H, W1)
    fmap2 = fmap2.view(B, D, H, W2)
    corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
    corr = corr.reshape(B, H, W1, 1, W2).contiguous()
    return corr / torch.sqrt(torch.tensor(D).float())


def matching(corr_volume):
    b, h, w1, _, w2 = corr_volume.shape

    init_grid = coords_grid(b, h, w1).to(corr_volume.device).reshape(b, 1, h, w1)  # [B, 2, H, W]
    print(init_grid)
    grid = init_grid[:, :, 0, :]  # [B, 2, W]
    grid = grid.permute(0, 2, 1)  # [B, W, 2]
    print(grid)

    correlation = corr_volume.view(b, h * w1, w2)  # [B, H*W, W]

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, W]

    correspondence = torch.matmul(prob, grid).view(b, h, w1, 1).permute(0, 3, 1, 2)  # [B, 2, H, W]

    disp = correspondence - init_grid
    disp[:, 1, :, :] = 0
    disp = disp[:, :1]

    return disp


def StereoMatching(corr_volume):
    b, h, w1, _, w2 = corr_volume.shape
    x_grid = torch.linspace(0, w1 - 1, w1, device=corr_volume.device)
    correlation = corr_volume.view(b, h, w1, w2)
    prob = F.softmax(correlation, dim=-1)
    correspondence = (x_grid.view(1, 1, 1, w1) * prob).sum(-1)  # [B, H, W]
    disparity = correspondence - x_grid.view(1, 1, w1).repeat(b, h, 1)

    return disparity.reshape(b, 1, h, w1)


def feature_add_position(feature0, feature1, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    position = pos_enc(feature0)

    feature0 = feature0 + position
    feature1 = feature1 + position

    return feature0, feature1
