# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py

import torch
import torch.nn as nn
import math


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        # 第i行全为i
        # [[1 1 1]
        #  [2 2 2]]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        # 第i列全为i
        # [[1 2 3]
        #  [1 2 3]]
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        # dim_t = [0., 1., ..., num_pos_feats-1.]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)

        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, num_pos_feats]
        pos_y = y_embed[:, :, :, None] / dim_t  # [B, H, W, num_pos_feats]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # print('pos_x dim = ', pos_x.ndim)
        # print('pos_x dim3 = ', pos_x.shape[-1])
        # print('pos_x dim2 = ', pos_x.shape[-2])
        # print('pos_x dim1 = ', pos_x.shape[-3])
        # print('pos_x dim1 = ', pos_x.shape[-4])
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # print('pos dim = ', pos.ndim)
        # print('pos dim3 = ', pos.shape[-1])
        # print('pos dim2 = ', pos.shape[-2])
        # print('pos dim1 = ', pos.shape[-3])
        # print('pos dim1 = ', pos.shape[-4])

        return pos
