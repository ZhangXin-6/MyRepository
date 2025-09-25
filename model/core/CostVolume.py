import torch
import torch.nn.functional as F


def CostDifference(FL, FR, min_disp, max_disp):
    assert FL.shape == FR.shape  # [N, C, H, W]
    FL = FL.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
    FR = FR.permute(0, 2, 3, 1).contiguous()

    Cost_volume = []
    for i in range(min_disp, max_disp):
        if i < 0:
            Cost_volume.append(F.pad(FL[:, :, :i, :] - FR[:, :, -i:, :],
                                     pad=(0, 0, 0, -i), mode='constant', value=0))
        elif i > 0:
            Cost_volume.append(F.pad(FL[:, :, i:, :] - FR[:, :, :-i, :],
                                     pad=(0, 0, i, 0), mode='constant', value=0))
        else:
            Cost_volume.append(FL - FR)
    Cost_volume = torch.stack(Cost_volume, 1)

    return Cost_volume.permute(0, 4, 2, 3, 1).contiguous()  # [N, C, H, W, D]


def CostConcat(FL, FR, min_disp, max_disp):
    assert FL.shape == FR.shape
    FL = FL.permute(0, 2, 3, 1).contiguous()
    FR = FR.permute(0, 2, 3, 1).contiguous()

    Cost_volume = []
    for i in range(min_disp, max_disp):
        if i < 0:
            Cost_volume.append(F.pad(torch.cat((FL[:, :, :i, :], FR[:, :, -i:, :]), dim=-1),
                                     pad=(0, 0, 0, -i), mode='constant', value=0))
        elif i > 0:
            Cost_volume.append(F.pad(torch.cat((FL[:, :, i:, :], FR[:, :, :-i, :]), dim=-1),
                                     pad=(0, 0, i, 0), mode='constant', value=0))
        else:
            Cost_volume.append(torch.cat((FL, FR), dim=-1))
    Cost_volume = torch.stack(Cost_volume, 1)

    return Cost_volume.permute(0, 4, 2, 3, 1).contiguous()  # [N, 2C, H, W, D]


def build_concat_volume(Left_feat, right_feat, mindisp, maxdisp):
    B, C, H, W = Left_feat.shape
    volume = Left_feat.new_zeros([B, 2 * C, maxdisp - mindisp, H, W])
    for i in range(mindisp, maxdisp):
        if i > 0:
            volume[:, :C, i - mindisp, :, i:] = Left_feat[:, :, :, i:]
            volume[:, C:, i - mindisp, :, i:] = right_feat[:, :, :, :-i]
        elif i == 0:
            volume[:, :C, i - mindisp, :, :] = Left_feat
            volume[:, C:, i - mindisp, :, :] = right_feat
        elif i < 0:
            volume[:, :C, i - mindisp, :, :i] = Left_feat[:, :, :, :i]
            volume[:, C:, i - mindisp, :, :i] = right_feat[:, :, :, -i:]
    volume = volume.permute(0, 1, 3, 4, 2).contiguous()  # [B, C, H, W, D]
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.reshape(B, num_groups, channels_per_group, H, W)
    fea2 = fea2.reshape(B, num_groups, channels_per_group, H, W)
    cost = torch.einsum("bndhw, bnehw->bnhw", fea1, fea2) / (channels_per_group ** 0.5)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(Left_feat, right_feat, mindisp, maxdisp, num_groups):
    B, C, H, W = Left_feat.shape
    volume = Left_feat.new_zeros([B, num_groups, maxdisp - mindisp, H, W])
    for i in range(-maxdisp, maxdisp):
        if i > 0:
            volume[:, :, i - mindisp, :, i:] = groupwise_correlation(Left_feat[:, :, :, i:], right_feat[:, :, :, :-i], num_groups)
        if i == 0:
            volume[:, :, i - mindisp, :, :] = groupwise_correlation(Left_feat, right_feat, num_groups)
        if i < 0:
            volume[:, :, i - mindisp, :, :i] = groupwise_correlation(Left_feat[:, :, :, :i], right_feat[:, :, :, -i:], num_groups)

    volume = volume.permute(0, 1, 3, 4, 2).contiguous()  # [B, C, H, W, D]
    return volume


def build_corr_volume(Left_feat, right_feat, mindisp, maxdisp, num_groups):
    B, C, H, W = Left_feat.shape
    volume = Left_feat.new_zeros([B, num_groups, maxdisp - mindisp, H, W])
    for i in range(-maxdisp, maxdisp):
        if i > 0:
            volume[:, :, i - mindisp, :, i:] = correlation(Left_feat[:, :, :, i:], right_feat[:, :, :, :-i], num_groups)
        if i == 0:
            volume[:, :, i - mindisp, :, :] = correlation(Left_feat, right_feat, num_groups)
        if i < 0:
            volume[:, :, i - mindisp, :, :i] = correlation(Left_feat[:, :, :, :i], right_feat[:, :, :, -i:], num_groups)

    volume = volume.permute(0, 1, 3, 4, 2).contiguous()  # [B, C, H, W, D]
    return volume


if __name__ == '__main__':
    tensor0 = torch.randn(1, 3, 1, 3)

    tensor1 = torch.randn(1, 3, 1, 3)

    volume1 = CostConcat(tensor0, tensor1, -2, 2)
    print(volume1)
    print(volume1.shape)
    volume2 = CostDifference(tensor0, tensor1, -2, 2)
    print(volume2)
    print(volume2.shape)
    volume3 = build_concat_volume(tensor0, tensor1, -2, 2)
    print(volume3)
    print(volume3.shape)
    volume4 = build_gwc_volume(tensor0, tensor1, -2, 2, 1)
    print(volume4)
    print(volume4.shape)
    volume5 = build_corr_volume(tensor0, tensor1, -2, 2, 1)
    print(volume5)
    print(volume5.shape)
