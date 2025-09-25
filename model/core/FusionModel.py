import torch
import torch.nn.functional as F
from torch import nn

from core.Attention import ChannelAttentionEnhancement, SpatialAttentionExtractor


class CostFusion(nn.Module):
    def __init__(self, dim):
        super(CostFusion, self).__init__()
        self.dim = dim
        self.CAE = ChannelAttentionEnhancement(dim)
        self.SAE = SpatialAttentionExtractor(kernel_size=7)

    # Low_scale----->High_scale
    # f0->f1->f2
    def forward(self, Cost_volume_list, scale):

        if len(Cost_volume_list) == 2:

            CL = Cost_volume_list[0]  # [B, C, H, W, D]
            CH = Cost_volume_list[1]  # [B, C, H, W, D]
            if scale > 1:
                CL = F.interpolate(CL, scale_factor=(scale, scale, scale), mode='trilinear', align_corners=False)
            else:
                CH = F.interpolate(CH, scale_factor=(scale, scale, scale), mode='trilinear', align_corners=False)

            assert CH.shape == CL.shape

            C_Sum = CH + CL
            return C_Sum
            # CH_CAE = self.CAE(C_Sum)
            # CH_CAE = CH_CAE * C_Sum
            # att = self.SAE(CH_CAE)
            # if scale == 2:
            #     CH = CH * att + CL * (1 - att)
            #     return CH.contiguous()  # [B, C, H, W, D]
            # elif scale == 0.5:
            #     CL = CL * att + CH * (1 - att)
            #     return CL.contiguous()  # [B, C, H, W, D]

        else:
            CL = Cost_volume_list[0]
            CM = Cost_volume_list[1]  # [B, C, H, W, D]
            CH = Cost_volume_list[2]
            CL = F.interpolate(CL, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
            CH = F.interpolate(CH, scale_factor=(0.5, 0.5, 0.5), mode='trilinear', align_corners=False)

            assert CH.shape == CL.shape == CM.shape
            C_Sum = CH + CL + CM
            # CM_CAE = self.CAE(C_Sum)
            # CM_CAE = CM_CAE * C_Sum
            # att = self.SAE(CM_CAE)
            # CM = CM * att + CH * (1 - att) + CL * (1 - att)
            # return CM.contiguous()  # [B, C, H, W, D]
            return C_Sum



if __name__ == '__main__':
    tensor1 = torch.randn(1, 32, 128, 128, 64)
    tensor1 = F.interpolate(tensor1, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
    print(tensor1.shape)
