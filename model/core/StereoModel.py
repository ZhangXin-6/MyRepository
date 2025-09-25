import torch
import torch.nn as nn
import torch.nn.functional as F

from core.CostAggregation import Hourglass, getDisp
from core.CostVolume import CostDifference
from core.FeatureTransformer import FeatureTransformer
from core.FusionModel import CostFusion
from core.SegModule import SegModule
from core.extractor import BasicEncoder, ContextEncoder
from core.update import BasicUpdateBlock
from core.utils.utils import feature_add_position, updisp4, upseg4

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StereoModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.min_disp = -96
        self.max_disp = 96

        self.upsample_factor = 2

        self.dim = 48

        self.Cost_dim = 48

        self.range = (self.max_disp - self.min_disp) // 4

        # 特征增强
        self.FeatureTransformer = FeatureTransformer(num_layers=args.tf_layers, mode=args.mode, head_num=args.head_num,
                                                     d_model=self.dim)

        self.featureNet = BasicEncoder(output_dim=self.dim, norm_fn='instance')
        self.contextNet = ContextEncoder(output_dim=128, norm_fn='batch')
        self.update_block = BasicUpdateBlock(hidden_dim=128, cor=self.range)

        self.Hourglass_1 = Hourglass(dim1=self.Cost_dim, dim2=self.Cost_dim // 2)
        self.Hourglass_2 = Hourglass(dim1=self.Cost_dim, dim2=self.Cost_dim // 2)
        self.Hourglass_3 = Hourglass(dim1=self.Cost_dim, dim2=self.Cost_dim // 2)
        self.Hourglass_4 = Hourglass(dim1=self.Cost_dim, dim2=self.Cost_dim // 2)
        self.Hourglass_5 = Hourglass(dim1=self.Cost_dim, dim2=self.Cost_dim // 2)
        self.Hourglass_6 = Hourglass(dim1=self.Cost_dim, dim2=self.Cost_dim // 2)

        # low -> high
        self.costFusion_16 = CostFusion(dim=self.Cost_dim)
        self.costFusion_8 = CostFusion(dim=self.Cost_dim)
        self.costFusion_4 = CostFusion(dim=self.Cost_dim)

        self.getDisp = getDisp(dim=self.Cost_dim, min_disp=self.min_disp, max_disp=self.max_disp)

        # self.seg = SegModule(dim=self.dim, category_nums=6)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask):  # with gru update
        """ Upsample flow field [H/4, W/4, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = disp.shape
        factor = 2 ** self.upsample_factor
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(factor * disp, [3, 3], padding=1)
        up_disp = up_disp.view(N, D, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=12, mode="train"):
        """ Estimate disparity between pairs of frames """
        # 归一化到[-1, 1]

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # image1 = image1.contiguous()
        # image2 = image2.contiguous()

        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.featureNet(image1)
            fmap2 = self.featureNet(image2)
            net, inp = self.contextNet(image1)
        assert len(fmap1) == len(fmap2) == 3

        for i in range(len(fmap1)):
            fmap1[i], fmap2[i] = feature_add_position(fmap1[i], fmap2[i], feature_channels=self.dim)
            fmap1[i], fmap2[i] = self.FeatureTransformer(fmap1[i], fmap2[i])

        # diff_volume       [B, C, H, W, D]


        cost_volume_1 = CostDifference(fmap1[2], fmap2[2], self.min_disp // 16, self.max_disp // 16)  # 1/16
        cost_volume_2 = CostDifference(fmap1[1], fmap2[1], self.min_disp // 8, self.max_disp // 8)  # 1/8
        cost_volume_3 = CostDifference(fmap1[0], fmap2[0], self.min_disp // 4, self.max_disp // 4)  # 1/4

        cost_volume_1 = self.Hourglass_1(cost_volume_1)  # 1/16
        cost_volume_2 = self.Hourglass_2(cost_volume_2)  # 1/8
        cost_volume_3 = self.Hourglass_3(cost_volume_3)  # 1/4

        cost_volume_1 = self.costFusion_16([cost_volume_1, cost_volume_2], 0.5)  # 1/16
        cost_volume_1 = self.Hourglass_4(cost_volume_1)  # 1/16
        cost_volume_2 = self.costFusion_8([cost_volume_1, cost_volume_2, cost_volume_3], 1)  # 1/8
        cost_volume_2 = self.Hourglass_5(cost_volume_2)  # 1/8
        cost_volume_3 = self.costFusion_4([cost_volume_2, cost_volume_3], 2)  # 1/4
        cost_volume_3 = self.Hourglass_6(cost_volume_3)  # 1/4

        cost, disp_4 = self.getDisp(cost_volume_3)  # 1 / 4 disp map



        if mode == "train":
            # get disp map 1/16 1/8 1/4
            dispList = []

            disp_init = updisp4(disp_4)

            for itr in range(iters):
                disp = disp_4.detach()

                with autocast(enabled=self.args.mixed_precision):
                    net, up_mask, delta_disp = self.update_block(net, inp, cost, disp)

                # F(t+1) = F(t) + \Delta(t)
                disp_4 = disp_4 + delta_disp
                # We do not need to upsample or output intermediate results in test_mode
                # upsample predictions
                disp_up = self.upsample_disp(disp_4, up_mask)
                dispList.append(disp_up)

            return dispList, disp_init
        else:
            for itr in range(iters):
                disp = disp_4.detach()
                # cost = cost_fn(disp_4, coords)

                with autocast(enabled=self.args.mixed_precision):
                    net, up_mask, delta_disp = self.update_block(net, inp, cost, disp)
                # F(t+1) = F(t) + \Delta(t)
                disp_4 = disp_4 + delta_disp
                # We do not need to upsample or output intermediate results in test_mode
                # upsample predictions
            disp_up = self.upsample_disp(disp_4, up_mask)

            return disp_up
