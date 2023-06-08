import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


@ARCH_REGISTRY.register()
class CaVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.
    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
    """

    def __init__(self, num_feat=64, num_block=15):
        super().__init__()
        self.num_feat = num_feat

        # stage 1 and 2
        self.first_fe = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.second_fe = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_mv(self, x):
        return 
 
    def forward(self, x, label):
        """Forward function of codec-aware propagation in each GoP.
        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
            label: A list contains the frame type (I, B, P) of each frame.
        """
        # initialize placeholder
        init_p = True
        out_l = []
        b, n, _, h, w = x.size()
        hidden_for = x.new_zeros(b, self.num_feat, h, w)
        hidden_back = x.new_zeros(b, self.num_feat, h, w)

        mv_forward, mv_backward = self.get_mv(x)
        feat_first_all = self.first_fe(x)

        for i in range(0, n):
            feat_stage1 = feat_first_all[i]

            # warp and aggregation
            if label[i] == 'I':
                feat_for_warp = x.new_zeros(b, self.num_feat, h, w)
                feat_back_warp = x.new_zeros(b, self.num_feat, h, w)
                feat_stage2 = torch.cat([feat_stage1, feat_for_warp, feat_back_warp], dim=1)
                feat_stage2 = self.second_fe(feat_stage2)
                hidden_for = feat_stage2
            elif label[i] == 'P':
                    feat_for_warp = self.mv_warp(hidden_for, mv_forward[i])
                    feat_back_warp = x.new_zeros(b, self.num_feat, h, w)
                    feat_stage2 = torch.cat([feat_stage1, feat_for_warp, feat_back_warp], dim=1)
                    feat_stage2 = self.second_fe(feat_stage2)
                if init_p == True:
                    hidden_back = feat_stage2
                    init_p = False
                else:
                    hidden_for = hidden_back
                    hidden_back = feat_stage2
            else:
                feat_for_warp = self.mv_warp(hidden_for, mv_forward[i])
                feat_back_warp = self.mv_warp(hidden_back, mv_backward[i])
                feat_stage2 = torch.cat([feat_stage1, feat_for_warp, feat_back_warp], dim=1)
                feat_stage2 = self.second_fe(feat_stage2)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_stage2)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x[i], scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)