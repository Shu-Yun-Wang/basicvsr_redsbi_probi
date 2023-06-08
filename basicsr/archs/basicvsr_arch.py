import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .spynet_arch import SpyNet
from .quant_spynet_arch import Quantizable_SpyNet


@ARCH_REGISTRY.register()
class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, flow_type='zero',spynet_path='pretrained_models/flownet/spynet_inbasicvsr.pytorch'):
        super().__init__()
        self.num_feat = num_feat
        # alignment
        self.flow_type = flow_type 
        self.spynet_path = spynet_path              

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow_from_mv(self,mvs_0): 
        # motion vectors for blocks
        forwardtemp = []
        for i in range(len(mvs_0)):
            if mvs_0[i][0]==-1:
                forwardtemp.append(mvs_0[i])
        if forwardtemp==[]:
            mvs_0 = np.zeros_like(mvs_0)
        else:
            mvs_0 = np.array(forwardtemp)
        # flow = np.swapaxes([mvs_0[:, 3], mvs_0[:, 4],   # start point (X0, Y0)
        #                     mvs_0[:,5] - mvs_0[:,3],    # delta X
        #                     mvs_0[:, 6] - mvs_0[:, 4]], # delta Y
        #                     0, 1)
        flow_fixed = np.swapaxes([mvs_0[:, 5], mvs_0[:, 6],   # start point (X0, Y0)
                    mvs_0[:,3] - mvs_0[:,5],    # delta X
                    mvs_0[:,4] - mvs_0[:,6]], # delta Y
                    0, 1)
        flow = flow_fixed
        h=180
        w=320
        vis_mat = np.zeros((h, w, 2))

        num_vec = np.shape(flow)[0]
        for mv in np.split(flow, num_vec):
            # print("---")
            start_pt = (mv[0][0], mv[0][1])
            for i in range(16):
                for j in range(16):
                    try:
                        cur = (start_pt[1] + i - 8, start_pt[0] + j - 8) # pixel currently on
                        vis_mat[cur[0]][cur[1]] = (mv[0][2], mv[0][3])
                    except:
                        continue
        return vis_mat

    def get_flow(self, x):
        b, n, c, h, w = x.size()
        ## flow_type: of光流 zero无光流 quant_of量化光流 mv运动矢量

        ## 正常光流
        if self.flow_type == 'of':
            state_dict = torch.load(self.spynet_path)
            spynet = SpyNet().cuda()
            spynet.load_state_dict(state_dict)
            spynet.eval()
            x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
            x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
            flows_backward = spynet(x_1, x_2).view(b, n - 1, 2, h, w)
            flows_forward = spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        ## 光流为0
        elif self.flow_type == 'zero':
            flows_forward = torch.zeros([b,n-1,2,h,w]).cuda()
            flows_backward = torch.zeros([b,n-1,2,h,w]).cuda()

        ## 量化光流
        elif self.flow_type == 'quant_of':
            #"pretrained_models/flownet/spynet_quant.pytorch"
            state_dict = torch.load(self.spynet_path)
            model_int8 = Quantizable_SpyNet()
            model_int8.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            model_int8_prepared = torch.quantization.prepare(model_int8)
            model_int8 = torch.quantization.convert(model_int8_prepared)
            model_int8.load_state_dict(state_dict)
            model_int8.eval()
            x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w).cpu()
            x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w).cpu()
            flows_backward = model_int8(x_1, x_2).view(b, n - 1, 2, h, w).cuda()
            flows_forward = model_int8(x_2, x_1).view(b, n - 1, 2, h, w).cuda()

        elif self.flow_type == 'mv':
            flows_backward = []
            flows_forward = []
            for i in range(6):
                mv = np.load('datasets/motion_vectors/020/forward/mvs-'+str(i+1)+'.npy')
                flow = self.get_flow_from_mv(mv)
                flows_forward.append(flow.transpose((2,0,1)))
   
            flows_forward = torch.Tensor(flows_forward).view(b, n - 1, 2, h, w).cuda()
            # flows_forward = torch.flip(flows_forward,dims=[1])
            # flows_forward = torch.zeros_like(flows_forward)

            for i in range(6):
                mv = np.load('datasets/motion_vectors/020/backward/mvs-'+str(i+1)+'.npy')
                flow = self.get_flow_from_mv(mv)
                flows_backward.append(flow.transpose((2,0,1)))
            flows_backward = torch.Tensor(flows_backward).view(b, n - 1, 2, h, w).cuda()
            flows_backward = torch.flip(flows_backward,dims=[1])
            # flows_backward = torch.zeros_like(flows_backward)
        
        else:
            raise('please input the correct type of the flow')
        
        # 可视化
        # hidden_flow = flows_forward.squeeze().cpu()
        # flow_sum = np.zeros((6, 3, h, w))
        # for i in range(6):
        #     temp = hidden_flow[i:i+1,:,:,:].squeeze()
        #     temp = flow2rgb(np.transpose(temp.numpy(), [1, 2, 0])).transpose(2, 0, 1)
        #     flow_sum[i, :, :, :] = temp.reshape((1, 3, h, w))
        # flow_sum = torch.from_numpy(flow_sum)
        # writer = SummaryWriter('results/BasicVSR_REDS/log')
        # flow_sum=make_grid(flow_sum)
        # writer.add_image('flow',flow_sum)
        
        return flows_forward, flows_backward


    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


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