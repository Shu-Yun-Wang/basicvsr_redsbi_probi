import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.quantization import fuse_modules
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import flow_warp

class Quantizable_BasicModule(nn.Module):

    def __init__(self):
        super(Quantizable_BasicModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.relu4 = nn.ReLU(inplace=False)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)

    def forward(self, tensor_input):
        conv1=self.conv1(tensor_input)
        relu1=self.relu1(conv1)
        conv2=self.conv2(relu1)
        relu2=self.relu2(conv2)
        conv3=self.conv3(relu2)
        relu3=self.relu3(conv3)
        conv4=self.conv4(relu3)
        relu4=self.relu4(conv4)
        conv5=self.conv5(relu4)
        return conv5

    def fuse_model(self):
        fuse_modules(self, ['conv1', 'relu1'], inplace=False)
        fuse_modules(self, ['conv2', 'relu2'], inplace=False)
        fuse_modules(self, ['conv3', 'relu3'], inplace=False)
        fuse_modules(self, ['conv4', 'relu4'], inplace=False)


@ARCH_REGISTRY.register()
class Quantizable_SpyNet(nn.Module):
    def __init__(self, load_path=None):
        super(Quantizable_SpyNet, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.add = torch.nn.quantized.FloatFunctional()

        self.basic_module = nn.ModuleList([Quantizable_BasicModule() for _ in range(6)])
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        # self.mean = torch.quantize_per_tensor(self.mean, scale=1.0, zero_point=0, dtype=torch.quint8)
        # tensor_output = self.add.mul(self.add.add(tensor_input,-self.mean),1/self.std)
        # tensor_output = self.add.add(tensor_input,self.mean)
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def forward(self, ref, supp):

        ref = self.quant(ref)
        supp = self.quant(supp)

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow = torch.zeros([ref.size(0), 2, int(math.floor(ref.size(2) / 2.0)), int(math.floor(ref.size(3) / 2.0))])
        flow = self.quant(flow)
        # ref = [self.preprocess(ref)]
        # supp = [self.preprocess(supp)]
        ref = [ref]
        supp = [supp]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        for level in range(len(ref)):
            upsampled_flow = self.add.mul_scalar(F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True),2.0)

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.add.cat([ref[level],flow_warp(supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),upsampled_flow], 1)
            
            flow = self.basic_module[level](flow)
            flow = self.dequant(flow)
            flow = flow+upsampled_flow

        flow = F.interpolate(input=flow, size=(h, w), mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow
        
    def fuse_model(self):
        for m in self.modules():
            if type(m) == Quantizable_BasicModule:
                m.fuse_model()

