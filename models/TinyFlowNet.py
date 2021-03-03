import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util import conv, predict_flow, deconv, crop_like, depthwise_separable_conv, dds_conv


__all__ = [
    'tinyflownet', 'tinyflownet_bn'
]


class TinyFlowNet(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True):
        super(TinyFlowNet,self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.batchNorm = batchNorm
        self.conv1   = depthwise_separable_conv(self.batchNorm, 6,  24, kernel_size=3, stride=2)
        self.conv2   = depthwise_separable_conv(self.batchNorm, 24, 48, kernel_size=3, stride=1)
        self.conv3   = depthwise_separable_conv(self.batchNorm, 48, 96, kernel_size=3, stride=1)
        self.conv3_1 = dds_conv(self.batchNorm, 96, 96)
        self.conv4   = depthwise_separable_conv(self.batchNorm, 96, 192, stride=1)
        self.conv4_1 = dds_conv(self.batchNorm, 192, 192, dilation=4)
        self.conv5   = depthwise_separable_conv(self.batchNorm, 192, 192, stride=1)
        self.conv5_1 = dds_conv(self.batchNorm, 192, 192)
        # self.conv6   = depthwise_separable_conv(self.batchNorm, 192, 384, stride=2)
        # self.conv6_1 = depthwise_separable_conv(self.batchNorm, 384, 384)

        # self.conv4 = depthwise_separable_conv(self.batchNorm, 96, 192, stride=1)
        # self.conv4_1 = depthwise_separable_conv(self.batchNorm, 192, 192)
        # self.conv5 = depthwise_separable_conv(self.batchNorm, 192, 192, stride=1)
        # self.conv5_1 = depthwise_separable_conv(self.batchNorm, 192, 192)

        # self.deconv5 = deconv(384,192)
        # self.deconv4 = deconv(192,192)
        # self.deconv3 = deconv(96,48)
        # self.deconv2 = deconv(48,24)

        # self.predict_flow6 = predict_flow(192)
        # self.predict_flow5 = predict_flow(194)
        # self.predict_flow4 = predict_flow(194)
        # self.predict_flow3 = predict_flow(48)
        self.predict_flow2 = predict_flow(192)

        # self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.dequant = torch.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(self.quant(x)))
        # print("Out_conv2 size is ", out_conv2.size())
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        # print("Out_conv3 size is ", out_conv3.size())
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        # print("Out_conv4 size is ", out_conv4.size())
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        # print("Out_conv5 size ", out_conv5.size())

        # out_conv5 = self.conv5_1(self.conv5(out_conv4))
        # # out_conv6 = self.conv6_1(self.conv6(out_conv5))
        #
        # # flow6       = self.predict_flow6(out_conv6)
        # # flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        # # out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)
        # #
        # # concat5 = torch.add((out_conv5,flow6_up),1)
        # # flow5       = self.predict_flow5(concat5)
        # # flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        # out_deconv4 = crop_like(self.deconv4(out_conv5), out_conv4)
        #
        # concat4 = torch.add(out_conv4,out_deconv4)
        # out_deconv3 = self.deconv3(out_conv3)
        # print("Out_deconv3 size is ", out_deconv3.size())
        #
        # concat3 = torch.add(out_conv3,out_deconv3)
        # out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        # out_deconv2 = self.conv3_1(self.conv3(self.conv2(self.conv1(self.quant(x)))))

        flow2 = self.dequant(self.predict_flow2(out_conv5)) # self.predict_flow2(concat2)
        # print("flow2 size is ", flow2.size())

        # flow4, flow5, flow6 = flow2, flow2, flow2

        if self.training:
            return flow2#,flow3,flow4,flow5,flow6
        else:
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def tinyflownet(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = TinyFlowNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def tinyflownet_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = TinyFlowNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
