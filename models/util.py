import torch.nn as nn
import torch.nn.functional as F

try:
    from spatial_correlation_sampler import spatial_correlation_sample
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load custom correlation module"
                      "which is needed for FlowNetC", ImportWarning)


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(0.1, inplace=True)
        )


def depthwise_separable_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        raise('Error')
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True, groups=in_planes),  # depth-wise
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

def dds_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dilation=2):
    if batchNorm:
        raise('Error')
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size-1)//2+round(dilation-1), bias=True, groups=in_planes),  # depth-wise
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        # nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True)
        # nn.LeakyReLU(0.1, inplace=True)
    )


def depthwise_separable_deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1,
                  bias=True, groups=in_planes),  # depth-wise
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear')
        # nn.UpsamplingBilinear2d(scale_factor=2),
        # nn.LeakyReLU(0.1, inplace=True)
    )


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]
