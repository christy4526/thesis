
import torch
from torch import nn
from torchvision.models.resnet import conv1x1, conv3x3, Bottleneck, BasicBlock


class ParameterizedSigmoid(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     weight = nn.Parameter(torch.ones(1))

    #     self.register_parameter('weight', weight)

    def _forward(self, x):
        w = 3.

        return x.mul(w).sigmoid()

    forward = _forward


class GlobalChannelWiseMaxPooling(nn.Module):
    '''
    Channel wise global max pooling module
    Unlike global max pooling(torch.nn.AdaptiveMaxPooling(1)), this module
    performs channel wise max pooling.

    Input:
        tensor shaped N C X Y ...
    Output:
        N 1 X Y ... tensor with max pooling
    '''

    def forward(self, x):
        x, _ = x.max(dim=1, keepdim=True)
        return x


class GlobalChannelWiseAvgPooling(nn.Module):
    '''Channel wise global average pooling module
    Unlike global average pooling(torch.nn.AdaptiveAvgPooling(1)), this module
    performs channel wise average pooling.

    Input:
        N C X Y ... tensor
    Output:
        N 1 X Y ... tensor with average pooling
    '''

    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        return x


class ChannelAttention(nn.Module):
    '''Channel attention module introduced in
    CBAM: Convolutional Block Attention Module (https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
    It performs both average and max pooling globally at spatial dimention
    with the same input `X`. And passing through to the MLP, sum and
    recfified by sigmoid to create channel attention.
    Finally, the original input multiplied by the attention is the output.

    Input:
        N C X Y ... tensor
    Output:
        N C X Y ... tensor forged by channel attention
    '''

    def __init__(self, inplanes, planes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(inplanes, planes),
            nn.ReLU(inplace=True),
            nn.Linear(planes, inplanes)
        )

    def forward(self, x):
        b_size = x.size(0)

        avg_chdes = self.gap(x).view(b_size, -1)
        avg_chdes = self.mlp(avg_chdes).view(b_size, -1, 1, 1)

        max_chdes = self.gmp(x).view(b_size, -1)
        max_chdes = self.mlp(max_chdes).view(b_size, -1, 1, 1)

        ch_descriptor = (max_chdes + avg_chdes).sigmoid()

        out = x * ch_descriptor

        return out


class SpatialAttention(nn.Module):
    '''Spatial attention module introduced in
    CBAM: Convolutional Block Attention Module (https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
    It performs global spatial average and max pooling. And passing through
    to the convolutional block. This procedure creates N 1 X Y ... tensor which
    is attention map.
    Finally, the original input multiplied by the attention is the output.

    Input:
        N C X Y ... tensor
    Output:
        N C X Y ... tensor forged by spartial attention.
    '''

    def __init__(self):
        super().__init__()

        self.gap = GlobalChannelWiseAvgPooling()
        self.gmp = GlobalChannelWiseMaxPooling()

        kernel_size = 7
        stride = 1
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(1, momentum=0.01)

    def forward(self, x):
        avg_spdes = self.gap(x)
        max_spdes = self.gmp(x)

        spatial_descriptor = torch.cat((avg_spdes, max_spdes), dim=1)

        spatial_descriptor = self.conv(spatial_descriptor)
        spatial_descriptor = self.bn(spatial_descriptor).sigmoid()

        out = x * spatial_descriptor

        return out


class CBAM(nn.Module):
    '''CBAM module introduced in
    CBAM: Convolutional Block Attention Module (https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
    It performs channel attention and spatial attention sequentially.

    Input:
        N C X Y ... tensor
    Output:
        N C X Y ... tensor forged by channel and spatial attention modules.
    '''

    def __init__(self, inplanes, planes):
        super().__init__()
        self.channel_attention = ChannelAttention(inplanes, planes)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)

        return out


class BottleneckCBAM(Bottleneck):
    '''Residual bottleneck block introduced in
    CBAM: Convolutional Block Attention Module (https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
    Attention mechanism have been added to the original procedure.
    This module is compatible with official torchvision implementation of ResNet
    '''

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, return_attention=True):
        super().__init__(inplanes, planes, stride=stride, downsample=downsample)
        planes = planes * self.expansion
        self.return_attention = return_attention
        self.attention = CBAM(planes, planes // 16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        before = out
        out = self.attention(out)
        after = out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        if self.return_attention:
            return out, self.relu(after - before)
        return out


class BasicBlockCBAM(BasicBlock):
    '''Residual bottleneck block introduced in
    CBAM: Convolutional Block Attention Module (https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
    Attention mechanism have been added to the original procedure.
    This module is compatible with official torchvision implementation of ResNet
    '''

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, return_attention=True):
        super().__init__(inplanes, planes, stride=stride, downsample=downsample)
        planes = planes * self.expansion
        self.attention = CBAM(planes, planes // 16)
        self.return_attention = return_attention

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        before = out
        out = self.attention(out)
        after = out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        if self.return_attention:
            return out, self.relu(after - before)
        return out


def convT3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    if stride == 1:
        conv = conv3x3(in_planes, out_planes, stride=stride,
                       groups=groups, dilation=dilation)
    else:
        conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                                  stride=stride, padding=dilation,
                                  groups=groups, bias=False, dilation=dilation)
    return conv


class BasicBlockTransposed(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(planes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convT3x3(planes, inplanes, stride)
        self.bn2 = norm_layer(inplanes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckTransposed(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(planes * self.expansion, width)
        self.bn1 = norm_layer(width)
        self.conv2 = convT3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, inplanes)
        self.bn3 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class GatedMerger(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, 1, 1, 1, 0)

    def forward(self, att, x):
        mask = x + att
        mask = self.conv(mask)
        mask = mask.sigmoid()

        x = att * mask
        return x
