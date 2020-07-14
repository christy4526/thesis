import torch
import itertools
from torch import nn

from .modules import conv1x1
from .modules import GatedMerger, ParameterizedSigmoid
from .modules import BottleneckTransposed, BasicBlockTransposed
from .modules import BottleneckCBAM, BasicBlockCBAM


class ResNetCBAM(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 return_attention=True, return_features=False):
        super().__init__()
        self.return_attention = return_attention
        self.return_features = return_features
        self.nblocks = layers
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(block, 64, layers[0]))
        self.layers.append(self._make_layer(block, 128, layers[1], stride=2))
        self.layers.append(self._make_layer(block, 256, layers[2], stride=2))
        self.layers.append(self._make_layer(block, 512, layers[3], stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes,
                        planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes, stride,
                            downsample, return_attention=self.return_attention))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                return_attention=self.return_attention))

        return layers

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        attentions = []
        features = []
        for layer in self.layers:
            for block in layer:
                x = block(x)

                if self.return_attention:
                    x, attention = x
                    attentions.append(attention)
                if self.return_features:
                    features.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        out = x
        if self.return_attention:
            out = out, attentions
        if self.return_features:
            out = out, features
        return out
    forward = _forward


class Explanation(nn.Module):
    def __init__(self, backbone, blockT, layers, norm_layer=None):
        super().__init__()
        if not isinstance(backbone, ResNetCBAM):
            raise TypeError(f'I want ResNet not {type(backbone)}')
        self.backbone = backbone
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        _layers = []
        _layers.append(self._make_layer(blockT, 64, layers[0]))
        _layers.append(self._make_layer(blockT, 128, layers[1], stride=2))
        _layers.append(self._make_layer(blockT, 256, layers[2], stride=2))
        _layers.append(self._make_layer(blockT, 512, layers[3], stride=2,
                                        gate_off=True))
        self.layers = nn.ModuleList(reversed(_layers))

        self.convT1 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.convT2 = nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False)
        self.conv_final = nn.Conv2d(1, 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1, gate_off=False):
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            output_padding = 1 if stride == 2 else 0
            upsample = nn.Sequential(
                nn.ConvTranspose2d(planes * block.expansion, self.inplanes, 1,
                                   stride, output_padding=output_padding,
                                   bias=False),
                norm_layer(self.inplanes)
            )
        gates = []
        for _ in range(blocks - 1):
            gates.append(GatedMerger(planes * block.expansion))
        if not gate_off:
            gates.append(GatedMerger(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample,
                            self.groups, self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                norm_layer=norm_layer))
        return nn.ModuleDict({'gates': nn.ModuleList(gates),
                              'blocks': nn.ModuleList(reversed(layers))})

    def forward(self, x):
        cls_scores, attentions = self.backbone(x)
        att_gen = iter(reversed(attentions))

        for layer in self.layers:
            gates, blocks = layer['gates'], layer['blocks']
            if len(gates) != len(blocks):
                feature = blocks[0](next(att_gen))
                blocks = blocks[1:]

            for gate, block in zip(gates, blocks):
                feature = gate(next(att_gen), feature)
                feature = block(feature)

        explanation = self.convT1(feature)
        explanation = self.bn1(explanation)
        explanation = self.relu(explanation)

        explanation = self.convT2(explanation)
        explanation = self.relu(explanation)

        explanation = self.conv_final(explanation)
        explanation = explanation.sigmoid()

        return cls_scores, explanation


class TinyExplanation(nn.Module):  # TODO freeze_backbone
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.return_attention = True
        self.backbone.return_features = True

        self.inplanes = 64
        self.layers = nn.ModuleList([
            nn.Sequential(
                self._make_layer(512, 1024),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                self._make_layer(256, 512),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(512, 512, 3, 1, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                self._make_layer(128, 256),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            self._make_layer(64, 64),
        ])
        self.conv = nn.Sequential(
            nn.Conv2d(64, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.interpolate = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes, outplanes):
        block = BottleneckTransposed
        downsample = None
        if planes * block.expansion != outplanes:
            downsample = nn.Sequential(
                conv1x1(planes * block.expansion, outplanes),
                nn.BatchNorm2d(outplanes)
            )
        return block(outplanes, planes, upsample=downsample)

    def parameters(self, recurse=True):
        for l in [self.layers, self.conv[0]]:
            for param in l.parameters():
                yield param

    def forward(self, x):
        with torch.no_grad():
            (y_pred, attentions), features = self.backbone(x)

        x = features[-1] * attentions[-1]
        x = self.layers[0](x)
        x = x + features[-4] * attentions[-4]
        x = self.layers[1](x)
        x = x + features[-4-6] * attentions[-4-6]
        x = self.layers[2](x)
        x = self.layers[3](x)

        x = self.conv(x)
        x = self.interpolate(x)

        return y_pred, x


def resnet18_cbam(num_classes, *args, **kwargs):
    return ResNetCBAM(BasicBlockCBAM, [2, 2, 2, 2],
                      num_classes=num_classes, return_attention=False)


def resnet50_cbam(num_classes, *args, **kwargs):
    return ResNetCBAM(BottleneckCBAM, [3, 4, 6, 3],
                      num_classes=num_classes, return_attention=False)


def resnet18_cbam_explanation(num_classes, *args, **kwargs):
    block = BasicBlockCBAM
    layers = [2, 2, 2, 2]
    backbone = ResNetCBAM(block, layers, num_classes=num_classes)
    return Explanation(backbone, BasicBlockTransposed, layers)


def resnet50_cbam_explanation(num_classes, *args, **kwargs):
    block = BottleneckCBAM
    layers = [3, 4, 6, 3]
    backbone = ResNetCBAM(block, layers, num_classes=num_classes)
    return Explanation(backbone, BottleneckTransposed, layers)


def tiny_ex(num_classes, *args, **kwargs):
    block = BottleneckCBAM
    layers = [3, 4, 6, 3]
    backbone = ResNetCBAM(block, layers, num_classes=num_classes,
                          return_attention=True, return_features=True)
    model = TinyExplanation(backbone)

    return model

# TODO backbone freezing
