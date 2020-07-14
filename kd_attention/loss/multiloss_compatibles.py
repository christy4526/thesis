from torch import nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, inputs, target):
        return {'celoss': super().forward(inputs, target)}
