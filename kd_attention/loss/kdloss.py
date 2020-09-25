import torch
from torch import nn
from torch.nn import functional as F


def prod(lst):
    if len(lst) == 0:
        return 1

    return lst[0] * prod(lst[1:])


def kl_div_with_softmax(inputs, target):
    return F.kl_div(
        torch.log_softmax(inputs, dim=1),
        torch.softmax(target, dim=1),
        reduction='batchmean'
    )


class KLDivwithSoftmax(nn.Module):
    def forward(self, inputs, target):
        return kl_div_with_softmax(inputs, target)


class _KDLoss(nn.Module):
    def __init__(self, alpha=1., temperature=2.5):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, inputs, target):
        raise NotImplementedError


class KnowledgeDistillationLoss(_KDLoss):
    def forward(self, inputs, target):
        output_student = inputs['output_student']
        output_teacher = inputs['output_teacher']
        T = self.temperature

        soft_loss = self.alpha * kl_div_with_softmax(
            output_student / T, output_teacher / T
        )
        lossdict = {'soft_loss': soft_loss}
        if self.alpha < 1.:
            hard_loss = F.cross_entropy(output_student, target)
            hard_loss = (1 - self.alpha) * hard_loss
            lossdict['hard_loss'] = hard_loss

        return lossdict


class KDExplanationLoss(nn.Module):
    def __init__(self, l1_strength=0.0005):
        super().__init__()
        self.l1_strength = l1_strength
        self.kdloss = KLDivwithSoftmax()
        self.maskloss = KLDivwithSoftmax()
        self.l1loss = lambda inputs, target: l1_strength * \
            F.l1_loss(inputs, target) * prod(inputs.shape[1:])

    def forward(self, inputs, target):
        kd_loss = self.kdloss(
            inputs['output_student'], inputs['output_teacher']
        )

        mask_loss = self.maskloss(
            inputs['output_masked'], inputs['output_teacher']
        )

        l1_loss = self.l1loss(
            inputs['explanation'],
            torch.zeros_like(inputs['explanation'], requires_grad=False),
        )

        return dict(kd_loss=kd_loss, mask_loss=mask_loss, l1_loss=l1_loss)


