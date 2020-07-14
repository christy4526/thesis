import torch
from torch import nn


class KDNet(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student

        self.teacher.eval()

    def train(self, mode=True):
        self.student.train(mode)

    def parameters(self, recurse=True):
        return self.student.parameters(recurse)

    def forward(self, x):
        with torch.no_grad():
            output_teacher = self.teacher(x)

        output_student = self.student(x)
        return {
            'output_student': output_student,
            'output_teacher': output_teacher}


class KDExplanationNet(KDNet):
    def __init__(self, teacher, student, freeze_backbone=True):
        super().__init__(teacher, student)
        self.freeze_backbone = freeze_backbone

    def train(self, mode=True):
        self.student.train(mode)
        if self.freeze_backbone:
            self.student.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            output_teacher = self.teacher(x)

        output_student, explanation = self.student(x)

        x_masked = x * explanation
        output_masked = self.teacher(x_masked)

        return {
            'output_student': output_student,
            'explanation': explanation,
            'output_masked': output_masked,
            'output_teacher': output_teacher}


class Classification(nn.Module):
    def __init__(self, student, **kwargs):
        super().__init__()
        self.student = student

    def forward(self, x):
        return self.student(x)
