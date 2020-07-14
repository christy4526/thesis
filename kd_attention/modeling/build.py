import os
import torch
from torchvision.models.utils import load_state_dict_from_url
from .catalog import Catalog


def _build_generic(model_cls, state_dict, num_classes, weight_root='weight', **kwargs):
    model = model_cls(num_classes=num_classes, **kwargs)
    if state_dict is None:
        return model
    if state_dict.startswith('http'):
        state_dict = load_state_dict_from_url(state_dict)
    else:
        state_dict = torch.load(os.path.join(weight_root, state_dict), 'cpu')
    model.load_state_dict(state_dict)
    return model


def build_teacher(arch, num_classes=1000, weight_root='weights'):
    model = _build_generic(*Catalog.teachers[arch], num_classes, weight_root)
    model.eval()
    return model


def build_student(arch, num_classes=1000, freeze_backbone=True):
    model = _build_generic(Catalog.students[arch], None, num_classes)
    model.train()
    return model


def build_meta(name, teacher_arch, student_arch, weight_root='weights', num_classes=1000, freeze_backbone=True):
    if teacher_arch != 'none':
        teacher = build_teacher(teacher_arch, num_classes, weight_root)
    else:
        teacher = None
    student = build_student(student_arch, num_classes,
                            freeze_backbone=freeze_backbone)

    meta = Catalog.metas[name]
    model = meta(teacher=teacher, student=student)
    return model
