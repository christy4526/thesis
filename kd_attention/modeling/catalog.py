import os
from torchvision import models as M
from torchvision.models.resnet import model_urls as resnet_model_urls
# from kd_attention.utils import remove_strs
from .meta import KDExplanationNet, KDNet, Classification
from .student import resnet18_cbam, resnet50_cbam
from .student import resnet18_cbam_explanation, resnet50_cbam_explanation
from .student import tiny_ex  # debuging


class Catalog:
    teachers = {
        'I-WR101-2': (M.wide_resnet101_2, resnet_model_urls['wide_resnet101_2']),
        'I-WR50-2': (M.wide_resnet50_2, resnet_model_urls['wide_resnet50_2']),
        'I-RX101-32x8d': (M.resnext101_32x8d, resnet_model_urls['resnext101_32x8d']),
        'I-RX50-32x4d': (M.resnext50_32x4d, resnet_model_urls['resnext50_32x4d']),
        'I-R152': (M.resnet152, resnet_model_urls['resnet152']),
        'I-R101': (M.resnet101, resnet_model_urls['resnet101']),
        'I-R50': (M.resnet50, resnet_model_urls['resnet50']),
        'I-R34': (M.resnet34, resnet_model_urls['resnet34']),
        'I-R18': (M.resnet18, resnet_model_urls['resnet18']),
        'Cars-R50': (M.resnet50, 'resnet50-cars.pth'),
        'CUB-R50': (M.resnet50, 'resnet50-cub200.pth'),
        'AirV-R50': (M.resnet50, 'resnet50-aircraft-variant.pth'),
        'AirF-R50': (M.resnet50, 'resnet50-aircraft-family.pth'),
        'AirM-R50': (M.resnet50, 'resnet50-aircraft-manufacturer.pth'),
    }
    students = {
        'R18': M.resnet18,
        'R50': M.resnet50,
        'R18-CBAM': resnet18_cbam,
        'R50-CBAM': resnet50_cbam,
        'R18-CBAM-E': resnet18_cbam_explanation,
        'R50-CBAM-E': resnet50_cbam_explanation,
        'R50-CBAM-TE': tiny_ex,
    }
    metas = {
        'KDNet': KDNet,
        'KDExplanationNet': KDExplanationNet,
        'Classification': Classification
    }
