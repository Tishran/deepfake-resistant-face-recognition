"""
Configuration for model selection and feature extractors in Open-Metric-Learning (OML).

Global Variables:
    DEVICE: current available device.

    OML_MODEL_NAME (str):
        Default model architecture to use. Supported options:
        - "resnet34_imagenet1k_v1": ResNet-34 pretrained on ImageNet-1k (default)
        - "resnet18_imagenet1k_v1": ResNet-18 pretrained on ImageNet-1k
        - "vits16_dino": Vision Transformer (ViT-S/16) pretrained with DINO method.

    OML_EXTRACTOR (dict):
        Mapping between model names and their corresponding feature extractor classes.
        Used by other modules to instantiate the correct model architecture.
"""

import torch
from oml.models import ViTExtractor, ResnetExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OML_MODEL_NAME = "resnet34_imagenet1k_v1"

OML_EXTRACTOR = {
    "vits16_dino": ViTExtractor,
    "resnet18_imagenet1k_v1": ResnetExtractor,
    "resnet34_imagenet1k_v1": ResnetExtractor,
}
