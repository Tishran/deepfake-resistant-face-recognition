from oml.models import ViTExtractor, ResnetExtractor

OML_MODEL_NAME = "resnet34_imagenet1k_v1"

OML_EXTRACTOR = {
    "vits16_dino": ViTExtractor,
    "resnet18_imagenet1k_v1": ResnetExtractor,
    "resnet34_imagenet1k_v1": ResnetExtractor,
}
