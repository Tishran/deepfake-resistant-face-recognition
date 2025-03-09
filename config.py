from oml.models import ViTExtractor, ResnetExtractor

OML_MODEL_NAME = "resnet34_imagenet1k_v1"

IM_SIZE = 256
CROP_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

OML_EXTRACTOR = {
    "vits16_dino": ViTExtractor,
    "resnet18_imagenet1k_v1": ResnetExtractor,
    "resnet34_imagenet1k_v1": ResnetExtractor,
}
