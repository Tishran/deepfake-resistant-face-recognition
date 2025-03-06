# model experiments
baseline: 0.382 EER

OML lib seems interesing, gonna keep exploring it


1. trying different pretrainerd extractors (without training):
* tried vitb14_dinov2 - it is huge, takes about 50 min to inference on test_public.csv with batch_size=1024 and takes about 14 GB of GPU memory, could not wait that time
* resnet18_imagenet1k_v1 - takes about 8 minutes to inferense, without any training gives 0.362 on public test
* vits16_dino - checking pretrained, do i need model.pth? got 0.35 on public, probably bad training of model.pth
