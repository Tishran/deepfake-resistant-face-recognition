# model experiments
baseline: 0.382 EER

OML lib seems interesing, gonna keep exploring it


1. trying different pretrainerd extractors (without or with training):
* tried vitb14_dinov2 - it is huge, takes about 50 min to inference on test_public.csv with batch_size=1024 and takes about 14 GB of GPU memory, could not wait that time
* resnet18_imagenet1k_v1 - takes about 8 minutes to inferense, without any training gives 0.362 on public test
* vits16_dino - checking pretrained, do i need model.pth? got 0.35 on public, probably bad training of model.pth
* vits16_dino - 0.0809 on test with training on reorganized data (balance sampler n_labels=20, n_instances=5), 1 epoch
* vits16_din - 0.0716 on test with training on reorganized data (balance sampler n_labels=20, n_intances=5), 1 epoch
* vits16_din - 0.0868 on test with training on reorganized data (balance sampler n_labels=25 n_instances=7), train on 1 epoch 18 minutes, inference on test 11 minutes, 1 epoch
* resnet18_imagenet1k_v1 - 0.0653 on test with training on reorganized data (balance sampler n_labels=25 n_intsances=7), train on 1 epoch
* resnet18_imagenet1k_v1 - 0.0571 on test with training on reorganized data (balance sampler n_labels=25 n_intsances=3)
* vits16_inshop - 0.07 on test

conlusion: maybe ViT model overfits more on train dataset than resnet
futher experiments will run on balance sampler with n_labels=20 and n_instances=4

2. reorganized data (dropped rows that are unique in their classes) - it gave 0.0571 EER on public with resnet18_imagenet1k_v1 (other things are from baseline)

3. Augmentations:
Added them consiquently (all using best model - see below)
* RandomHorizontalFlip(p=0.4) - 0.008250248563998317 eer on val

4. Losses:
(all run on best model - see below)
* SoftTripletLoss - 0.0058 eer on val
* CosineTripletLoss (reduction "mean") - 0.00849980514814729 eer on val, 0.006 on test
* CosineTripletLoss (reduction "mean") - 0.007050144767020873 eer on val

NOTE: correleation between val and test EER metrics seems to exist :)

**best submission**: resnet34_imagenet1k_v1 on validation set EER: 
0.00524998907155482, on test set EER: 0.0486