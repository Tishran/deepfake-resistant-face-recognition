import os
import random
import torch
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as t
from torchvision.transforms import InterpolationMode
from config import *

from oml import datasets as d
from oml.losses import TripletLossWithMiner
from oml.miners import AllTripletsMiner
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.samplers import BalanceSampler

from my_secrets import WANDB_API_KEY

from loss import CosineTripletLossWithMiner
from validation import (
    print_metrics,
    calc_metrics,
    val_inference,
    gen_query_gallery_pairs,
    get_ground_truth,
)

MODEL_WEIGHTS_SAVE_PATH = "./model_weights/"

device = "cuda"
epochs = 3


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    wandb.login(key=WANDB_API_KEY)

    wandb.init(
        project="Kryptonite_OML",
        config={
            "architecture": OML_MODEL_NAME,
            "epochs": epochs,
            "optimizer": "Adam",
            "loss": "TripletLoss",
        },
    )

    fix_seed(seed=42)

    if not os.path.exists(os.path.join(MODEL_WEIGHTS_SAVE_PATH, OML_MODEL_NAME)):
        os.makedirs(os.path.join(MODEL_WEIGHTS_SAVE_PATH, OML_MODEL_NAME))

    # model = (
    #     OML_EXTRACTOR[OML_MODEL_NAME].from_pretrained(OML_MODEL_NAME).to(device).train()
    # )

    model = (
        ResnetExtractor(
            weights=None,
            arch="resnet34",
            gem_p=None,
            remove_fc=True,
            normalise_features=True,
        )
        .to(device)
        .train()
    )

    transform = t.Compose(
        [
            t.Resize(IM_SIZE, interpolation=InterpolationMode.BICUBIC),
            t.CenterCrop(CROP_SIZE),
            # t.RandomHorizontalFlip(p=0.4),
            t.ToTensor(),
            t.Normalize(mean=MEAN, std=STD),
        ]
    )

    df_train, df_val = pd.read_csv("reorganized_train.csv"), pd.read_csv("val.csv")
    train = d.ImageLabeledDataset(df_train, transform=transform)

    df_val = gen_query_gallery_pairs(df_val)
    df_gt_val = get_ground_truth(df_val)

    val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

    optimizer = Adam(model.parameters(), lr=1e-4)
    # criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
    criterion = CosineTripletLossWithMiner(
        0.1, AllTripletsMiner(), reduction="mean", need_logs=True
    )
    sampler = BalanceSampler(train.get_labels(), n_labels=20, n_instances=4)

    def training():
        for epoch in range(epochs):
            pbar = tqdm(DataLoader(train, batch_sampler=sampler))
            pbar.set_description(f"epoch: {epoch}/{epochs}")
            for batch in pbar:
                embeddings = model(batch["input_tensors"].to(device))
                loss = criterion(embeddings, batch["labels"].to(device))
                loss.backward()

                wandb.log({"train loss": loss.item()})

                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix(criterion.last_logs)

            embeddings, df_pred_val = val_inference(model, val, df_val)
            rank_metrics, eer_metric = calc_metrics(
                embeddings, df_gt_val, df_pred_val, val
            )
            print_metrics(rank_metrics, eer_metric)

    training()
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_WEIGHTS_SAVE_PATH, f"{OML_MODEL_NAME}/model.pth"),
    )
