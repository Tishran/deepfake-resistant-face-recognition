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
from dotenv import load_dotenv

from oml import datasets as d
from oml.losses import TripletLossWithMiner
from oml.miners import AllTripletsMiner
from oml.samplers import BalanceSampler

from validation import (
    print_metrics,
    calc_metrics,
    val_inference,
    gen_query_gallery_pairs,
    get_ground_truth,
)

load_dotenv()

MODEL_WEIGHTS_SAVE_PATH = "./model_weights/"
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

epochs = 2


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    wandb.login(key=WANDB_API_KEY)

    fix_seed(seed=42)

    if not os.path.exists(
        os.path.join(MODEL_WEIGHTS_SAVE_PATH, OML_MODEL_NAME)
    ):
        os.makedirs(os.path.join(MODEL_WEIGHTS_SAVE_PATH, OML_MODEL_NAME))

    model = (
        OML_EXTRACTOR[OML_MODEL_NAME]
        .from_pretrained(OML_MODEL_NAME)
        .to(DEVICE)
        .train()
    )

    transform = t.Compose(
        [
            t.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
            t.CenterCrop(CROP_SIZE),
            t.ToTensor(),
            t.Normalize(mean=MEAN, std=STD),
        ]
    )

    df_train = pd.read_csv("./data_csv/reorganized_train.csv")
    df_val = pd.read_csv("./data_csv/val.csv")
    train = d.ImageLabeledDataset(df_train, transform=transform)

    df_val = gen_query_gallery_pairs(df_val)
    df_gt_val = get_ground_truth(df_val)

    val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
    sampler = BalanceSampler(train.get_labels(), n_labels=20, n_instances=4)

    # wandb init
    wandb.init(
        project="Kryptonite_ML_Challenge",
        config={
            "architecture": OML_MODEL_NAME,
            "epochs": epochs,
            "optimizer": optimizer.__class__.__name__,
            "loss": criterion.__class__.__name__,
        },
    )
    # wandb init

    def training():
        for epoch in range(epochs):
            pbar = tqdm(DataLoader(train, batch_sampler=sampler))
            pbar.set_description(f"epoch: {epoch}/{epochs}")
            for batch in pbar:
                embeddings = model(batch["input_tensors"].to(DEVICE))
                loss = criterion(embeddings, batch["labels"].to(DEVICE))
                loss.backward()

                wandb.log({"train loss": loss.item()})

                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix(criterion.last_logs)

            embeddings, df_pred_val = val_inference(model, val, df_val)
            rank_metrics = calc_metrics(embeddings, df_gt_val, df_pred_val, val)

            wandb.log(rank_metrics)

            print_metrics(rank_metrics)

    training()
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_WEIGHTS_SAVE_PATH, f"{OML_MODEL_NAME}/model.pth"),
    )
