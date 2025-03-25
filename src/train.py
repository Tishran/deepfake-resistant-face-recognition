"""
train.py
--------

Module for training a face recognition/authentication model.

Functionality:
    - Fixing random seed for reproducibility.
    - Loading and preparing training and validation data.
    - Training the model using the Adam optimizer, TripletLossWithMiner, and a balanced sampler.
    - Logging experiments via wandb and saving model weights.
"""

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
from dotenv import load_dotenv

from oml import datasets as d
from oml.losses import TripletLossWithMiner
from oml.miners import AllTripletsMiner
from oml.samplers import BalanceSampler

from validation import val_inference
from metrics import print_metrics, calc_metrics
from data_utils import get_ground_truth, gen_query_gallery_pairs
from configs.config import *

load_dotenv()

MODEL_WEIGHTS_SAVE_PATH = "./model_weights/"
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

IMG_SIZE = 256
CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

epochs = 2


def fix_seed(seed: int):
    """
    Fixes random seed for reproducibility.

    Parameters:
        seed (int): The seed value for random number generators.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    wandb.login(key=WANDB_API_KEY)

    fix_seed(seed=42)

    # Create directory for saving model weights
    model_save_dir = os.path.join(MODEL_WEIGHTS_SAVE_PATH, OML_MODEL_NAME)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Initialize the model from a pretrained variant, move to device, and set to train mode
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

    # Load training and validation data
    df_train = pd.read_csv("./data/reorganized_train.csv")
    df_val = pd.read_csv("./data/val.csv")
    train = d.ImageLabeledDataset(df_train, transform=transform)

    df_val = gen_query_gallery_pairs(df_val)
    df_gt_val = get_ground_truth(df_val)

    val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
    sampler = BalanceSampler(train.get_labels(), n_labels=20, n_instances=4)

    # Initialize wandb for experiment tracking
    wandb.init(
        project="Kryptonite_ML_Challenge",
        config={
            "architecture": OML_MODEL_NAME,
            "epochs": epochs,
            "optimizer": optimizer.__class__.__name__,
            "loss": criterion.__class__.__name__,
        },
    )

    def training():
        """
        Training loop for the model.
        """

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

            # Perform validation after each epoch
            embeddings, df_pred_val = val_inference(model, val, df_val)
            rank_metrics = calc_metrics(embeddings, df_gt_val, df_pred_val, val)

            wandb.log(rank_metrics)

            print_metrics(rank_metrics)

    training()

    # Save the trained model weights
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_WEIGHTS_SAVE_PATH, f"{OML_MODEL_NAME}/model.pt"),
    )
