import os
from typing import List
import torch
import pandas as pd
from torch.nn import functional as F
from oml import datasets as d
from oml.inference import inference
from config import OML_MODEL_NAME, OML_EXTRACTOR

from oml.models import ViTExtractor, ResnetExtractor
from oml.registry import get_transforms_for_pretrained


OUTPUT_PATH = "./submissions"


def create_sample_sub(pair_ids: List[str], sim_scores: List[float]):
    sub_sim_column = "similarity"
    id_column = "pair_id"
    return pd.DataFrame({id_column: pair_ids, sub_sim_column: sim_scores})


def make_submission():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    device = "cuda"
    test_path = "test.csv"

    model = OML_EXTRACTOR[OML_MODEL_NAME].from_pretrained(OML_MODEL_NAME)
    state_dict = torch.load(
        f"./model_weights/{OML_MODEL_NAME}/model.pth", map_location="cuda"
    )
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    transform, _ = get_transforms_for_pretrained(OML_MODEL_NAME)

    df_test = pd.read_csv(test_path)
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)
    embeddings = inference(model, test, batch_size=1024, num_workers=3, verbose=True)

    e1 = embeddings[::2]
    e2 = embeddings[1::2]
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()

    pair_ids = df_test["label"].apply(lambda x: f"{x:08d}").to_list()
    pair_ids = pair_ids[::2]

    sub_df = create_sample_sub(pair_ids, sim_scores)
    sub_df.to_csv(f"{OUTPUT_PATH}/cosine_triplet_loss_mean.csv", index=False)


if __name__ == "__main__":
    make_submission()
