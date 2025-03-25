"""
make_submission.py
-------------

Generates submission files for model predictions using cosine similarity scores.

Functions:
    - create_sample_sub: Formats predictions into a submission DataFrame.
    - run_inference: Runs inference on test data, computes similarity scores, and saves results.
"""

import os
from typing import List
import torch
import pandas as pd
from torch.nn import functional as F
from oml import datasets as d
from oml.inference import inference
from configs.config import OML_MODEL_NAME, OML_EXTRACTOR

from oml.registry import get_transforms_for_pretrained

OUTPUT_PATH = "./results"


def create_sample_sub(pair_ids: List[str], sim_scores: List[float]):
    """
    Creates a submission DataFrame in the required format.

    Parameters:
        pair_ids (List[str]): List of pair IDs (e.g., ["00000001", "00000002"]).
        sim_scores (List[float]): Predicted similarity scores.

    Returns:
        pd.DataFrame: Submission DataFrame with columns ["pair_id", "similarity"].
    """

    sub_sim_column = "similarity"
    id_column = "pair_id"
    return pd.DataFrame({id_column: pair_ids, sub_sim_column: sim_scores})


def run_inference():
    """
    Generates a submission file using a trained model.

    Workflow:
        1. Load model weights from `./model_weights/`
        2. Compute embeddings for test pairs
        3. Calculate cosine similarity scores
        4. Save results to `./submissions/`

    Requires:
        - `test.csv` with columns ["path", "label"]
        - Pretrained model defined in `config.py`
    """
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    device = "cuda"
    test_path = "./data/test.csv"

    model = OML_EXTRACTOR[OML_MODEL_NAME].from_pretrained(OML_MODEL_NAME)
    state_dict = torch.load(
        f"./model_weights/{OML_MODEL_NAME}/model.pt", map_location="cuda"
    )
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    transform, _ = get_transforms_for_pretrained(OML_MODEL_NAME)

    df_test = pd.read_csv(test_path)
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)
    embeddings = inference(
        model, test, batch_size=1024, num_workers=3, verbose=True
    )

    e1 = embeddings[::2]
    e2 = embeddings[1::2]
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()

    pair_ids = df_test["label"].apply(lambda x: f"{x:08d}").to_list()
    pair_ids = pair_ids[::2]

    sub_df = create_sample_sub(pair_ids, sim_scores)
    sub_df.to_csv(f"{OUTPUT_PATH}/result.csv", index=False)


if __name__ == "__main__":
    run_inference()
