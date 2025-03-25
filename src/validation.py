"""
validation.py
-------------

Module for model validation and metric computation, including retrieval metrics and EER.

Functions:
    - val_inference: Performs inference using the model, computes embeddings, and generates submission predictions.
"""

from torch.nn import functional as F
from oml.inference import inference

from src.inference import create_sample_sub
from data_utils import gen_pair_ids
from configs.config import DEVICE


def val_inference(model, data, df):
    """
    Performs inference with the model and generates a submission DataFrame with predicted similarity scores.

    Parameters:
        model: The trained PyTorch model.
        data: The dataset for inference.
        df (DataFrame): DataFrame with validation information.

    Returns:
        tuple: (embeddings, submission DataFrame), where embeddings are the computed representations and
               the submission DataFrame contains similarity predictions for each pair.
    """

    embeddings = inference(
        model, data, batch_size=512, num_workers=6, verbose=True
    )
    e1 = embeddings[::2]
    e2 = embeddings[1::2]
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()

    pair_ids = df["label"].to_list()[::2]
    pair_ids = gen_pair_ids(pair_ids)

    sub_df = create_sample_sub(pair_ids, sim_scores)

    return embeddings, sub_df
