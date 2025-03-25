"""
metrics.py
-------------

Implements evaluation metrics for face recognition.

Functions:
    - compute_eer: Computes the Equal Error Rate (EER) given true labels and predicted scores.
    - calc_metrics: Calculates metrics including retrieval metrics and the EER.
    - print_metrics: Prints the computed metrics to the console.
"""

import numpy as np

from sklearn.metrics import roc_curve
from oml.metrics import calc_retrieval_metrics_rr
from oml.retrieval import RetrievalResults, AdaptiveThresholding


def compute_eer(label, pred, positive_label=1):
    """
    Computes the Equal Error Rate (EER) given true labels and predicted scores.

    Parameters:
        label (list or numpy.array): True binary labels.
        pred (list or numpy.array): Predicted scores (or probabilities).
        positive_label (int, optional): The label for the positive class (default is 1).

    Returns:
        float: The computed EER.
    """

    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


def calc_metrics(embeddings, gt_df, sub_df, data, tag="val"):
    """
    Calculates metrics including retrieval metrics and the EER.

    Parameters:
        embeddings: Computed embeddings.
        gt_df (DataFrame): DataFrame with ground truth data.
        sub_df (DataFrame): DataFrame with predictions.
        data: The dataset for inference.
        tag (str): A tag for logging metrics (e.g., 'val').

    Returns:
        dict: A dictionary containing the computed metrics.
    """

    rr = RetrievalResults.from_embeddings(embeddings, data, n_items=10)
    rr = AdaptiveThresholding(n_std=2).process(rr)
    rr.visualize(query_ids=[10, 15, 20, 25], dataset=data, show=True)
    rank_metrics = calc_retrieval_metrics_rr(
        rr, map_top_k=(10,), cmc_top_k=(1, 5, 10)
    )

    eer_metric = compute_eer(gt_df, sub_df)

    rank_metrics_dict = {}
    for metric_name in rank_metrics.keys():
        for k, v in rank_metrics[metric_name].items():
            rank_metrics_dict[f"{tag}_{metric_name}_{k}"] = v.item()

    rank_metrics_dict[f"{tag}_eer"] = eer_metric

    return rank_metrics_dict


def print_metrics(rank_metrics):
    """
    Prints the computed metrics to the console.

    Parameters:
        rank_metrics (dict): A dictionary of metrics.
    """

    for k, v in rank_metrics.items():
        print(f"{k}: {v}")
