"""
data_utils.py
-------------

Utilities for data preprocessing and ground truth generation.

Functions:
    - gen_pair_ids: Generates unique pair identifiers based on the input list of identifiers.
    - get_ground_truth: Constructs the ground truth using fake indicators from a metadata JSON file.
    - gen_query_gallery_pairs: Creates query-gallery pairs of dataset.
"""

import json
import pandas as pd

from collections import defaultdict

from src.inference import create_sample_sub


def gen_pair_ids(ids):
    """
    Generates unique pair identifiers based on the input list of identifiers.

    Parameters:
        ids (iterable): A list of identifiers.

    Returns:
        list: A list of unique pair IDs.
    """

    ids_set = defaultdict(int)

    pair_ids = list()
    for id in ids:
        pair_ids.append(f"{id:08d}_{ids_set[id]}")
        ids_set[id] += 1

    return pair_ids


def get_ground_truth(df):
    """
    Constructs the ground truth for validation using fake indicators from a metadata JSON file.

    Parameters:
        df (DataFrame): DataFrame containing image paths and labels.

    Returns:
        DataFrame: A submission DataFrame with pair IDs and binary similarity scores.
    """

    with open(f"./data/train/meta.json", "r") as f:
        fake_indicators = json.load(f)

    # Extract image paths (using the last two segments of the path)
    paths = df["path"].apply(lambda x: "/".join(x.split("/")[-2:])).to_list()
    sim_scores = list()
    for i in range(0, len(paths) - 1, 2):
        sim_scores.append(
            int(fake_indicators[paths[i]] == fake_indicators[paths[i + 1]])
        )

    pair_ids = df["label"].to_list()[::2]
    pair_ids = gen_pair_ids(pair_ids)

    return create_sample_sub(pair_ids, sim_scores)


def gen_query_gallery_pairs(df):
    """
    Creates query-gallery pairs of dataset.

    Parameters:
        df (DataFrame): DataFrame containing information on image paths, labels, and query/gallery flags.

    Returns:
        DataFrame: A DataFrame with merged query-gallery pairs.
    """

    split_val = df.loc[0, "split"]
    queries = df[df["is_query"]][["label", "path"]]
    galleries = df[df["is_gallery"]][["label", "path"]]

    merged_pairs = pd.merge(
        queries, galleries, on="label", suffixes=("_query", "_gallery")
    )

    merged_pairs["pair_id"] = merged_pairs.index

    query_rows = merged_pairs[["pair_id", "label", "path_query"]].rename(
        columns={"path_query": "path"}
    )
    query_rows["is_query"] = True
    query_rows["is_gallery"] = False

    gallery_rows = merged_pairs[["pair_id", "label", "path_gallery"]].rename(
        columns={"path_gallery": "path"}
    )
    gallery_rows["is_query"] = False
    gallery_rows["is_gallery"] = True

    paired_df = pd.concat([query_rows, gallery_rows])
    paired_df["split"] = split_val
    paired_df = paired_df[
        ["pair_id", "label", "path", "split", "is_query", "is_gallery"]
    ]
    paired_df = paired_df.sort_values(
        ["pair_id", "is_query"], ascending=[True, False]
    )
    paired_df = paired_df.drop(columns="pair_id").reset_index(drop=True)

    return paired_df
