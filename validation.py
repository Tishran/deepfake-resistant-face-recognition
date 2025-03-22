import torch
import pandas as pd
import json

from collections import defaultdict
from torch.nn import functional as F
from oml import datasets as d
from oml.inference import inference
from oml.metrics import calc_retrieval_metrics_rr

from oml.retrieval import RetrievalResults, AdaptiveThresholding

from eer import compute_eer
from make_submission import create_sample_sub

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_pair_ids(ids):
    ids_set = defaultdict(int)

    pair_ids = list()
    for id in ids:
        pair_ids.append(f"{id:08d}_{ids_set[id]}")
        ids_set[id] += 1

    return pair_ids


def get_ground_truth(df):
    with open(f"./data/train/meta.json", "r") as f:
        fake_indicators = json.load(f)

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


def val_inference(model, data, df):
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


def calc_metrics(embeddings, gt_df, sub_df, data, tag="val"):
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
    for k, v in rank_metrics.items():
        print(f"{k}: {v}")
