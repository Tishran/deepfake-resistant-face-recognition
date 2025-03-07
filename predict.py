import torch
import pandas as pd

from torch.nn import functional as F
from oml import datasets as d
from oml.inference import inference
from oml.metrics import calc_retrieval_metrics_rr

from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding

from eer import compute_eer
from make_submission import create_sample_sub

device = "cuda"


def gen_query_gallery_pairs(df):
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
    paired_df = paired_df.sort_values(["pair_id", "is_query"], ascending=[True, False])
    paired_df = paired_df.drop(columns="pair_id").reset_index(drop=True)

    return paired_df


def gen_pair_ids(ids):
    pass


def test_inference(model, data):
    embeddings = inference(model, data, batch_size=512, num_workers=6, verbose=True)
    e1 = embeddings[::2]
    e2 = embeddings[1::2]
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()

    pair_ids = df_test["label"].apply(lambda x: f"{x:08d}").to_list()
    pair_ids = pair_ids[::2]

    sub_df = create_sample_sub(pair_ids, sim_scores)

    return embeddings, sub_df


def calc_metrics(embeddigns, gt_df, sub_df):
    rr = RetrievalResults.from_embeddings(embeddings, test, n_items=10)
    rr = AdaptiveThresholding(n_std=2).process(rr)
    rr.visualize(query_ids=[2, 1], dataset=test, show=True)
    rank_metrics = calc_retrieval_metrics_rr(rr, map_top_k=(10,), cmc_top_k=(1, 5, 10))

    eer_metric = compute_eer(gt_df, sub_df)

    return rank_metrics, eer_metric


def print_metrics(rank_metrics, eer_metric):
    print(f"EER: {eer_metric}")

    for metric_name in rank_metrics.keys():
        for k, v in rank_metrics[metric_name].items():
            print(f"{metric_name}@{k}: {v.item()}")


# if __name__ == "__main__":
#     model = ViTExtractor.from_pretrained("vits16_dino")
#     state_dict = torch.load("model.pth", map_location="cpu")
#     model.load_state_dict(state_dict)
#     model = model.to(device).eval()

#     transform, _ = get_transforms_for_pretrained("vits16_dino")

#     df_test = pd.read_csv("val.csv")
#     df_test = gen_query_gallery_pairs(df_test)
#     test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)

#     predict()
