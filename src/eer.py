"""
eer.py
------

Module for computing the Equal Error Rate (EER) – a metric commonly used to evaluate binary classifiers,
especially in tasks such as face recognition and authentication.

Functions:
    - run_compute_eer: Merges ground truth and prediction data, then calculates the EER.

Usage:
    Run this module as a script, providing CSV file paths for public (and optionally private) datasets.
"""

import pandas as pd
import argparse
import json

from metrics import compute_eer


def run_compute_eer(gt_df, sub_df):
    """
    Merges ground truth data and predictions, then computes the EER.

    Parameters:
        gt_df (DataFrame): DataFrame containing ground truth labels. Expects a 'similarity' column (renamed to 'label').
        sub_df (DataFrame): DataFrame containing predictions. Must include 'pair_id' and 'similarity' columns.

    Returns:
        float: The computed Equal Error Rate.
    """

    gt_label_column = "label"
    sub_sim_column = "similarity"
    id_column = "pair_id"

    gt_df = gt_df.rename(columns={"similarity": gt_label_column})
    gt_df = gt_df.join(sub_df.set_index(id_column), on=id_column, how="left")

    if gt_df[sub_sim_column].isna().any():
        print("Not all `pair_id` values are present in the submission file.")

    y_score = sub_df[sub_sim_column].tolist()
    y_true = gt_df[gt_label_column].tolist()

    return compute_eer(y_true, y_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Equal Error Rate (EER) for public and optionally"
            " private datasets."
        )
    )
    parser.add_argument(
        "--public_test_url",
        type=str,
        required=True,
        help="Path to the CSV file with public test data.",
    )
    parser.add_argument(
        "--public_prediction_url",
        type=str,
        required=True,
        help="Path to the CSV file with public predictions.",
    )
    parser.add_argument(
        "--private_test_url",
        type=str,
        required=False,
        help="Path to the CSV file with private test data.",
    )
    parser.add_argument(
        "--private_prediction_url",
        type=str,
        required=False,
        help="Path to the CSV file with private predictions.",
    )
    args = parser.parse_args()

    # Read public data
    public_test_df = pd.read_csv(args.public_test_url)
    public_prediction_df = pd.read_csv(args.public_prediction_url)
    public_score = run_compute_eer(public_test_df, public_prediction_df)

    private_score = None
    if args.private_test_url and args.private_prediction_url:
        private_test_df = pd.read_csv(args.private_test_url)
        private_prediction_df = pd.read_csv(args.private_prediction_url)
        private_score = compute_eer(private_test_df, private_prediction_df)

    # Output the results in JSON format
    print(
        json.dumps(
            {
                "public_score": public_score,
                "private_score": private_score,
            }
        )
    )
