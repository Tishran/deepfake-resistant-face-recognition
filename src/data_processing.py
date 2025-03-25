"""
data_processing.py
-------------

Processes training data to adjust labels for fake samples and filter invalid classes.

Functions:
    - fix_labeling_dataframe: Adjusts labels for fake samples and removes singleton classes.
"""

import json
import os
import pandas as pd

TRAIN_IMAGES_PATH = "data/train/images"


def fix_labeling_dataframe(dataframe, fake_indicators):
    """
    Adjusts labels for fake samples and removes classes with only one instance.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame with columns ["path", "label"].
        fake_indicators (dict): Dictionary mapping image paths (relative to TRAIN_IMAGES_PATH)
                                to boolean values indicating if they are fake (e.g., {"image.jpg": True}).

    Returns:
        pd.DataFrame: Processed DataFrame with:
                      - Fake samples assigned new labels (original_label + max_label + 1)
                      - Singleton classes (only one sample) removed
    """

    def fix_df_func(df, max_label):
        df.loc[df["path"].isin(fakes_set), "label"] = df[
            df["path"].isin(fakes_set)
        ].apply(lambda x: x.label + max_label + 1, axis=1)

    fakes_set = set()
    for path, is_fake in fake_indicators.items():
        if is_fake:
            fakes_set.add(os.path.join(TRAIN_IMAGES_PATH, path))

    max_label = max(list(dataframe.label))
    fix_df_func(dataframe, max_label)

    # getting rid of classes wich contains only one element
    dataframe = dataframe[dataframe["label"].duplicated(keep=False)]
    return dataframe


if __name__ == "__main__":
    with open(f"{TRAIN_IMAGES_PATH}/../meta.json", "r") as f:
        fake_indicators = json.load(f)

    train_df = pd.read_csv("train.csv")
    train_df = fix_labeling_dataframe(train_df, fake_indicators)
    train_df.to_csv("reorganized_train.csv", index=False)
