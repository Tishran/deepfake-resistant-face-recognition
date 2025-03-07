import json
import os
import pandas as pd
import shutil

TRAIN_IMAGES_PATH = "data/train/images"


def fix_labeling_dataframe(dataframe, fake_indicators):
    def gen_new_path(path):
        splitted_path = path.split("/")
        splitted_path[-2] = f"{splitted_path[-2]}_fake"
        return "/".join(splitted_path)

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

    train_df = pd.read_csv("val.csv")
    train_df = fix_labeling_dataframe(train_df, fake_indicators)
    train_df.to_csv("reorganized_val.csv", index=False)
