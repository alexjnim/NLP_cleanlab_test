import pandas as pd
from sklearn.model_selection import train_test_split
from utils.text_normalizer import normalize_corpus


def get_data(reduce_factor=None, top_categories=None):
    df = pd.read_json("data/News_Category_Dataset_v2.json", lines=True)

    df["original_text"] = df["headline"] + " " + df["short_description"]
    if top_categories:
        print("\nsize of data: {}".format(len(df)))
        print("extracting top {} categories only".format(top_categories))
        top_cat = df["category"].value_counts().index.tolist()
        df = df[df["category"].isin(top_cat[:5])]
        print("reduced size of data: {}\n".format(len(df)))
        print(df["category"].value_counts())
    if reduce_factor:
        print("\nextracting {} of the data".format(reduce_factor))
        print("original size of data: {}".format(len(df)))
        _, df = train_test_split(df, test_size=reduce_factor, random_state=42)
        print("reduced size of data: {}\n".format(len(df)))

    df["text"] = normalize_corpus(
        corpus=df["original_text"],
        html_stripping=True,
        contraction_expansion=True,
        accented_char_removal=True,
        text_lemmatization=False,
        text_stemming=False,
        special_char_removal=True,
        remove_digits=False,
        stopword_removal=False,
        lower_case=True,
    )

    return df[["category", "text", "original_text"]]
