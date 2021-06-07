from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def get_TFIDF_embeddings(train_df, test_df):
    train_corpus, test_corpus, train_label_names, test_label_names = (
        train_df["text"],
        test_df["text"],
        train_df["category"],
        test_df["category"],
    )

    print("getting tfidf vectors")
    tv = TfidfVectorizer(
        min_df=0.0, max_df=1.0, norm="l2", use_idf=True, smooth_idf=True
    )  # ngram_range = (1,2)

    tv_train_features = tv.fit_transform(train_corpus)
    tv_test_features = tv.transform(test_corpus)

    return tv_train_features, tv_test_features, train_label_names, test_label_names, tv
