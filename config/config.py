reduce_factor = 0.2
top_categories = 5
test_size = 0.2
# all if you want to select all
percent_to_augment = 0.5
new_sent_per_sent = 1
num_words_replace = 3

# tfidf, word2vec, fastttext, BERT, sentencetransformer
vectorizer_type = "tfidf"

test_file_path = (
    "data/test_data_rf_" + str(reduce_factor) + "_ts_" + str(test_size) + ".csv"
)

train_file_path = (
    "data/train_data_rf_" + str(reduce_factor) + "_ts_" + str(test_size) + ".csv"
)
