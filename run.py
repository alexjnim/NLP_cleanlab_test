import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from pipeline.get_data import get_data
from pipeline.vectorize_text import vectorize_text
from cleanlab.pruning import get_noise_indices
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from config import config

#######################
#      load data
#######################

if Path(config.train_file_path).is_file():
    print("loading data...")
    train_df = pd.read_csv(config.train_file_path)
    test_df = pd.read_csv(config.test_file_path)
else:
    print("training data does not exist\ngenerating data now")
    df = get_data(
        reduce_factor=config.reduce_factor, top_categories=config.top_categories
    )
    train_df, test_df = train_test_split(
        df, test_size=config.test_size, random_state=42
    )
    train_df.to_csv(config.train_file_path, index=False)
    test_df.to_csv(config.test_file_path, index=False)

#######################
#    vectorise text
#######################
# tfidf, word2vec, fasttext, BERT, sentencetransformer
(
    train_vectors,
    test_vectors,
    train_label_names,
    test_label_names,
    tv,
) = vectorize_text(train_df, test_df, type=config.vectorizer_type)

# change categorical labels to numerical
le = LabelEncoder()
le.fit(train_label_names)
le_test_label_names = le.transform(test_label_names)
le_train_label_names = le.transform(train_label_names)

#######################
#    build model
#######################

logistic_clf = LogisticRegression(
    n_jobs=-1, random_state=2020, C=2, penalty="predicted_classes", max_iter=1000
)

logistic_clf.fit(train_vectors, le_train_label_names)

y_pred = logistic_clf.predict(test_vectors)
print("Accuracy: {:.1%}".format(accuracy_score(le_test_label_names, y_pred)))
print("F1: {:.1%}".format(f1_score(le_test_label_names, y_pred, average="micro")))

#######################
#    prune data
#######################

# get class prediction probabilities
y_pred_proba = logistic_clf.predict_proba(test_vectors)

incorrect_labels = get_noise_indices(
    s=le_test_label_names,
    psx=y_pred_proba,
    sorted_index_method="normalized_margin",  # Orders label errors
)

print("We found {} label errors.".format(len(incorrect_labels)))

#######################
# predict correct labels
#######################

# build model wrapped over cleanlab
cleanlab_model = LearningWithNoisyLabels(clf=logistic_clf)
cleanlab_model.fit(X=train_vectors, s=le_train_label_names)

# set up incorrect data for prediction
text_to_correct = test_df["text"].iloc[incorrect_labels.tolist()]
vectors_to_correct = tv.transform(text_to_correct)

# Estimate the predictions you would have gotten by training with *no* label errors.
predicted_correct_labels = cleanlab_model.predict(vectors_to_correct)
predicted_classes = le.inverse_transform(predicted_correct_labels.tolist())

# build dataframe to see results
cl_df = test_df[["category", "original_text"]].iloc[incorrect_labels.tolist()]
cl_df["prediciton"] = predicted_classes

print(cl_df.to_string())
