import pandas as pd
import mlflow
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from utils.log_config_params import log_config_params


def train_multiple_models(
    train_corpus, test_corpus, train_label_names, test_label_names, aug_logging=False
):
    lr = LogisticRegression(penalty="l2", max_iter=1000, C=1, random_state=42)
    svm = LinearSVC(penalty="l2", C=1, random_state=42)
    svm_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, random_state=42)
    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)

    scores_df = pd.DataFrame(
        columns=["train_accuracy", "val_accuracy", "test_accuracy", "average_fit_time"]
    )

    models = [lr, svm, svm_sgd, rfc, gbc]

    names = [
        "Logistic Regression",
        "Linear SVC",
        "SGD Classifier",
        "Random Forest Classifier",
        "Gradient Boosting Classifier",
    ]

    for model, name in zip(models, names):
        with mlflow.start_run():
            print("now training {}".format(model))
            temp_list = []
            model.fit(train_corpus, train_label_names)
            scores = cross_validate(
                model,
                train_corpus,
                train_label_names,
                scoring=("accuracy"),
                return_train_score=True,
                cv=5,
            )

            mean_train_score = scores["train_score"].mean()
            mean_val_score = scores["test_score"].mean()
            temp_list.append(mean_train_score)
            temp_list.append(mean_val_score)

            test_score = model.score(test_corpus, test_label_names)
            temp_list.append(test_score)
            temp_list.append(scores["fit_time"].mean())

            scores_df.loc[name] = temp_list

            # logging
            log_config_params(aug_logging=aug_logging)
            mlflow.log_param("model_name", name)
            mlflow.log_param("num_train_sentences", train_corpus.shape[0])
            mlflow.log_param("num_test_sentences", test_corpus.shape[0])
            mlflow.log_metric("train_accuracy", mean_train_score)
            mlflow.log_metric("val_accuracy", mean_val_score)
            mlflow.log_metric("test_accuracy", test_score)
            mlflow.log_metric("average_fit_time", scores["fit_time"].mean())

        mlflow.end_run()
    return scores_df
