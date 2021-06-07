import mlflow
from config import config


def log_config_params(aug_logging=True):
    mlflow.log_param("reduc_factor", config.reduce_factor)
    mlflow.log_param("top_categories", config.top_categories)
    mlflow.log_param("vectorizer_type", config.vectorizer_type)
    if aug_logging:
        mlflow.log_param("percent_to_augment", config.percent_to_augment)
        mlflow.log_param("new_sent_per_sent", config.new_sent_per_sent)
        mlflow.log_param("num_words_replace", config.num_words_replace)
