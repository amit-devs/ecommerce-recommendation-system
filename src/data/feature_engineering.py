import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os


def create_combined_text_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combined text feature for recommendation.
    Uses product name, brand, description, and category.
    """

    df["combined_text"] = (
        df["product_name"].fillna("") + " " +
        df["brand_name"].fillna("") + " " +
        df["about_item"].fillna("") + " " +
        df["product_description"].fillna("") + " " +
        df["breadcrumbs"].fillna("")
    )

    df["combined_text"] = df["combined_text"].str.lower()

    return df


def create_weighted_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weighted rating score using product rating.
    """

    C = df["rating"].mean()
    m = df["rating"].quantile(0.75)

    def weighted_score(row):
        R = row["rating"]

        return (R + C) / 2

    df["weighted_rating"] = df.apply(weighted_score, axis=1)

    return df


def create_tfidf_matrix(df: pd.DataFrame, model_save_path: str):
    """
    Create TF-IDF matrix from combined text.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=4000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    with open(model_save_path, "wb") as f:
        pickle.dump(vectorizer, f)

    matrix_path = os.path.join(
        os.path.dirname(model_save_path),
        "tfidf_matrix.pkl"
    )

    with open(matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    return tfidf_matrix