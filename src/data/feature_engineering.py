import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os


def create_combined_text_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine important text columns into one single feature
    for content-based filtering.
    """

    df["combined_text"] = (
        df["Product Name"].fillna("") + " " +
        df["Product Brand"].fillna("") + " " +
        df["Product Category"].fillna("") + " " +
        df["Product Description"].fillna("")
    )

    return df


def create_weighted_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weighted rating score using rating and review count.
    """

    C = df["Product Rating"].mean()
    m = df["Product Reviews Count"].quantile(0.75)

    def weighted_score(row):
        v = row["Product Reviews Count"]
        R = row["Product Rating"]
        return (v / (v + m)) * R + (m / (v + m)) * C

    df["weighted_rating"] = df.apply(weighted_score, axis=1)

    return df


def create_tfidf_matrix(df: pd.DataFrame, model_save_path: str):
    """
    Create TF-IDF matrix from combined_text column
    and save vectorizer.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    # Save vectorizer
    with open(model_save_path, "wb") as f:
        pickle.dump(vectorizer, f)

    return tfidf_matrix