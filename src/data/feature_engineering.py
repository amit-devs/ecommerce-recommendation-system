import pandas as pd


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
    Useful for popularity and hybrid model.
    """

    # Global average rating
    C = df["Product Rating"].mean()

    # Minimum reviews required (threshold)
    m = df["Product Reviews Count"].quantile(0.75)

    def weighted_score(row):
        v = row["Product Reviews Count"]
        R = row["Product Rating"]

        return (v / (v + m)) * R + (m / (v + m)) * C

    df["weighted_rating"] = df.apply(weighted_score, axis=1)

    return df