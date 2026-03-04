import pandas as pd


def load_data(products_path: str, reviews_path: str):
    """
    Load products and reviews datasets.
    """

    products = pd.read_csv(products_path)
    reviews = pd.read_csv(reviews_path)

    return products, reviews


def merge_data(products: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Merge products and reviews using product ASIN.
    """

    df = pd.merge(
        reviews,
        products,
        left_on="productASIN",
        right_on="asin",
        how="inner"
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean merged dataset.
    """

    # Keep only important columns
    df = df[[
        "reviewID",
        "productASIN",
        "rating",
        "title",
        "brand_name",
        "about_item",
        "product_description",
        "breadcrumbs"
    ]]

    # Remove missing values
    df = df.dropna()

    # Rename columns to standard format
    df = df.rename(columns={
        "reviewID": "user_id",
        "productASIN": "product_id",
        "title": "product_name"
    })

    # Convert rating to numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Drop invalid ratings
    df = df.dropna(subset=["rating"])

    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """
    Save cleaned dataset to processed folder.
    """

    df.to_csv(output_path, index=False)