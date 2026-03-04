from src.data.preprocessing import (
    load_data,
    merge_data,
    clean_data,
    save_cleaned_data
)

from src.data.feature_engineering import (
    create_combined_text_feature,
    create_weighted_rating
)


def run_pipeline():

    print("Loading datasets...")

    products, reviews = load_data(
        "data/raw/products.csv",
        "data/raw/reviews.csv"
    )

    print("Merging datasets...")
    df = merge_data(products, reviews)

    print("Cleaning dataset...")
    df = clean_data(df)

    print("Creating text features...")
    df = create_combined_text_feature(df)

    print("Creating weighted ratings...")
    df = create_weighted_rating(df)

    print("Saving processed dataset...")
    save_cleaned_data(df, "data/processed/cleaned_products.csv")

    print("Preprocessing + Feature Engineering completed successfully!")


if __name__ == "__main__":
    run_pipeline()