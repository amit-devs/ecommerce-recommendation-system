import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from given file path.
    """
    df = pd.read_csv(filepath, sep="\t")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning:
    - Remove duplicates
    - Handle missing values
    - Convert rating to numeric
    - Standardize text columns
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Drop rows missing critical columns
    df = df.dropna(subset=[
        "Product Name",
        "Product Description",
        "Product Rating"
    ])

    # Convert rating to numeric
    df["Product Rating"] = pd.to_numeric(
        df["Product Rating"],
        errors="coerce"
    )

    # Fill missing review counts with 0
    if "Product Reviews Count" in df.columns:
        df["Product Reviews Count"] = df["Product Reviews Count"].fillna(0)

    # Standardize important text columns
    text_columns = [
        "Product Name",
        "Product Brand",
        "Product Category",
        "Product Description"
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """
    Save cleaned dataset to CSV.
    """
    df.to_csv(output_path, index=False)