import os
from src.data.preprocessing import load_data, clean_data, save_cleaned_data
from src.data.feature_engineering import create_combined_text_feature, create_weighted_rating

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv"
)

PROCESSED_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_products.csv"
)

# Load
df = load_data(RAW_PATH)

# Clean
df_clean = clean_data(df)

# Feature Engineering
df_features = create_combined_text_feature(df_clean)
df_features = create_weighted_rating(df_features)

# Save
save_cleaned_data(df_features, PROCESSED_PATH)

print("Preprocessing + Feature Engineering completed successfully!")