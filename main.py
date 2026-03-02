import os
from src.data.preprocessing import load_data, clean_data, save_cleaned_data

# Get base directory (ML_Project folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Exact dataset file name
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

# Load dataset
df = load_data(RAW_PATH)

# Clean dataset
df_clean = clean_data(df)

# Save cleaned dataset
save_cleaned_data(df_clean, PROCESSED_PATH)

print("Preprocessing completed successfully!")