import pandas as pd

def get_top_n(df, n=10):
    """
    Returns top n recommendations.
    """
    return df.head(n)

def load_data(path):
    """
    Load processed dataset.
    """
    return pd.read_csv(path)