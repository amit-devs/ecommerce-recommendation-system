import pandas as pd


class PopularityRecommender:

    def __init__(self, data_path):
        # Load dataset
        self.data = pd.read_csv(data_path)

        # Remove duplicate products
        self.data = self.data.drop_duplicates(subset=["product_name"])

    def get_top_products(self, top_n=5):
        """
        Returns top N products based on weighted rating
        """

        # Sort products by weighted rating
        df = self.data.sort_values(
            by="weighted_rating",
            ascending=False
        )

        results = df[[
            "product_name",
            "brand_name",
            "breadcrumbs",
            "weighted_rating"
        ]].head(top_n)

        return results

    def recommend(self, top_n=5):
        """
        Wrapper method so main.py can call recommend()
        """
        return self.get_top_products(top_n)