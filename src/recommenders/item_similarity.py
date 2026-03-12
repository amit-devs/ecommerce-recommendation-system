import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemSimilarityRecommender:

    def __init__(self, data_path="data/processed/cleaned_products.csv"):

        # Load dataset
        self.data = pd.read_csv(data_path)

        # Remove duplicate products
        self.data = self.data.drop_duplicates(subset=["product_id"])

        # Load ratings dataset
        ratings = pd.read_csv("data/raw/reviews.csv")

        ratings = ratings.rename(columns={
            "reviewID": "user_id",
            "productASIN": "product_id",
            "rating": "rating"
        })

        # Create user‑item matrix
        self.user_item_matrix = ratings.pivot_table(
            index="product_id",
            columns="user_id",
            values="rating"
        ).fillna(0)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(self.user_item_matrix)

        self.similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

    # ---------------------------------------------------
    # Product Search
    # ---------------------------------------------------

    def search_product(self, keyword):

        keyword = keyword.lower()

        matches = self.data[
            self.data["product_name"].str.lower().str.contains(keyword, na=False)
        ]

        if matches.empty:
            return None, None

        product_row = matches.iloc[0]

        return product_row["product_id"], product_row["breadcrumbs"]

    # ---------------------------------------------------
    # Recommendation Function
    # ---------------------------------------------------

    def recommend(self, keyword, top_n=5):

        product_id, category = self.search_product(keyword)

        if product_id is None:
            print("No matching product found.")
            return None

        if product_id not in self.similarity_df.index:
            print("Product not found in similarity matrix.")
            return None

        # Get similarity scores
        similarities = self.similarity_df[product_id].sort_values(ascending=False)

        # Remove the same product
        similar_products = similarities.iloc[1:100].index

        recommendations = self.data[
            self.data["product_id"].isin(similar_products)
        ]

        # ---------------------------------------------------
        # CATEGORY FILTERING (IMPORTANT FIX)
        # ---------------------------------------------------

        recommendations = recommendations[
            recommendations["breadcrumbs"] == category
        ]

        # Remove duplicates
        recommendations = recommendations.drop_duplicates(subset=["product_name"])

        # Sort by rating
        recommendations = recommendations.sort_values(
            by="weighted_rating",
            ascending=False
        )

        results = recommendations.head(top_n)[
            ["product_name", "brand_name", "breadcrumbs", "weighted_rating"]
        ]

        return results