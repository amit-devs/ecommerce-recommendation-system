import pandas as pd

from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.knn_model import KNNRecommender


class HybridRecommender:

    def __init__(self,
                 data_path="data/processed/cleaned_products.csv",
                 tfidf_matrix_path="models/amit/content_based/tfidf_matrix.pkl"):

        self.data = pd.read_csv(data_path)

        self.content_model = ContentBasedRecommender(
            data_path=data_path,
            tfidf_matrix_path=tfidf_matrix_path
        )

        self.knn_model = KNNRecommender(
            data_path=data_path
        )

    # ---------------------------------------------------
    # Hybrid Recommendation
    # ---------------------------------------------------

    def recommend(self, query, top_n=5):

        # Get recommendations from both models
        content_recs = self.content_model.recommend(query, top_n=top_n)
        knn_recs = self.knn_model.recommend(query, top_n=top_n)

        if content_recs is None and knn_recs is None:
            return None

        content_df = pd.DataFrame(content_recs) if content_recs is not None else pd.DataFrame()
        knn_df = pd.DataFrame(knn_recs) if knn_recs is not None else pd.DataFrame()

        # Add model source
        if not content_df.empty:
            content_df["source"] = "content"

        if not knn_df.empty:
            knn_df["source"] = "knn"

        # Merge results
        hybrid_df = pd.concat([content_df, knn_df], ignore_index=True)

        if hybrid_df.empty:
            return None

        hybrid_df = hybrid_df.dropna(subset=["product_name"])

        # Count how many models recommended the product
        model_count = hybrid_df.groupby("product_name")["source"].count()

        hybrid_df = hybrid_df.drop_duplicates(subset="product_name")

        hybrid_df["model_votes"] = hybrid_df["product_name"].map(model_count)

        # Hybrid scoring
        hybrid_df["hybrid_score"] = (
            hybrid_df["model_votes"] * 0.5 +
            hybrid_df["weighted_rating"] * 0.5
        )

        # Sort by hybrid score
        hybrid_df = hybrid_df.sort_values(
            by="hybrid_score",
            ascending=False
        )

        hybrid_df = hybrid_df.head(top_n)

        hybrid_df = hybrid_df.reset_index(drop=True)

        return hybrid_df[
            [
                "product_name",
                "brand_name",
                "breadcrumbs",
                "weighted_rating",
                "hybrid_score"
            ]
        ]