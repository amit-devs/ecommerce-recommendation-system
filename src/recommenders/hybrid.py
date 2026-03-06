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

    def recommend(self, product_name, top_n=5):

        # Get recommendations from both models
        content_recs = self.content_model.recommend(product_name, top_n=top_n)
        knn_recs = self.knn_model.recommend(product_name, top_n=top_n)

        # If both models return nothing
        if content_recs is None and knn_recs is None:
            return None

        # Convert to DataFrame safely
        content_df = pd.DataFrame(content_recs) if content_recs is not None else pd.DataFrame()
        knn_df = pd.DataFrame(knn_recs) if knn_recs is not None else pd.DataFrame()

        # Add source labels
        if not content_df.empty:
            content_df["source"] = "content"

        if not knn_df.empty:
            knn_df["source"] = "knn"

        # Merge results
        hybrid_df = pd.concat([content_df, knn_df], ignore_index=True)

        # Remove rows without product name
        if "product_name" in hybrid_df.columns:
            hybrid_df = hybrid_df.dropna(subset=["product_name"])

        # Remove duplicates
        hybrid_df = hybrid_df.drop_duplicates(subset="product_name")

        # Hybrid scoring
        hybrid_df["hybrid_score"] = hybrid_df["source"].map({
            "content": 0.6,
            "knn": 0.4
        })

        # Sorting
        if "weighted_rating" in hybrid_df.columns:
            hybrid_df = hybrid_df.sort_values(
                by=["hybrid_score", "weighted_rating"],
                ascending=False
            )
        else:
            hybrid_df = hybrid_df.sort_values(
                by="hybrid_score",
                ascending=False
            )

        hybrid_df = hybrid_df.head(top_n)
        hybrid_df = hybrid_df.reset_index(drop=True)

        return hybrid_df