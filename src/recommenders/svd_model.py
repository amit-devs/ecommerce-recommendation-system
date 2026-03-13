import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class SVDRecommender:

    def __init__(self, data_path):

        # Load dataset
        self.data = pd.read_csv(data_path)

        # Create user-item matrix
        user_item = self.data.pivot_table(
            index="product_name",
            columns="user_id",
            values="rating",
            fill_value=0
        )

        self.product_names = user_item.index

        # Apply SVD
        svd = TruncatedSVD(n_components=50)

        self.matrix_reduced = svd.fit_transform(user_item)

        # Compute similarity
        self.similarity_matrix = cosine_similarity(self.matrix_reduced)

        # Metadata
        self.products = (
            self.data.sort_values("weighted_rating", ascending=False)
            .drop_duplicates(subset="product_name")
            .reset_index(drop=True)
        )

    def recommend(self, query, top_n=5):

        query = query.lower()

        matches = self.products[
            self.products["product_name"]
            .str.lower()
            .str.contains(query, na=False)
        ]

        if matches.empty:
            return None

        product = matches.iloc[0]["product_name"]

        idx = list(self.product_names).index(product)

        scores = list(enumerate(self.similarity_matrix[idx]))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        recommendations = []

        for i, score in scores[1:top_n+1]:
            recommendations.append(self.product_names[i])

        results = self.products[
            self.products["product_name"].isin(recommendations)
        ][[
            "product_name",
            "brand_name",
            "breadcrumbs",
            "weighted_rating"
        ]]

        return results.head(top_n)