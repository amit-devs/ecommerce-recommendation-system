import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:

    def __init__(self, data_path, tfidf_matrix_path):

        self.df = pd.read_csv(data_path)

        with open(tfidf_matrix_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, query, top_n=5):

        query = query.lower()

        matches = self.df[self.df["product_name"].str.lower().str.contains(query)]

        if matches.empty:
            print("Product not found in dataset.")
            return None

        # choose best match
        idx = matches["weighted_rating"].idxmax()

        product_category = self.df.loc[idx, "breadcrumbs"]

        similarity_scores = list(enumerate(self.similarity_matrix[idx]))

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        recommendations = []

        for i, score in similarity_scores[1:]:

            # keep only same category
            if self.df.loc[i, "breadcrumbs"] != product_category:
                continue

            if score < 0.15:
                continue

            recommendations.append((i, score))

            if len(recommendations) >= top_n:
                break

        product_indices = [i[0] for i in recommendations]

        results = self.df.iloc[product_indices][
            [
                "product_name",
                "brand_name",
                "breadcrumbs",
                "weighted_rating"
            ]
        ]

        return results