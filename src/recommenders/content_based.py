import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:

    def __init__(self, data_path, tfidf_matrix_path):

        df = pd.read_csv(data_path)

        # convert to product-level dataset
        self.df = (
            df.sort_values("weighted_rating", ascending=False)
            .drop_duplicates(subset="product_name")
            .reset_index(drop=True)
        )

        # load tfidf matrix
        with open(tfidf_matrix_path, "rb") as f:
            tfidf_matrix = pickle.load(f)

        self.tfidf_matrix = tfidf_matrix[self.df.index]

        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, query, top_n=5):

        query = query.lower()

        matches = self.df[
            self.df["product_name"].str.lower().str.contains(query, na=False)
        ]

        if matches.empty:
            return None

        # best product match
        idx = matches["weighted_rating"].idxmax()

        product_category = self.df.loc[idx, "breadcrumbs"]

        similarity_scores = list(enumerate(self.similarity_matrix[idx]))

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        same_category = []
        other_products = []

        for i, score in similarity_scores[1:]:

            if self.df.loc[i, "breadcrumbs"] == product_category:
                same_category.append(i)
            else:
                other_products.append(i)

        # prioritize same category
        final_indices = same_category[:top_n]

        # fill remaining if needed
        if len(final_indices) < top_n:
            needed = top_n - len(final_indices)
            final_indices.extend(other_products[:needed])

        results = self.df.iloc[final_indices][
            [
                "product_name",
                "brand_name",
                "breadcrumbs",
                "weighted_rating"
            ]
        ]

        return results