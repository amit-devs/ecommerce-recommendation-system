import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class SVDRecommender:

    def __init__(self, data_path):

        # Load dataset
        self.data = pd.read_csv(data_path)

        # Create USER-ITEM matrix
        user_item = self.data.pivot_table(
            index="user_id",
            columns="product_name",
            values="rating",
            fill_value=0
        )

        # Transpose → product matrix
        self.product_matrix = user_item.T

        self.product_names = list(self.product_matrix.index)

        # Apply SVD
        svd = TruncatedSVD(n_components=50, random_state=42)

        self.product_features = svd.fit_transform(self.product_matrix)

        # Compute similarity
        self.similarity_matrix = cosine_similarity(self.product_features)

        # Product metadata
        self.products = (
            self.data
            .sort_values("weighted_rating", ascending=False)
            .drop_duplicates(subset="product_name")
            .reset_index(drop=True)
        )

        self.product_info = self.products.set_index("product_name")

    def recommend(self, query, top_n=5):

        query = query.lower()

        # Find product
        matches = self.products[
            self.products["product_name"]
            .str.lower()
            .str.contains(query, na=False)
        ]

        if matches.empty:
            return None

        product = matches.iloc[0]["product_name"]

        if product not in self.product_names:
            return None

        idx = self.product_names.index(product)

        # Similarity scores
        scores = list(enumerate(self.similarity_matrix[idx]))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Get main category keyword
        category = self.product_info.loc[product]["breadcrumbs"].lower()
        category_keyword = category.split("›")[-1].strip()

        recommendations = []

        # First pass → same category
        for i, score in scores[1:200]:

            rec_product = self.product_names[i]

            if rec_product not in self.product_info.index:
                continue

            rec_category = self.product_info.loc[rec_product]["breadcrumbs"].lower()

            if category_keyword in rec_category:
                recommendations.append((rec_product, score))

            if len(recommendations) >= top_n:
                break

        # Second pass → fill remaining slots if needed
        if len(recommendations) < top_n:

            for i, score in scores[1:]:

                rec_product = self.product_names[i]

                if rec_product == product:
                    continue

                if rec_product not in [r[0] for r in recommendations]:

                    recommendations.append((rec_product, score))

                if len(recommendations) >= top_n:
                    break

        rec_names = [r[0] for r in recommendations]

        results = self.products[
            self.products["product_name"].isin(rec_names)
        ][[
            "product_name",
            "brand_name",
            "breadcrumbs",
            "weighted_rating"
        ]]

        return results.head(top_n)