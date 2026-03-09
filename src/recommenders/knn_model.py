import pandas as pd
from sklearn.neighbors import NearestNeighbors


class KNNRecommender:

    def __init__(self, data_path="data/processed/cleaned_products.csv"):

        df = pd.read_csv(data_path)

        # Product-level dataset
        self.df = (
            df.sort_values("weighted_rating", ascending=False)
              .drop_duplicates(subset="product_id")
              .reset_index(drop=True)
        )

        # User-item matrix (for collaborative filtering)
        self.user_item_matrix = df.pivot_table(
            index="product_id",
            columns="user_id",
            values="rating",
            fill_value=0
        )

        # Product information
        product_info = self.df[
            ["product_id", "product_name", "brand_name", "breadcrumbs", "weighted_rating"]
        ]

        # Mappings
        self.product_id_to_name = dict(
            zip(product_info.product_id, product_info.product_name)
        )

        self.product_name_to_id = dict(
            zip(product_info.product_name, product_info.product_id)
        )

        # KNN model
        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute"
        )

        # Train model
        self.model.fit(self.user_item_matrix)

    # ---------------------------------------------------
    # Recommendation Function
    # ---------------------------------------------------

    def recommend(self, query, top_n=5):

        query = query.lower()

        # Keyword search
        matches = self.df[
            self.df["product_name"].str.lower().str.contains(query, na=False)
        ]

        if matches.empty:
            return None

        # Select best match
        product_row = matches.iloc[0]
        product_id = product_row["product_id"]
        product_category = product_row["breadcrumbs"]

        if product_id not in self.user_item_matrix.index:
            return None

        product_vector = self.user_item_matrix.loc[[product_id]]

        distances, indices = self.model.kneighbors(
            product_vector,
            n_neighbors=top_n * 10
        )

        same_category = []
        other_products = []

        for idx in indices.flatten():

            similar_product_id = self.user_item_matrix.index[idx]

            if similar_product_id == product_id:
                continue

            product_data = self.df[self.df["product_id"] == similar_product_id]

            if product_data.empty:
                continue

            product_info = product_data.iloc[0]

            if product_info["breadcrumbs"] == product_category:
                same_category.append(product_info)
            else:
                other_products.append(product_info)

        # Prioritize same category
        results = same_category[:top_n]

        if len(results) < top_n:
            remaining = top_n - len(results)
            results.extend(other_products[:remaining])

        results_df = pd.DataFrame(results)[
            [
                "product_name",
                "brand_name",
                "breadcrumbs",
                "weighted_rating"
            ]
        ]

        results_df = results_df.drop_duplicates(subset="product_name")

        return results_df.head(top_n)