import pandas as pd
from sklearn.neighbors import NearestNeighbors


class KNNRecommender:

    def __init__(self, data_path="data/processed/cleaned_products.csv"):

        # Load dataset
        self.df = pd.read_csv(data_path)

        # Create user-item matrix
        self.user_item_matrix = self.df.pivot_table(
            index="product_id",
            columns="user_id",
            values="rating",
            fill_value=0
        )

        # Create product mappings
        product_info = self.df[["product_id", "product_name"]].drop_duplicates()

        self.product_id_to_name = dict(
            zip(product_info.product_id, product_info.product_name)
        )

        self.product_name_to_id = dict(
            zip(product_info.product_name, product_info.product_id)
        )

        # Initialize KNN model
        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute"
        )

        # Train model
        self.model.fit(self.user_item_matrix)

        print("KNN Model trained successfully")

    # ---------------------------------------------------
    # Recommendation Function
    # ---------------------------------------------------

    def recommend(self, product_name, top_n=5):

        # Check if product exists
        if product_name not in self.product_name_to_id:
            print("Product not found in dataset")
            return None

        product_id = self.product_name_to_id[product_name]

        if product_id not in self.user_item_matrix.index:
            print("Product not present in interaction matrix")
            return None

        # Get product vector
        product_vector = self.user_item_matrix.loc[[product_id]]

        # Find nearest neighbors
        distances, indices = self.model.kneighbors(product_vector, n_neighbors=top_n + 1)

        recommendations = []

        for idx in indices.flatten():

            similar_product_id = self.user_item_matrix.index[idx]

            if similar_product_id == product_id:
                continue

            product_name = self.product_id_to_name.get(similar_product_id)

            if product_name not in recommendations:
                recommendations.append(product_name)

        return recommendations[:top_n]