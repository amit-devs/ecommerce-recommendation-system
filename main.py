from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.knn_model import KNNRecommender
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.item_similarity import ItemSimilarityRecommender
from src.recommenders.svd_model import SVDRecommender

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

DATA_PATH = "data/processed/cleaned_products.csv"


# ---------------------------------------------------
# Evaluation Function
# ---------------------------------------------------

def evaluate_results(results):

    df = pd.read_csv(DATA_PATH)

    if results is None or len(results) == 0:
        print("\nNo recommendations to evaluate.")
        return

    # Ensure DataFrame format
    if isinstance(results, list):
        rec_df = pd.DataFrame({"product_name": results})
    else:
        rec_df = results.copy()

    merged = rec_df.merge(
        df[["product_name", "rating"]],
        on="product_name",
        how="left"
    )

    merged["rating"] = merged["rating"].fillna(0)

    # Define relevance
    merged["actual"] = merged["rating"].apply(lambda x: 1 if x >= 4 else 0)

    # Recommended items are predicted relevant
    merged["predicted"] = 1

    y_true = merged["actual"]
    y_pred = merged["predicted"]

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\nModel Evaluation")
    print("---------------------------")
    print("Accuracy:", round(acc, 4))
    print("\nConfusion Matrix:")
    print(cm)


# ---------------------------------------------------
# Content-Based Model
# ---------------------------------------------------

def run_content_based():

    recommender = ContentBasedRecommender(
        data_path=DATA_PATH,
        tfidf_matrix_path="models/amit/content_based/tfidf_matrix.pkl"
    )

    product = input("Enter product name: ")

    results = recommender.recommend(product, top_n=5)

    if results is None:
        print("Product not found in dataset.")
        return

    print("\nContent-Based Recommendations:\n")

    for _, row in results.iterrows():
        print(
            f"{row['product_name']} | "
            f"Rating: {round(row['weighted_rating'],2)}"
        )

    evaluate_results(results)


# ---------------------------------------------------
# KNN Model
# ---------------------------------------------------

def run_knn():

    recommender = KNNRecommender(data_path=DATA_PATH)

    product = input("Enter product name: ")

    results = recommender.recommend(product, top_n=5)

    if results is None:
        print("Product not found in dataset.")
        return

    print("\nKNN Recommendations:\n")

    for _, row in results.iterrows():
        print(
            f"{row['product_name']} | "
            f"Rating: {round(row['weighted_rating'],2)}"
        )

    evaluate_results(results)


# ---------------------------------------------------
# Hybrid Model
# ---------------------------------------------------

def run_hybrid():

    recommender = HybridRecommender(data_path=DATA_PATH)

    product = input("Enter product name: ")

    results = recommender.recommend(product, top_n=5)

    if results is None:
        print("Product not found in dataset.")
        return

    print("\nHybrid Recommendations:\n")

    for _, row in results.iterrows():

        if "hybrid_score" in row:
            print(
                f"{row['product_name']} | "
                f"Rating: {round(row['weighted_rating'],2)} | "
                f"Hybrid Score: {round(row['hybrid_score'],2)}"
            )
        else:
            print(
                f"{row['product_name']} | "
                f"Rating: {round(row['weighted_rating'],2)}"
            )

    evaluate_results(results)


# ---------------------------------------------------
# Popularity Model
# ---------------------------------------------------

def run_popularity():

    recommender = PopularityRecommender(data_path=DATA_PATH)

    results = recommender.get_top_products(top_n=5)

    print("\nTop Popular Products:\n")

    for _, row in results.iterrows():
        print(
            f"{row['product_name']} | "
            f"{row['brand_name']} | "
            f"{row['breadcrumbs']} | "
            f"Rating: {round(row['weighted_rating'],2)}"
        )

    evaluate_results(results)


# ---------------------------------------------------
# Item Similarity Model
# ---------------------------------------------------

def run_item_similarity():

    recommender = ItemSimilarityRecommender(data_path=DATA_PATH)

    product = input("Enter product name: ")

    results = recommender.recommend(product, top_n=5)

    if results is None or len(results) == 0:
        print("Product not found in dataset.")
        return

    print("\nItem Similarity Recommendations:\n")

    for _, row in results.iterrows():
        print(
            f"{row['product_name']} | "
            f"Rating: {round(row['weighted_rating'],2)}"
        )

    evaluate_results(results)


# ---------------------------------------------------
# SVD Model
# ---------------------------------------------------

def run_svd():

    recommender = SVDRecommender(data_path=DATA_PATH)

    product = input("Enter product name: ")

    results = recommender.recommend(product, top_n=5)

    if results is None or len(results) == 0:
        print("Product not found in dataset.")
        return

    print("\nSVD Recommendations:\n")

    for _, row in results.iterrows():
        print(
            f"{row['product_name']} | "
            f"{row['brand_name']} | "
            f"{row['breadcrumbs']} | "
            f"Rating: {round(row['weighted_rating'],2)}"
        )

    evaluate_results(results)


# ---------------------------------------------------
# CLI Menu
# ---------------------------------------------------

def main():

    print("\nChoose Recommendation Model")
    print("1 → Content Based Recommender")
    print("2 → KNN Collaborative Filtering")
    print("3 → Hybrid Recommender")
    print("4 → Popularity Recommender")
    print("5 → Item Similarity Recommender")
    print("6 → SVD Matrix Factorization")

    choice = input("Enter choice: ")

    if choice == "1":
        run_content_based()

    elif choice == "2":
        run_knn()

    elif choice == "3":
        run_hybrid()

    elif choice == "4":
        run_popularity()

    elif choice == "5":
        run_item_similarity()

    elif choice == "6":
        run_svd()

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()