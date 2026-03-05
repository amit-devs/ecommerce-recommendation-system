from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.knn_model import KNNRecommender


def run_content_based():

    recommender = ContentBasedRecommender(
        data_path="data/processed/cleaned_products.csv",
        tfidf_matrix_path="models/amit/content_based/tfidf_matrix.pkl"
    )

    product = input("Enter product name: ")

    results = recommender.recommend(product, top_n=5)

    if results is not None:
        print("\nContent-Based Recommendations:\n")
        print(results)


def run_knn():

    recommender = KNNRecommender(
        data_path="data/processed/cleaned_products.csv"
    )

    product = input("Enter product name: ")

    results = recommender.recommend(product, top_n=5)

    if results is not None:
        print("\nKNN Recommendations:\n")

        for i, item in enumerate(results, 1):
            print(f"{i}. {item}")


def main():

    print("\nChoose Recommendation Model")
    print("1 → Content Based Recommender")
    print("2 → KNN Collaborative Filtering")

    choice = input("Enter choice: ")

    if choice == "1":
        run_content_based()

    elif choice == "2":
        run_knn()

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()