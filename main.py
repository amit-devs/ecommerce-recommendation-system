from src.recommenders.content_based import ContentBasedRecommender


def main():

    recommender = ContentBasedRecommender(
        data_path="data/processed/cleaned_products.csv",
        tfidf_matrix_path="models/amit/content_based/tfidf_matrix.pkl"
    )

    product = input("Enter product name: ")

    results = recommender.recommend(product, top_n=5)

    if results is not None:
        print("\nTop Recommendations:\n")
        print(results)


if __name__ == "__main__":
    main()