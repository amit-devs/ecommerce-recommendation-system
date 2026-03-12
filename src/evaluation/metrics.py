import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

DATA_PATH = "data/processed/cleaned_products.csv"


def precision_at_k(actual, predicted, k=5):

    actual_k = actual[:k]
    predicted_k = predicted[:k]

    relevant_items = sum((actual_k == 1) & (predicted_k == 1))

    return relevant_items / k


def recall_at_k(actual, predicted, k=5):

    actual_k = actual[:k]
    predicted_k = predicted[:k]

    relevant_items = sum((actual_k == 1) & (predicted_k == 1))

    total_relevant = sum(actual == 1)

    if total_relevant == 0:
        return 0

    return relevant_items / total_relevant


def evaluate_results(results, k=5):

    df = pd.read_csv(DATA_PATH)

    if results is None or len(results) == 0:
        print("\nNo recommendations to evaluate.")
        return

    # Convert results to DataFrame
    if isinstance(results, list):
        rec_df = pd.DataFrame({"product_name": results})
    else:
        rec_df = results.copy()

    # Merge ratings
    merged = rec_df.merge(
        df[["product_name", "rating"]],
        on="product_name",
        how="left"
    )

    merged["rating"] = merged["rating"].fillna(0)

    # Define relevance (rating >= 4 is relevant)
    merged["actual"] = merged["rating"].apply(lambda x: 1 if x >= 4 else 0)

    # Recommended items are predicted relevant
    merged["predicted"] = 1

    y_true = merged["actual"]
    y_pred = merged["predicted"]

    # --------------------------------
    # Standard Metrics
    # --------------------------------

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except:
        roc_auc = 0

    cm = confusion_matrix(y_true, y_pred)

    # --------------------------------
    # Recommender Metrics
    # --------------------------------

    precision_k = precision_at_k(y_true.values, y_pred.values, k)
    recall_k = recall_at_k(y_true.values, y_pred.values, k)

    # --------------------------------
    # Print Results
    # --------------------------------

    print("\nModel Evaluation")
    print("-----------------------------")

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    print(f"\nPrecision@{k}: {precision_k:.4f}")
    print(f"Recall@{k}: {recall_k:.4f}")

    print("\nConfusion Matrix:")
    print(cm)