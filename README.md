# 🛒 E-Commerce Recommendation System

A **Machine Learning–based E-Commerce Recommendation System** that recommends relevant products using multiple recommendation algorithms.

The system implements several recommendation techniques used in modern **e-commerce platforms** and allows users to generate recommendations through a **Command Line Interface (CLI)**.

Different recommendation models are implemented and evaluated using multiple **machine learning evaluation metrics** to compare their performance.

---

# 🚀 Features

* Implementation of multiple recommendation algorithms
* Modular **ML project architecture**
* Command Line Interface for generating recommendations
* Model performance evaluation using multiple metrics
* Comparison of multiple recommendation strategies
* Clean and scalable project structure

---

# 🤖 Recommendation Models Implemented

The system includes the following recommendation models:

### Content-Based Filtering

Recommends products based on similarity between product attributes such as category, description, and metadata.

### Collaborative Filtering (KNN)

Uses **user interaction patterns** to recommend products liked by similar users using the **K-Nearest Neighbors algorithm**.

### Hybrid Recommendation System

Combines **content-based and collaborative filtering** to improve recommendation accuracy.

### Popularity-Based Recommendation

Suggests products based on **overall popularity and rating scores**.

### Item Similarity Recommendation

Recommends products that are **similar to a given product** using item-to-item similarity.

### SVD Matrix Factorization

Uses **Singular Value Decomposition (SVD)** to learn latent relationships between users and items.

---

# 📊 Evaluation Metrics

Each recommendation model is evaluated using multiple metrics.

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

### Ranking Metrics

* Precision@K
* Recall@K

### Additional Evaluation

* Confusion Matrix

These metrics help compare different recommendation approaches and identify the most effective model.

---

# 📁 Project Structure

```text
ML_Project
│
├── data
│   ├── raw
│   │   ├── dataset_file_1.csv
│   │   └── dataset_file_2.csv
│   │
│   └── processed
│       └── cleaned_products.csv
│
├── experiments
│   └── (model experimentation and testing files)
│
├── notebooks
│   │
│   ├── amit
│   │   └── testing.ipynb
│   │
│   ├── collab
│   │   └── testing.ipynb
│   │
│   └── model_comparison.ipynb
│
├── src
│   │
│   ├── data
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   └── preprocessing.py
│   │
│   ├── evaluation
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   ├── recommender
│   │   ├── __init__.py
│   │   ├── content.py
│   │   ├── hybrid.py
│   │   ├── item_similarity.py
│   │   ├── knn_model.py
│   │   ├── popularity.py
│   │   └── svd.py
│   │
│   ├── utils
│   │   ├── __init__.py
│   │   └── helper.py
│   │
│   └── __init__.py
│
├── main.py
│
├── requirements.txt
│
├── .gitignore
│
├── LICENSE
│
└── README.md
```

---

# ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### Install dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

# ▶️ Running the Project

Run the system using:

```bash
python main.py
```

You will be prompted to choose a recommendation model:

```
Choose Recommendation Model

1 → Content Based Recommender
2 → KNN Collaborative Filtering
3 → Hybrid Recommender
4 → Popularity Recommender
5 → Item Similarity Recommender
6 → SVD Matrix Factorization
```

The system will generate product recommendations and display **evaluation metrics**.

---

# 📌 Example Output

```
Hybrid Recommendations:

Levi's Men's 550 Relaxed Fit Jeans | Rating: 4.79
Lee Men's Legendary Regular Straight Jean | Rating: 4.79
Wrangler Authentics Men's Classic Fit Jean | Rating: 4.79
```

Model performance metrics are also displayed.

---

# 👨‍💻 Contributors

### Amit

Responsible for:

* Content-Based Recommender
* KNN Collaborative Filtering
* Hybrid Recommender
* Evaluation Pipeline

### Shouri

Responsible for:

* Popularity Recommender
* Item Similarity Recommender
* SVD Recommender
* Project Documentation (README and documentation support)

---

# 🔮 Future Improvements

Potential enhancements:

* Web interface for recommendations
* Advanced ranking metrics (MAP, NDCG)
* Real-time recommendation pipeline
* API deployment using FastAPI or Flask
* Deep learning–based recommender systems

---

# 📜 License

This project is licensed under the terms specified in the **LICENSE** file.
