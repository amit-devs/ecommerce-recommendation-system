![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Recommender%20System-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)



# 🛒 E-Commerce Recommendation System

A **Recommendation System for E-Commerce platforms** that suggests relevant products using multiple recommendation algorithms.

The system implements commonly used recommendation techniques and allows users to generate recommendations through a **Command Line Interface (CLI)**.

These algorithms improve product discovery by analyzing product similarity, user interactions, and popularity trends.

---

## 🚀 Features

* Implementation of multiple recommendation algorithms
* Modular and scalable project architecture
* Command Line Interface (CLI) for generating recommendations
* Comparison of multiple recommendation strategies
* Clean and organized code structure

---

## 🤖 Recommendation Algorithms Implemented

### 🔹 Content-Based Filtering

Recommends products based on similarity between product attributes such as category, metadata, and text features.

### 🔹 Collaborative Filtering (KNN)

Uses **user-item interaction data** to recommend products based on similarity using the **K-Nearest Neighbors algorithm**.

### 🔹 Hybrid Recommendation System

Combines **content-based and collaborative filtering** to improve recommendation quality.

### 🔹 Popularity-Based Recommendation

Suggests products based on **overall popularity and weighted rating scores**.

### 🔹 Item Similarity Recommendation

Recommends products similar to a given product using **item-to-item similarity techniques**.

### 🔹 SVD Matrix Factorization

Uses **Singular Value Decomposition (SVD)** to learn hidden relationships between users and products.

---

## 📊 Evaluation Metrics

Each recommendation algorithm is evaluated using ranking-based metrics:

### 🔹 Ranking Metrics

* Precision@K
* Recall@K

### 🔹 Additional Evaluation

* Confusion Matrix *(for analysis purposes)*

These metrics help compare different recommendation approaches and determine their effectiveness.

---

## 📁 Project Structure

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
│   └── (experiments and testing files)
│
├── notebooks
│   ├── amit
│   │   └── testing.ipynb
│   │
│   ├── shouri
│   │   └── testing.ipynb
│   │
│   └── model_comparison.ipynb
│
├── src
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
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

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

## ▶️ Running the Project

Run the system using:

```bash
python main.py
```

You will be prompted to choose a recommendation algorithm:

```
Choose Recommendation Algorithm

1 → Content Based Recommender
2 → KNN Collaborative Filtering
3 → Hybrid Recommender
4 → Popularity Recommender
5 → Item Similarity Recommender
6 → SVD Matrix Factorization
```

The system will generate product recommendations.

---

## 📌 Example Output

```
Hybrid Recommendations:

Levi's Men's 550 Relaxed Fit Jeans | Rating: 4.79
Lee Men's Legendary Regular Straight Jean | Rating: 4.79
Wrangler Authentics Men's Classic Fit Jean | Rating: 4.79
```

---

## 👨‍💻 Contributors

### Amit

* Content-Based Recommender
* KNN Collaborative Filtering
* Hybrid Recommender
* Evaluation Pipeline

### Shouri

* Popularity Recommender
* Item Similarity Recommender
* SVD Recommender
* Project Documentation

---

## 🔮 Future Improvements

* Web interface for recommendations
* Advanced ranking metrics (MAP, NDCG)
* Real-time recommendation system
* API deployment using FastAPI or Flask
* Deep learning–based recommender systems

---

## 📜 License

This project is licensed under the terms specified in the **LICENSE** file.
