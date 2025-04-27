# 🏦 Income Prediction Using Machine Learning

## Overview
This project applies multiple machine learning models to predict whether an individual earns more than \$50,000 per year based on demographic data from the U.S. Census Bureau.  
The goal was to explore and compare different supervised learning methods for this binary classification problem.

---

## Key Accomplishments
- **Performed comprehensive data cleaning and feature engineering** to handle missing values, encode categorical variables, and scale numerical features.
- **Conducted exploratory data analysis (EDA)** to uncover trends between demographics and income levels.
- **Built and tuned multiple classification models:**
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Logistic Regression
  - Support Vector Machine (SVM)
- **Achieved high model performance**, with Random Forest achieving the highest accuracy (~85.6%) among the models tested.
- **Benchmarked results** against models built by external sources (Kaggle, UCSD studies) for cross-validation.

---

## Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **Scikit-learn** (RandomForestClassifier, KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression, SVC)
- **Data Preprocessing:** Label Encoding, Standard Scaling
- **Model Evaluation Metrics:** Accuracy Score, Visual Comparison

---

## Project Structure
| Module | Description |
|:------|:------------|
| `EDA` | Visual exploration of demographic features vs income (bar charts, histograms, correlation heatmaps). |
| `Data Cleaning` | Replaced missing values, combined categories, label encoded categorical variables. |
| `Feature Engineering` | Simplified education levels, created age groups. |
| `Model Training & Evaluation` | Built Random Forest, KNN, Decision Tree, Logistic Regression, and SVM models; compared accuracy. |
| `Benchmarking` | Compared results to Kaggle and UCSD study benchmarks. |

---

## Results
| Model | Best Accuracy | Notes |
|:-----|:--------------|:------|
| **Random Forest** | **85.6%** | Best performer overall |
| **KNN (Manhattan, 13 neighbors)** | 83.1% | Close second |
| **Decision Tree (Max Depth 10)** | 84.7% | Fast and interpretable |
| **SVM (C=10, RBF Kernel, gamma=0.1)** | 83% | High but computationally expensive |
| **Logistic Regression** | 79% | Significantly lower than tree-based methods |

- **Key insights:** Higher income correlated with being male, married, having higher education, and working private sector jobs.
- **Best models for this dataset:** Random Forest, KNN, Decision Tree.

---



