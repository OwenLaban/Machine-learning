# Machine Learning Coursework, UTS & UAS – Owen

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Tools-Google_Colab-orange?style=for-the-badge&logo=googlecolab&logoColor=white)

## 👤 Student Identification
> This repository is submitted as part of the Machine Learning & Deep Learning coursework.

* **Name:** Josua Owen Fernandi Silaban  
* **Class:** TK-46-04  
* **NIM:** 1103223117  

---

## 📌 Repository Purpose

This repository documents my learning journey in **Machine Learning** and **Deep Learning** during the course.  
It is organized to show my progress from basic data processing to building complete, end-to-end ML/DL pipelines for:

1. **Supervised Learning**
   * Regression – predicting continuous targets (e.g. song release year).
   * Classification – fraud detection on transaction data.
2. **Unsupervised Learning**
   * Customer clustering and representation learning using autoencoders.
3. **Model Evaluation & Interpretation**
   * Using appropriate metrics (MSE, RMSE, R², AUC, F1-score, Silhouette, etc.).
   * Visualizing model performance and feature importance.

The repository contains:

- **Weekly Assignments** – small, focused exercises.  
- **UTS Projects** – 3 end-to-end midterm projects.  
- **UAS Tasks** – 3 final exam notebooks (`Task 1.ipynb`, `Task 2.ipynb`, `Task 3.ipynb`) that consolidate and refine the UTS work.

---

## 📚 Weekly Assignments

The `Weekly_Assignments` folder (or equivalent) summarizes my step-by-step learning:

- **Chapter 1–3**
  - Python refresher, NumPy arrays, Pandas DataFrame operations.
  - Basic visualizations: line plots, histograms, boxplots.

- **Chapter 4–6**
  - Data cleaning (missing values, outliers, scaling).
  - Intro to supervised learning: Linear Regression, Logistic Regression, KNN, basic model metrics.

- **Chapter 7–8**
  - Train/validation/test split, cross-validation.
  - Model tuning (GridSearch / RandomizedSearch).
  - Handling imbalanced datasets and basic feature engineering.

> Exact notebook names may differ depending on the course template used.

---

## 📂 UTS – End-to-End Midterm Projects

The `UTS` folder contains **three main end-to-end projects** based on the official mid-term instructions.

### 1. 🕵️‍♂️ End-To-End Fraud Detection (Classification)

* **File (UTS):** `UTS_Fraud_Detection.ipynb`  
* **Dataset:** `train_transaction.csv`, `test_transaction.csv` (with `isFraud` label in train).  
* **Objective:**  
  Build a classifier that predicts the probability that a transaction is fraudulent (`isFraud = 1`), while handling **class imbalance**.

**Main Steps**

- **Data Loading & Cleaning**
  - Load train & test transaction data from Kaggle-like dataset.
  - Select a subset of informative features (e.g. `TransactionAmt`, `ProductCD`, `card4`, `card6`, `addr1`, `dist1`, etc.).
  - Handle missing values using `SimpleImputer` (median/most_frequent).

- **Preprocessing Pipeline**
  - Separate **numeric** and **categorical** features.
  - Numeric: median imputation + `StandardScaler`.
  - Categorical: most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`.
  - Combine using `ColumnTransformer`.

- **Models Explored**
  - **Logistic Regression** with `class_weight='balanced'`.
  - **Random Forest Classifier** with `class_weight='balanced'`.

- **Evaluation**
  - Train/validation split with `stratify=y`.
  - Metrics:
    - **ROC-AUC** (main metric).
    - **Classification report** (Precision, Recall, F1).
    - **Confusion Matrix** to inspect False Positive / False Negative.
    - ROC Curve plots and class distribution plot to show imbalance.

- **Prediction & Submission**
  - Apply the same preprocessing pipeline to `test_transaction.csv`.
  - Use the best model to generate `isFraud` probabilities.
  - Save as `submission_fraud.csv` with columns: `TransactionID`, `isFraud`.

---

### 2. 📈 End-To-End Regression Pipeline (Song Year Prediction)

* **File (UTS):** `UTS_Regression_Pipeline.ipynb`  
* **Dataset:** `midterm-regresi-dataset.csv` (first column = target year, remaining columns = audio features).  
* **Objective:**  
  Predict the **release year** of a song based on its numerical audio features.

**Main Steps**

- **Data Understanding**
  - Load dataset without header; column 0 used as `year` target.
  - Rename features as `feature_1`, `feature_2`, … for clarity.
  - (Optional) **Row sampling** and `float32` casting to reduce RAM usage.

- **Preprocessing**
  - Clip extreme values (1%–99%) to reduce the effect of outliers.
  - Split into train/test sets.
  - Apply `StandardScaler` for all numeric features.

- **Models Explored**
  - **Linear Regression** as baseline.
  - **Random Forest Regressor** as a non-linear model.
  - (Optional) **Deep Learning Regressor** using a small Dense Neural Network in TensorFlow/Keras.

- **Evaluation Metrics**
  - **MSE** (Mean Squared Error).
  - **RMSE** (Root Mean Squared Error).
  - **MAE** (Mean Absolute Error).
  - **R² Score** (coefficient of determination).

- **Visualization**
  - Plot **True vs Predicted year** for the best model.
  - Distribution of target `year` to understand the prediction difficulty (older vs newer songs).

---

### 3. 👥 Customer Clustering with Autoencoder (Unsupervised)

* **File (UTS):** `UTS_Customer_Clustering.ipynb`  
* **Dataset:** `clusteringmidterm.csv` (credit card usage & payment behavior).  
* **Objective:**  
  Segment customers into groups based on their transaction behavior, using a combination of **Autoencoder** and **K-Means Clustering**.

**Main Steps**

- **Preprocessing**
  - Drop `CUST_ID`.
  - Handle missing values (median filling).
  - Standardize all numeric features (`StandardScaler`, `float32` for memory efficiency).
  - (Optional) Row sampling for Colab RAM limitation.

- **Representation Learning (Autoencoder)**
  - Build a symmetric autoencoder with a low-dimensional latent space (e.g. 4–8 neurons).
  - Train to reconstruct input features using MSE loss.
  - Extract the **latent representation** (bottleneck layer) for each customer.

- **Clustering**
  - Run **K-Means** on latent features for various `k`.
  - Use **Elbow method** and **Silhouette Score** to select the best number of clusters.
  - Assign cluster labels back to customers.

- **Analysis**
  - Compute mean statistics per cluster (average balance, purchase frequency, cash advance, etc.).
  - Interpret business meaning of each cluster (e.g., “high spender”, “cash advance heavy user”, “low usage customer”).

---

## 🧪 UAS – Final Exam Tasks

The **UAS** part of the course is implemented in **three consolidated notebooks**:

- `Task 1.ipynb`  
- `Task 2.ipynb`  
- `Task 3.ipynb`  

These notebooks reuse the same three themes as UTS (classification, regression, clustering) but are written in a more compact, RAM-efficient, and deployment-oriented style suitable for Google Colab (12 GB RAM).

### 🔹 UAS – Task 1: Extreme-Light Fraud Detection (Classification)

* **File (UAS):** `Task 1.ipynb`  
* **Dataset:** `train_transaction.csv`, `test_transaction.csv`.  
* **Focus:** Build a **lightweight fraud detection pipeline** that can run reliably in Colab with limited RAM.

**Highlights**

- Heavy use of:
  - **Row sampling** (limit number of training rows).
  - **Numeric-only subset** of features.
  - `StandardScaler` with `float32` casting.
- Models:
  - **Logistic Regression** and **Random Forest** as compact baselines.
  - Optional **ANN (MLP)** with small hidden layers.
- Metrics and outputs:
  - ROC-AUC, confusion matrix, classification report.
  - Final `submission_fraud_extreme_small_*.csv` file for deployment.

---

### 🔹 UAS – Task 2: Regression on Audio Features

* **File (UAS):** `Task 2.ipynb`  
* **Dataset:** `midterm-regresi-dataset.csv`.  
* **Focus:** Implement a **clean, end-to-end regression pipeline** under memory constraints.

**Highlights**

- Treat the first column as the `year` target, others as `feature_1..N`.
- Use `StandardScaler` and (optionally) sampling to control memory usage.
- Models:
  - Baseline (mean year),
  - **Linear Regression**,
  - **Ridge Regression** (several alphas),
  - **Random Forest Regressor** with moderate size,
  - Optional small **MLP regressor**.
- Evaluate using **MSE, RMSE, MAE, R²** and compare models in a single summary table.

---

### 🔹 UAS – Task 3: Customer Clustering & Autoencoder Refinement

* **File (UAS):** `Task 3.ipynb`  
* **Dataset:** `clusteringmidterm.csv`.  
* **Focus:** Build a **compact customer clustering notebook** that refines the UTS work.

**Highlights**

- Standardize features, remove `CUST_ID`, and convert to `float32`.
- Train a small **Autoencoder** to obtain latent features.
- Run **K-Means** on both:
  - the original scaled features, and
  - the autoencoder latent space.
- Compare:
  - **Silhouette Score** and other cluster metrics,
  - Cluster size distribution and average behavior per cluster.
- Summarize business interpretation of each segment.

---

## 🚀 How to Run the Notebooks

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/owenlaban/Machine-Learning.git
   cd Machine-Learning
