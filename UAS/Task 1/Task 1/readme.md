# End-to-End Fraud Detection Pipeline
## Transaction Analysis Report

---

## Executive Summary

This project implements a comprehensive fraud detection system using machine learning to identify fraudulent transactions. The pipeline achieved a **95.11% ROC-AUC score** on validation data, demonstrating strong predictive performance for fraud detection.

**Key Metrics:**
- **ROC-AUC Score:** 0.9511 (Validation)
- **Precision-Recall AUC:** 0.7601
- **Fraud Detection Rate:** 51.8% recall at 0.5 threshold
- **False Positive Rate:** 0.14%

---

## Dataset Overview

### Data Characteristics
- **Training Set:** 590,540 transactions × 394 features
- **Test Set:** 506,691 transactions × 393 features
- **Target Variable:** `isFraud` (binary classification)
- **Class Distribution:** Highly imbalanced (3.50% fraud rate)
- **Imbalance Ratio:** 1:27.6 (fraud to non-fraud)

### Feature Categories
- **Numeric Features:** 380 columns
- **Categorical Features:** 14 columns
- **Transaction Details:** Amount, timestamp, product code
- **Card Information:** Card1-6 attributes
- **Address Data:** Addr1-2 fields
- **Email Domains:** Purchaser and recipient email domains
- **Anonymized Features:** V1-V339, C1-C14, D1-D15, M1-M9

### Data Quality Issues
- **High Missing Values:** Several features with >85% missing data
- **dist2:** 93.6% missing
- **D7, D13, D14:** ~89-93% missing
- **V-features (153, 149, 141, etc.):** ~86% missing

---

## Methodology

### 1. Data Preprocessing

#### Missing Value Treatment
- **Numeric columns:** Filled with -999 (missing indicator)
- **Categorical columns:** Filled with 'missing' string
- **High missing columns (>95%):** Removed from analysis

#### Feature Alignment
- Aligned train and test datasets to 392 common features
- Ensured consistent data types across datasets

### 2. Feature Engineering

Created 11 new features to capture transaction patterns:

#### Transaction Amount Features
- `TransactionAmt_log`: Log-transformed amount for better distribution
- `TransactionAmt_decimal`: Decimal portion (detects round numbers)
- `TransactionAmt_rounded`: Rounded to nearest 100

#### Time-Based Features
- `Transaction_hour`: Hour of day (0-23)
- `Transaction_day`: Day number from start
- `Transaction_day_of_week`: Day of week (0-6)
- `Transaction_is_weekend`: Weekend indicator

#### Interaction Features
- `card1_card2`: Card combination identifier
- `addr1_addr2`: Address combination identifier
- `P_emaildomain_isNull`: Missing email indicator
- `R_emaildomain_isNull`: Missing recipient email indicator

**Total Features After Engineering:** 403

### 3. Feature Selection

Applied Random Forest feature importance on 50,000-sample subset for computational efficiency.

**Top 10 Most Important Features:**
1. V201 (4.55%)
2. V257 (4.22%)
3. V242 (3.03%)
4. V258 (2.76%)
5. C1 (2.26%)
6. V45 (2.21%)
7. V189 (1.53%)
8. C7 (1.42%)
9. C4 (1.42%)
10. V244 (1.41%)

**Selected Features:** Top 150 features retained for modeling

### 4. Handling Class Imbalance

#### Strategy: SMOTE (Synthetic Minority Over-sampling Technique)
- **Sampling Strategy:** 0.3 (30% fraud rate target)
- **Original Train Size:** 472,432 transactions
- **Balanced Train Size:** 592,672 transactions
- **New Fraud Rate:** 23.08% (increased from 3.50%)

This approach created synthetic fraud examples to improve model learning without causing extreme overfitting.

### 5. Model Training

#### Algorithm: LightGBM (Gradient Boosting)

**Model Hyperparameters:**
- `objective`: binary classification
- `metric`: ROC-AUC
- `num_leaves`: 31
- `learning_rate`: 0.05
- `max_depth`: 10
- `feature_fraction`: 0.8 (80% features per tree)
- `bagging_fraction`: 0.8 (80% data per iteration)
- `min_child_samples`: 20

**Training Configuration:**
- **Number of Boosting Rounds:** 1000
- **Early Stopping:** 50 rounds
- **Final Best Iteration:** 1000
- **Training Time:** Several minutes on GPU

---

## Results & Performance Analysis

### Model Performance Metrics

#### ROC-AUC Scores
- **Training Set:** 0.9968
- **Validation Set:** 0.9511
- **Overfitting Gap:** 0.0457 (acceptable)

#### Precision-Recall Analysis
- **PR-AUC:** 0.7601
- Indicates good performance on imbalanced data

### Classification Performance (0.5 threshold)

#### Confusion Matrix
```
                Predicted
                No      Yes
Actual No     113,816   159
Actual Yes      1,990  2,143
```

#### Detailed Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Fraud | 0.98 | 1.00 | 0.99 | 113,975 |
| Fraud | 0.93 | 0.52 | 0.67 | 4,133 |
| **Accuracy** | | | **0.98** | **118,108** |

### Key Insights

1. **High Precision for Fraud (93%):** When model predicts fraud, it's correct 93% of the time
2. **Moderate Recall (52%):** Model catches about half of all fraud cases at 0.5 threshold
3. **Very Low False Positive Rate (0.14%):** Minimal false alarms for legitimate transactions
4. **Trade-off Consideration:** Lower threshold could improve recall but reduce precision

### Feature Importance in Final Model

**Top 20 Most Impactful Features:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | C12 | 516,099 |
| 2 | V279 | 369,316 |
| 3 | V294 | 331,515 |
| 4 | C14 | 311,286 |
| 5 | C8 | 214,800 |
| 6 | V317 | 204,507 |
| 7 | C11 | 201,814 |
| 8 | D2 | 166,609 |
| 9 | C1 | 159,806 |
| 10 | V280 | 154,812 |

**Observations:**
- C-features (counts/aggregates) are highly predictive
- V-features (anonymized behavioral signals) critical
- D-features (timedelta features) moderately important
- Engineered `TransactionAmt_decimal` ranked 15th

---

## Test Set Predictions

### Prediction Statistics

- **Total Predictions:** 506,691
- **Mean Fraud Probability:** 3.68%
- **Median Fraud Probability:** 0.76%
- **Predictions > 0.5 threshold:** 9,587 (1.89%)
- **Predictions > 0.1 threshold:** 30,072 (5.93%)

### Distribution Analysis
- **Minimum Score:** 0.000044
- **Maximum Score:** 0.999954
- Model successfully assigns very low scores to likely legitimate transactions
- Small percentage flagged as high-risk (appropriate for fraud detection)

---

## Technical Implementation

### Technologies Used
- **Python 3.x**
- **pandas & numpy:** Data manipulation
- **scikit-learn:** Preprocessing, metrics, train-test split
- **LightGBM:** Gradient boosting model
- **imbalanced-learn:** SMOTE implementation
- **matplotlib & seaborn:** Visualization
- **Google Colab:** Development environment

### Pipeline Structure

```
1. Data Download (gdown from Google Drive)
   ↓
2. Data Loading & Exploration
   ↓
3. Preprocessing & Cleaning
   ↓
4. Feature Engineering
   ↓
5. Categorical Encoding (Label Encoding)
   ↓
6. Feature Selection (Random Forest importance)
   ↓
7. Train-Val Split (80-20, stratified)
   ↓
8. SMOTE Balancing (train set only)
   ↓
9. LightGBM Training
   ↓
10. Model Evaluation
   ↓
11. Test Predictions & Submission
```

### Memory Optimization
- Removed original datasets after preprocessing
- Used garbage collection (`gc.collect()`) at key stages
- Selected top 150 features to reduce dimensionality
- Efficient LightGBM implementation

---

## Recommendations

### For Production Deployment

1. **Threshold Optimization**
   - Current 0.5 threshold provides high precision
   - Consider lowering to 0.2-0.3 for higher recall if false negatives are costly
   - Implement dynamic thresholding based on transaction amount

2. **Model Monitoring**
   - Track model performance over time
   - Monitor for concept drift in fraud patterns
   - Retrain quarterly or when performance degrades

3. **Feature Engineering Extensions**
   - Add velocity features (transactions per hour/day)
   - Incorporate device fingerprinting
   - Add geographic distance anomalies
   - Include historical user behavior patterns

4. **Ensemble Approach**
   - Combine LightGBM with other algorithms (XGBoost, Neural Networks)
   - Use stacking for improved robustness

### Business Applications

1. **Real-Time Fraud Prevention**
   - Deploy model as API endpoint
   - Flag high-risk transactions for manual review
   - Automatic blocking for scores > 0.8

2. **Risk Scoring System**
   - Low Risk (0-0.3): Auto-approve
   - Medium Risk (0.3-0.7): Additional verification
   - High Risk (0.7-1.0): Block and investigate

3. **Cost-Benefit Analysis**
   - At 52% recall: Catching ~$52 of every $100 in fraud
   - At 0.14% FPR: Minimal customer friction
   - Optimize threshold based on fraud losses vs. investigation costs

---

## Limitations & Future Work

### Current Limitations

1. **Feature Interpretability:** Many V-features are anonymized
2. **Temporal Validation:** No time-based split validation
3. **Recall Trade-off:** 52% recall means missing half of fraud cases
4. **Static Threshold:** Single threshold may not fit all scenarios

### Future Enhancements

1. **Deep Learning:** Explore LSTM/Transformer models for sequence patterns
2. **Graph Neural Networks:** Model transaction networks
3. **Explainable AI:** Implement SHAP values for prediction explanations
4. **Online Learning:** Continuous model updates with new data
5. **Multi-Stage Detection:** Combine rule-based and ML approaches

---

## Conclusion

This fraud detection pipeline demonstrates strong performance with a 95.11% ROC-AUC score, successfully balancing precision and recall for imbalanced fraud data. The system is ready for deployment with appropriate threshold calibration based on business requirements.

**Key Achievements:**
- ✅ Robust preprocessing pipeline handling missing data
- ✅ Effective feature engineering and selection
- ✅ Successful handling of severe class imbalance
- ✅ High-performance gradient boosting model
- ✅ Production-ready submission file generated

**Output:** `submission.csv` with 506,691 fraud probability predictions

---

## File Structure

```
project/
│
├── E2E_Fraud_Detection_Pipeline_Transaction_Analysis.ipynb
├── train_transaction.csv (651.69 MB)
├── test_transaction.csv (584.79 MB)
├── submission.csv (generated output)
└── README.md (this file)
```

---

**Project Completion Date:** December 2025  
**Model Version:** LightGBM v1.0  
**Status:** Ready for submission/deployment